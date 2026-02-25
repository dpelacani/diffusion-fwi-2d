import numpy as np
import os
from stride import *
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from diffusionModel.models import UNetAttnMoreD
from diffusionModel.diffusion import DiffusionProcess
import torch
import torch.nn.functional as F

def preprocess_vp(tensor):
    """
    Helper function to preprocess the velocity model tensor.
    The preprocessing includes resizing, normalization, and logarithmic transformation.
    Because the velocity model from FWI has different size and range compared to the training data of the diffusion model,
    we need to preprocess it before feeding it into the diffusion model. Here we use the same preprocessing as in the training data.

    Args:
        tensor: Input velocity model tensor.

    Returns:
        Preprocessed tensor.
    """
    if tensor.dim() > 2:
        tensor = tensor.squeeze()
    # Resize tensor to 128*128
    tensor = F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=True)
    
    # normalization
    tensor = tensor / 3000.0
    tensor = torch.clamp(tensor, min=1e-6)
    tensor = torch.log(tensor) + 1.0

    # Normalize to [-1, 1]
    tensor = (tensor - 0.5) / 0.5
    return tensor

def postprocess_vp(tensor):
    """
    Helper function to postprocess the velocity model tensor.
    The postprocessing includes denormalization, exponential transformation, and resizing back to original size.
    This is the inverse operation of the preprocessing function.

    Args:
        tensor: Input velocity model tensor.

    Returns:
        Postprocessed tensor.
    """
    tensor = tensor * 0.5 + 0.5

    tensor = torch.exp(tensor - 1.0)

    tensor = tensor * 3000.0
    # Resize back to original shape
    tensor = F.interpolate(tensor, size=(320, 256), mode='bilinear', align_corners=True)
    tensor = tensor.squeeze(0).squeeze(0)  # Remove batch and channel
    return tensor

class DummyGradientOperation(Operator):

    def forward(self, grad, **kwargs):
        # `grad.data` is a Numpy array, and can be operated on as needed
        grad_data = grad.data
        grad_data = grad_data + 0.0
        # this method should return a new copy of the array
        new_grad = grad.alike(data=grad_data)
        return new_grad

class DDPMVpUpdate:
    """
    Class to denoise the velocity model using a pre-trained diffusion model.
    """
    def __init__(self, vp, model, diffusion):
        """ Initialize the DDPMVpUpdate class."""
        super().__init__()
        # the DDPM model
        self.model = model
        # the device to run the model on
        self.device = next(model.parameters()).device
        # the diffusion process
        self.diffusion = diffusion
        # the maximum number of iterations that we want to use for the timestep schedule
        self.max_iter = 96
        # the current diffusion state, in our case the denoised velocity model
        self.diffusion_state = None
        # the number of diffusion steps to use, this is then adjusted based on the timestep schedule later
        self.steps = 10

    def update_vp(self, vp, iteration):
        vp_data = vp.data
        # Convert both vp and grad to tensors on the same device
        vp_tensor = torch.tensor(vp_data, device=self.device, dtype=torch.float32)
        
        # Preprocess vp for diffusion model input
        vp_preprocessed = preprocess_vp(vp_tensor)
        # Schedule t based on the current iteration, gradually reducing t as FWI iterations progress
        t = self.schedule_t(iteration, self.max_iter)
        # update the number of diffusion steps to use
        self.steps = t

        t = torch.tensor(t, device=self.device).unsqueeze(0)
        e = torch.randn_like(vp_preprocessed)
        # we add the noise corresponding to timestep t, and then denoise it back to timestep 0
        # perform forward diffusion to add noise
        x_t = self.diffusion.forward_diffusion(vp_preprocessed, t, e)

        # perform reverse diffusion to denoise
        self.diffusion_state = self.diffusion_sample(
                x_t, t, self.steps
            )
        
        # postprocess the denoised velocity model back to original format
        updated_vp = postprocess_vp(self.diffusion_state)

        # update the vp data
        new_vp = vp.alike()
        new_vp.data[:] = updated_vp.cpu().numpy()
        return new_vp

    def diffusion_sample(self, x, t_start, num_steps):
        self.model.eval()
        
        # send to the GPU
        x = x.to(self.device)

        # Context management to ensure gradients are not tracked for any torch tensors or parameters
        # since we are only interested in sampling, not training
        with torch.no_grad():
            start_t = t_start - num_steps
            if start_t < 0:
                start_t = 0
            for t in reversed(range(start_t, t_start)):

                # sample z from a normal distribution if condition met
                if t > 0:
                    z = torch.randn_like(x)
                else:
                    z = 0
        
                # convert t to a tensor, send to the GPU, 
                # and expand its first dimension to be equal to batch size
                t_tensor = torch.tensor(t).to(self.device).expand(x.shape[0])

                # denoise x
                x = self.diffusion.reverse_diffusion(x, t_tensor, self.model(x, t_tensor), z)
                # clipping to ensure the sample values remain in the range -0.4172 to 1
                # but only after the last step
                if t == 0:
                    x = torch.clamp(x, -0.4172, 1)
        return x
    
    def schedule_t(self, iteration, max_iter):
        # a simple linear schedule from t=60 to t=5
        t_max = 60
        t_min = 5
        t = int(t_max - (t_max - t_min) * (iteration / max_iter))
        return max(t_min, t)
    
    

async def main(runtime):
    np.random.seed(12345)

    # Create the grid
    # specify the discretisation to be used
    shape = (320, 256)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (0.5e-3, 0.5e-3)

    space = Space(shape=shape,
                  extra=extra,
                  absorbing=absorbing,
                  spacing=spacing)

    start = 0.
    step = 0.08e-6
    num = 2500

    time = Time(start=start,
                step=step,
                num=num)

    # Create problem
    problem = Problem(name='anastasio2D',
                      space=space, time=time)

    # Create medium
    # this is the speed of sound of the region of interest
    # vp contains a Numpy array with the velocity for every point on the grid
    vp = ScalarField.parameter(name='vp',
                               grid=problem.grid, needs_grad=True)
    vp.fill(1500.)

    problem.medium.add(vp)

    # Create transducers
    # we generally assume point transducers in simulation, but
    # in reality more complex transducer geometries are used
    problem.transducers.default()

    # Create geometry
    # this will make a ring of 256 transducers around the region of interest
    num_locations = 256
    problem.geometry.default('elliptical', num_locations, radius=((space.limit[0] - 7.e-3) / 2,
                              (space.limit[1] - 5.e-3) / 2))

    # Create acquisitions
    # load the simulated observed file generated by the forward script
    problem.acquisitions.load(path=problem.output_folder,
                              project_name=problem.name, version=0)

    # Plot
    # some plotting of what we have built so far
    problem.plot()

    # Create the PDE
    # the physics of our problem are represented by the iso-acoustic wave equation
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Create loss
    # there are potentially multiple loss functions we could use to run FWI,
    # but we will stick with the L2 Norm for now
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    # Create optimiser
    # to run the optimisation, we will use gradient descent with a fixed step size
    step_size = 5

    # Load the diffusion model
    # Determine device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load the model
    diffusion_model = UNetAttnMoreD(in_channels=1, out_channels=1).to(device)
    # here you may need to change the path to the model
    absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    diffusion_model_path = os.path.join(absolute_path, 'diffusion model/model_added_res.pt')
    diffusion_model.load_state_dict(torch.load(diffusion_model_path))
    diffusion_model.eval()  # Set to evaluation mode
    
    # initialise diffusion process
    T = 1000
    diffusion = DiffusionProcess(device, T)
    diffusion.set_variables()
    
    # we can use this syntax to do operations on the FWI gradient at every iteration
    process_grad = ProcessGlobalGradient()
    process_grad.append('dummy_gradient_operation', DummyGradientOperation())

    # add a box constraint: vp should be between 1480 and 3000 m/s(skull has higher velocity)
    process_model = ProcessModelIteration(min=1480., max=3000.)

    # create the optimiser
    optimiser = GradientDescent(vp, step_size=step_size,
                                process_grad=process_grad,
                                process_model=process_model,
                                dump_grad=True)

    # Run optimisation
    # an optimisation loop is a fancy wrapper for a `for` loop
    optimisation_loop = OptimisationLoop()

    # FWI is a non-linear, non-convex optimisation problem, so to ensure convergence,
    # we start the inversion at low frequencies and add higher frequencies progressively
    # we call each of these frequency bands a block, and we run each block for a fixed
    # number of iterations
    max_freqs = [0.1e6, 0.2e6, 0.3e6, 0.35e6]
    num_blocks = len(max_freqs)

    # we can define how many iterations to run in each block, here we do more iterations
    # for the lower frequency blocks as they are more likely to help with convergence. The last
    # block has the highest frequency and we run it slightly longer to get more details.
    block_iters = [40, 24, 24, 32]
    
    # running all shots in every iteration would be too costly,
    # so we run a subset of them (a batch)
    num_shots_per_iter = 32
    
    # create the diffusion updater for later denoising
    diffusion_updater = DDPMVpUpdate(vp, diffusion_model, diffusion)
    for i, (block, freq) in enumerate(optimisation_loop.blocks(num_blocks, max_freqs)):
        num_iters = block_iters[i]
        # prepare some initial configuration for the block
        operators = init_block(problem, optimiser,
                               f_max=freq, max_freqs=max_freqs)

        # inside the block, we run for a specific number of iterations
        for iteration in block.iterations(num_iters):
            # run one single iteration of FWI
            # vp.data will have been updated at the end of this call
            await run_iteration(problem, pde, loss,
                                optimisation_loop, optimiser, operators, vp,
                                select_shots=dict(num=num_shots_per_iter, randomly=True),
                                f_max=freq, max_freqs=max_freqs, iteration=iteration,
                                kernel='OT4', platform='nvidia-acc')
            # get the current iteration number
            current_iter = iteration.abs_id
            # decide whether to use the diffusion model to update vp
            # we use the diffusion model every 4 iterations between iteration 24 and 80
            # and then every 8 iterations between iteration 80 and 100
            use_diffusion = ((current_iter%4==0) and (current_iter > 24) and (current_iter < 80)
                             ) or (current_iter >= 80 and current_iter < 100 and current_iter%8==0)
            # using the diffusion model to update vp
            if diffusion_updater is not None and use_diffusion:
                vp = optimiser.variable
                # Update vp using diffusion model
                updated_vp = diffusion_updater.update_vp(vp, current_iter)
                # set the updated vp back to the optimiser variable
                optimiser.variable.data[:] = updated_vp.data

            # save vp to disk after this iteration
            optimiser.dump(path=problem.output_folder,
                           project_name=problem.name,
                           version=iteration.abs_id + 1)

    vp.plot()


def init_block(problem, optimiser, **kwargs):
    """

    Parameters
    ----------
    problem : Problem
        Problem to run the optimisation on.
    optimiser : LocalOptimiser
        Local optimiser associated with the variable for which we are inverting.
    f_min : float, optional
        Min. frequency to filter wavelets and data with. If not given, no high-pass filter
        is applied.
    f_max : float, optional
        Max. frequency to filter wavelets and data with. If not given, no low-pass filter
        is applied.

    Returns
    -------
    list
        List of operators created for the current block.

    """
    runtime = mosaic.runtime()

    f_min = kwargs.pop('f_min', None)
    f_max = kwargs.pop('f_max', None)

    filter_wavelets_relaxation = 0.75
    filter_traces_relaxation = 0.75
    process_wavelets = ProcessWavelets.remote(f_min=f_min, f_max=f_max,
                                              filter_traces=True,
                                              filter_relaxation=filter_wavelets_relaxation,
                                              len=runtime.num_workers, **kwargs)
    process_observed = ProcessObserved.remote(f_min=f_min, f_max=f_max,
                                              filter_traces=True,
                                              filter_relaxation=filter_wavelets_relaxation,
                                              len=runtime.num_workers, **kwargs)
    process_wavelets_observed = ProcessWaveletsObserved.remote(f_min=f_min, f_max=f_max,
                                                               filter_traces=True,
                                                               len=runtime.num_workers, **kwargs)
    process_traces = ProcessTraces.remote(f_min=f_min, f_max=f_max,
                                          filter_traces=True,
                                          filter_relaxation=filter_traces_relaxation,
                                          len=runtime.num_workers, **kwargs)

    problem.acquisitions.reset_selection()

    if optimiser.reset_block:
        optimiser.reset()

    return process_wavelets, process_observed, process_wavelets_observed, process_traces


async def run_iteration(problem, pde, loss, optimisation_loop, optimiser, operators, *args, **kwargs):
    """
    Use a ``problem`` forward using a given ``pde``. The given ``args`` and ``kwargs``
    will be passed on to the PDE.

    Parameters
    ----------
    problem : Problem
        Problem to run the optimisation on.
    pde : Operator
        PDE operator to run for each shot in the problem.
    loss : Operator
        Loss function operator.
    optimisation_loop : OptimisationLoop
        Optimisation loop.
    optimiser : LocalOptimiser
        Local optimiser associated with the variable for which we are inverting.
    operators : list
        List of operators already created for the current block.
    select_shots : dict, optional
        Rules for selecting available shots per iteration, defaults to taking all shots. For
        details on this see :func:`~stride.problem.acquisitions.Acquisitions.select_shot_ids`.
    f_min : float, optional
        Min. frequency to filter wavelets and data with. If not given, no high-pass filter
        is applied.
    f_max : float, optional
        Max. frequency to filter wavelets and data with. If not given, no low-pass filter
        is applied.
    iteration : Iteration
        Current iteration object
    args : optional
        Extra positional arguments for the operators.
    kwargs : optional
        Extra keyword arguments for the operators.

    Returns
    -------

    """
    logger = mosaic.logger()
    runtime = mosaic.runtime()

    select_shots = kwargs.pop('select_shots', {})
    iteration = kwargs.pop('iteration', None)
    block = iteration.block

    # gather operators already created
    process_wavelets, process_observed, process_wavelets_observed, process_traces = operators

    # prepare GPU configuration
    platform = kwargs.get('platform', 'cpu')
    using_gpu = 'nvidia' in platform or 'gpu' in platform
    if using_gpu:
        devices = kwargs.pop('devices', None)
        num_gpus = mosaic.utils.gpu_count() if devices is None else len(devices)
        devices = list(range(num_gpus)) if devices is None else devices

    # clear gradient and reset optimiser if needed
    optimiser.clear_grad()
    if optimiser.reset_iteration:
        optimiser.reset()

    # pre-distribute relevant parameters to parallel processes
    published_args = [runtime.put(each, publish=True) for each in args]
    published_args = await asyncio.gather(*published_args)

    logger.perf('Starting iteration %d (out of %d), '
                'block %d (out of %d)' %
                (iteration.id+1, block.num_iterations, block.id+1,
                 optimisation_loop.num_blocks))

    # select a subset of all shots (a batch)
    shot_ids = problem.acquisitions.select_shot_ids(**select_shots)
    num_shots = len(shot_ids)
    shot_dir = problem.output_folder
    iter_shot_file = os.path.join(shot_dir, f'shotids-iter-{iteration.abs_id+1}.npy')
    np.save(iter_shot_file, shot_ids)

    # another fancy for loop that runs in parallel
    @runtime.async_for(shot_ids, safe=False)
    async def loop(worker, shot_id):
        _kwargs = kwargs.copy()

        logger.perf('\n')
        logger.perf('Giving shot %d to %s (%d out of %d)'
                    % (shot_id, worker.uid,
                       iteration.num_submitted, num_shots))

        sub_problem = problem.sub_problem(shot_id)
        iteration.add_submitted(sub_problem.shot)
        wavelets = sub_problem.shot.wavelets
        observed = sub_problem.shot.observed

        if wavelets is None:
            raise RuntimeError('Shot %d has no wavelet data' % shot_id)

        if observed is None:
            raise RuntimeError('Shot %d has no observed data' % shot_id)

        if using_gpu:
            deviceid = devices[worker.indices[1] % num_gpus]
            if platform in ['nvidia-acc', 'nvidia-cuda']:
                devito_args = _kwargs.get('devito_args', {}).copy()
                devito_args['deviceid'] = deviceid
                _kwargs['devito_args'] = devito_args
            elif platform == 'gpu':
                _kwargs['deviceid'] = deviceid
            else:
                raise ValueError('Unknown platform %s' % platform)

        # pre-process wavelets and observed traces
        wavelets = process_wavelets(wavelets,
                                    iteration=iteration, problem=sub_problem,
                                    runtime=worker, **_kwargs)
        observed = process_observed(observed,
                                    iteration=iteration, problem=sub_problem,
                                    runtime=worker, **_kwargs)
        processed = process_wavelets_observed(wavelets, observed,
                                              iteration=iteration, problem=sub_problem,
                                              runtime=worker, **_kwargs)
        wavelets = processed.outputs[0]
        observed = processed.outputs[1]

        # run PDE
        modelled = pde(wavelets, *published_args,
                       iteration=iteration, problem=sub_problem,
                       runtime=worker, **_kwargs)

        # post-process modelled and observed traces
        scale_to = sub_problem.shot.observed
        traces = process_traces(modelled, observed,
                                scale_to=scale_to,
                                iteration=iteration, problem=sub_problem,
                                runtime=worker, **_kwargs)
        modelled = traces.outputs[0]
        observed = traces.outputs[1]

        # calculate loss
        fun = loss(modelled, observed,
                   keep_residual=False,
                   iteration=iteration, problem=sub_problem,
                   runtime=worker, **_kwargs)

        # run adjoint
        fun_value = await fun.remote.adjoint(**_kwargs).result()

        iteration.add_loss(fun_value)
        logger.perf('Functional value for shot %d: %s' % (shot_id, fun_value))

        iteration.add_completed(sub_problem.shot)
        logger.perf('Retrieved gradient for shot %d (%d out of %d)'
                    % (sub_problem.shot_id,
                       iteration.num_completed, num_shots))

    await loop

    # take a step of gradient descent
    f_min = kwargs.pop('f_min', None)
    f_max = kwargs.pop('f_max', None)
    
    await optimiser.step(iteration=iteration, problem=problem,
                         f_min=f_min, f_max=f_max,
                         filter_relaxation=0.75,
                         **kwargs)

    if iteration.prev_run is not None:
        prev_loss = iteration.prev_run.total_loss
        prev_loss = ' [previous loss %e]' % prev_loss
    else:
        prev_loss = ''

    iter_loss = os.path.join(problem.output_folder, f'loss_iter-{iteration.abs_id+1}.npy')
    np.save(iter_loss, iteration.total_loss)

    logger.perf('Done iteration %d (out of %d), '
                'block %d (out of %d) - Total loss %e%s' %
                (iteration.id+1, block.num_iterations, block.id+1,
                 optimisation_loop.num_blocks, iteration.total_loss, prev_loss))
    logger.perf('====================================================================')

    iteration.clear_run()


if __name__ == '__main__':
    mosaic.run(main)
