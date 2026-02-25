import torch
import numpy as np

from stride import *  

from typing import Optional

from .pipelines import DiffusionFWIPipeline, DataProcessingPipeline
from ..diffusion.diffusionProcess import DiffusionProcess

import logging


class DiffusionVpOperator(Operator):
    """Diffusion-based velocity model operator for FWI.

    Args:
        x_dim (int): Dimension of the velocity model volume (assumed cubic).
        total_iterations (int): Total number of iterations for the optimisation.
        diffusion_model (torch.nn.Module): Pre-trained diffusion model for FWI.
        mask (np.ndarray, optional): Optional mask to apply during blending.
        update_fn (callable, optional): Optional function to modify the velocity model
            update after diffusion. Should take in the current velocity model and
            return a modified velocity model.
        scheduling_args (dict, optional): Optional dictionary of arguments for scheduling 
            the interleaving scheme. Should contain keys 'iters_to_run', 't_starts', and 'alphas'
            representing the iteration numbers to run diffusion, the corresponding t_start values,
            and alpha values passed to update_fn. All should be lists of the same length.
        seed (int, optional): Random seed for diffusion sampling to ensure reproducibility.
        device (str, optional): Device to run the diffusion model on ("cpu" or "cuda").
    """

    def __init__(
        self,
        input_dim: int,
        original_dim: Optional[tuple],
        total_iterations: int,
        diffusion_model: torch.nn.Module,
        diffusion_process: DiffusionProcess,
        mask: Optional[np.ndarray] = None,
        update_fn: Optional[callable] = None,
        scheduling_args: Optional[dict] = None,
        init_seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__(name="diffusion_vp_operator")

        # Store parameters
        self.init_seed = init_seed
        self.device = device
        self.diffusion_model = diffusion_model.to(device)
        self.mask = mask

        # Instantiate data preprocessing pipeline
        self.preproc = DataProcessingPipeline(x_dim=input_dim, original_shape=original_dim)

        # Instantiate diffusion FWI pipeline
        self.pipeline = DiffusionFWIPipeline(
            diffusion_model=self.diffusion_model,
            diffusion_process=diffusion_process,
            data_pipeline=self.preproc,
            update_fn=update_fn,
            device=self.device,
        )
        logging.info("(diffusionfwi) Diffusion FWI Pipeline initialized with device: {}".format(device))

        # Set up interleaving scheduling
        self.set_scheduling(total_iterations=total_iterations, **(scheduling_args or {}))

        # Initialise iteration counter 
        self.iteration = 0

    def set_scheduling(
        self,
        total_iterations: Optional[int] = None,
        iters_to_run: Optional[list] = None,
        t_starts: Optional[list] = None,
        alphas: Optional[list] = None,
    ):
        """
        Set up the scheduling for diffusion FWI iterations.
        Either provide total_iterations to use default scheduling, or provide
        all three schedulers (iters_to_run, t_starts, alphas).
        Args:
            total_iterations (int): Total number of iterations for the optimisation.
            iters_to_run (list): List of iteration numbers to run diffusion.
            t_starts (list): List of t_start values for diffusion,
                with order and length matching to iters_to_run.
            alphas (list): List of alpha values for diffusion,
                with order and length matching to iters_to_run.
        """

        if total_iterations is None and (
            t_starts is None or iters_to_run is None or alphas is None
        ):
            raise ValueError(
                "Either total_iterations or all schedulers must be provided."
            )

        if iters_to_run is None:
            # Run every 4 iterations
            iters_to_run = [i for i in range(8, total_iterations * 2 + 1, 8)]

        if t_starts is None:
            t_starts = np.linspace(300, 100, len(iters_to_run)).astype(int).tolist()
        else:
            assert len(t_starts) == len(iters_to_run), (
                "t_starts and iters_to_run must have the same length."
            )

        if alphas is None:
            alphas = np.linspace(0.3, 0.1, len(iters_to_run)).tolist()
        else:
            assert len(alphas) == len(iters_to_run), (
                "alphas and iters_to_run must have the same length."
            )

        self.iters_to_run = iters_to_run
        self.t_starts = iter(t_starts)
        self.alphas = iter(alphas)

        
        logging.info("(diffusionfwi) Setting diffusion interleaving scheduling")
        logging.info("(diffusionfwi) Diffusion iters_to_run: {}".format(iters_to_run))
        logging.info("(diffusionfwi) Diffusion t_starts: {}".format(t_starts))
        logging.info("(diffusionfwi) Diffusion alphas: {}".format(alphas))

    def forward(self, vp, **kwargs):
        #  Increment iteration counter
        self.iteration += 1

        # If current iteration is in the list to run diffusion, execute the pipeline
        if self.iteration in self.iters_to_run:
            random_seed = self.init_seed + self.iteration  # Ensure different seed each time
            t_start = next(self.t_starts)
            alpha = next(self.alphas)

            logging.info("(diffusionfwi) Starting diffusion FWI pipeline")

            logging.info(
                "\t Triggering diffusion at iteration {}"
                " with t_start={} and alpha={}".format(self.iteration, t_start, alpha)
            )

            # Kwargs that will be passed to the update function in the pipeline, if needed
            update_kwargs = kwargs.get("update_kwargs", {})
            update_kwargs.update({"alpha": alpha, "mask": self.mask})

            # Rewrite vp.data in-place with the output of the diffusion pipeline
            with torch.no_grad():
                vp.data[:] = self.pipeline.run(
                    vp.data,
                    t_start=t_start,
                    random_seed=random_seed,
                    update_kwargs=update_kwargs,
                    verbose=True
                )
                return vp
        else:
            return vp

    def reset_iteration(self):
        self.iteration = 0