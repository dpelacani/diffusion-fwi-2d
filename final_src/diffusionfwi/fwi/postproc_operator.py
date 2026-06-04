import logging
import os
from typing import Optional
import numpy as np
import torch
from stride import Operator  

from diffusionfwi.diffusion import DiffusionProcess
from diffusionfwi.utils import cosine_schedule
from .pipeline import DataProcessingPipeline, DiffusionFWIPipeline

logger = logging.getLogger(__name__)

class PostprocessingOperator(Operator):
    """
    Diffusion post-processing operator for FWI.
    Current FWI velocity model denoised at scheduled iterations by diffusion prior.
    Diffusion output blended with FWI estimate as next FWI iterate.
    - split=True: skull and soft tissue channels are denoised jointly but blended independently (alpha_skull, alpha_tissue)
    - N_SAMPLES=8 diffusion trajectories are averaged in merge and split mode

    Args:
        diffusion_model (torch.nn.Module): Pre-trained diffusion model for FWI.
        diffusion_process (DiffusionProcess): DiffusionProcess with cosine schedule
        input_dim (int): Dimension of the velocity model volume (128 x 128)
        original_dim (Tuple): Shape of FWI grid (320, 256)
        split: if True, split skull and tissue channels for blending
        iters_to_run: List of iterations to run diffusion
        t_start: Diffusion time at lowest frequency
        t_end: Diffusion time at highest frequency
        alpha_skull: Blending parameter for diffusion and FWI output at lowest frequency for skull channel 
        alpha_tissue: Blending parameter for diffusion and FWI output at lowest frequency for tissue channel
            skull and tissue parameters should be equal for "merged" guidance scenario
        alpha_end: Blending parameter for diffusion and FWI output at highest frequency for both skull and tissue
        init_seed (int, optional): Base random seed for diffusion sampling to ensure reproducibility.
        device (str, optional): Device to run the diffusion model on ("cpu" or "cuda").
        visual_dir (optional): Optional directory to save diagnostics
    """

    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        diffusion_process: DiffusionProcess,
        input_dim: int = 128,
        original_dim: tuple = (320, 256),
        split: bool = True,
        iters_to_run: list = None,
        t_start: int = 500,
        t_end: int = 100,
        alpha_skull: float = 0.9,
        alpha_tissue: float = 0.9,
        alpha_end: float = 0.1,
        init_seed: int = 42,
        device: str = "cpu",
        visual_dir: Optional[str] = None
    ):
        super().__init__(name="diffusion_postprocessing")

        self.init_seed = init_seed
        self.device = device
        self.visual_dir = visual_dir

        # Instantiate data preprocessing pipeline
        data_pipeline = DataProcessingPipeline(x_dim=input_dim, original_shape=original_dim)

        # Instantiate diffusion FWI pipeline
        self.pipeline = DiffusionFWIPipeline(
            diffusion_model=diffusion_model,
            diffusion_process=diffusion_process,
            data_pipeline=data_pipeline,
            split=split,
            device=device,
        )

        # Set up interleaving scheduling
        self._set_scheduling(iters_to_run, t_start, t_end, alpha_skull, alpha_tissue, alpha_end)
        # Initialise iteration counter 
        self.iteration = 0

        logger.info("(diffusionfwi) Diffusion Postprocessing FWI Pipeline initialized (split=%s) with device=%s}", split, device)


    def _set_scheduling(self, iters_to_run, t_start, t_end, alpha_skull, alpha_tissue, alpha_end):
        """
        Set up the scheduling for diffusion FWI iterations.
        Implements cosine annealing between t_start -> t_end and alpha_skull/tissue -> alpha_end parameters across iterations.

        Args:
            total_iterations (int): Total number of iterations for the optimisation.
            iters_to_run (list): List of iteration numbers to run diffusion.
            t_start: Diffusion time at lowest frequency
            t_end: Diffusion time at highest frequency
            alpha_skull: Alpha value for skull channel at lowest frequency.
            alpha_tissue: Alpha value for tissue channel at lowest frequency.
            alpha_end: Alpha value at highest frequency.
        """

        if iters_to_run is None:
            raise ValueError("iters_to_run must be provided, derived from scenario's block_iters")
        K = len(iters_to_run)
        self.iters_to_run   = iters_to_run

        t_list = [int(round(v)) for v in cosine_schedule(t_start, t_end, K)]
        skull_list = [round(v, 4) for v in cosine_schedule(alpha_skull, alpha_end, K)]
        tissue_list = [round(v, 4) for v in cosine_schedule(alpha_tissue, alpha_end, K)]

        logger.info("(diffusionfwi) Setting diffusion interleaving scheduling")
        logger.info("(diffusionfwi) Diffusion iters_to_run: {}".format(iters_to_run))
        logger.info("(diffusionfwi) Diffusion t_starts: {}".format(t_list))
        logger.info("(diffusionfwi) Diffusion alphas_skull: {}".format(skull_list))
        logger.info("(diffusionfwi) Diffusion alphas_tissue: {}".format(tissue_list))

        self._t_starts = iter(t_list)
        self._alphas_skull = iter(skull_list)
        self._alphas_tissue = iter(tissue_list)
    
    def forward(self, vp, **kwargs):
        """  """
        #  Increment iteration counter
        self.iteration += 1

        if self.iteration not in self.iters_to_run:
            return vp

        t_start      = next(self._t_starts)
        alpha_skull  = next(self._alphas_skull)
        alpha_tissue = next(self._alphas_tissue)
        random_seed  = self.init_seed + self.iteration  # ensure different seed each time

        logger.info(
            "Diffusion postprocessing at iter %d: t=%d alpha_skull=%.3f alpha_tissue=%.3f",
            self.iteration, t_start, alpha_skull, alpha_tissue
        )

        # Visualize vp before diffusion
        if self.visual_dir:
            os.makedirs(self.visual_dir, exist_ok=True)
            np.save(
                f"{self.visual_dir}/iter{self.iteration}_vp_prediffusion.npy",
                vp.data.copy()
            )

        logger.info("(diffusionfwi) Starting diffusion FWI pipeline")

        # Kwargs that will be passed to the update function in the pipeline, if needed
        update_kwargs = {
            "alpha_skull":  alpha_skull,
            "alpha_tissue": alpha_tissue
        }

        # Rewrite vp.data in-place with the output of the diffusion pipeline
        with torch.no_grad():
            vp.data[:] = self.pipeline.run(
                vp.data,
                t_start = t_start,
                random_seed = random_seed,
                update_kwargs = update_kwargs,
                verbose = True,
                visual_dir = self.visual_dir,
                visual_iter = self.iteration
            )
            return vp

    def adjoint(self, *args, **kwargs):
        return args[0] if args else None

    def reset_iteration(self):
        self.iteration = 0