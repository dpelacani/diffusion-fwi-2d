import os
from typing import Union, Optional, Tuple
import numpy as np
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from diffusionfwi.dataset.buildDataset import (
    AcousticNormalization, 
    ReverseAcousticNormalization,
    LogMinMaxNormalization,
    ReverseLogMinMaxNormalization
)
from diffusionfwi.diffusion import DiffusionProcess
from diffusionfwi.utils import merge_channels, split_channels

logger = logging.getLogger(__name__)

# Number of diffusion samples averaged per guidance step (both merged and split strategy)
N_SAMPLES = 8

# Physical clamp applied after denoising to ensure valid noramlized velocity range before transformation to m/s
NORM_MIN = -1.0
NORM_MAX = 1.0

def _log_stats(arr: Union[torch.Tensor, np.ndarray], tag: str = None):
    """ Utility function to log statistics of a tensor or numpy array. """
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    logger.info(
        f"\t {tag} - shape: {arr.shape}, dtype: {arr.dtype}, "
        f"min: {arr.min():.4f}, max: {arr.max():.4f}, mean: {arr.mean():.4f}, std: {arr.std():.4f}"
    )


class DataProcessingPipeline(nn.Module):
    """
    Preprocessing and postprocessing pipeline for diffusion FWI.
    Forward: resize to x_dim x x_dim -> Acoustic Normalization -> LogMinMaxNormalization
    Inverse: ReverseLogMinMaxNormalization -> ReverseAcousticNormalization -> resize to original shape

    Args:
        x_dim: square spatial dimension of diffusion model
        original_shape: spatial shape of FWI grid
    """

    def __init__(self, x_dim: int=128, original_shape: Tuple[int, int] = (320, 256)):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((x_dim, x_dim), interpolation=InterpolationMode.BILINEAR, antialias=True),
            AcousticNormalization(),
            LogMinMaxNormalization(),          
        ])
        
        self.inverse_transform = transforms.Compose([
            ReverseLogMinMaxNormalization(),   
            ReverseAcousticNormalization(),
            # NN interpolation to avoid creating velocity artifacts
            transforms.Resize(original_shape, interpolation=InterpolationMode.NEAREST)
        ])

    def process_forward(self, x):
        return self.transform(x)

    def process_inverse(self, x):
        return self.inverse_transform(x)


class DiffusionFWIPipeline:
    """
    Diffusion FWI pipeline encapsulating all steps related to running the diffusion model for FWI,
    including data preprocessing, model inference, and blending with the original velocity model.
    
    It supports both merged and split-channel guidance:
    - split=True: velocity map is split into tissue and skull channels before denoising
                  per-channel alpha blending is applied
    - split=False: single-channel velocity map passed directly to diffusion model

    Both modes average across N_SAMPLES independent results of the reverse-diffusion trajectories.

    Args:
        diffusion_model (nn.Module): Pre-trained diffusion model 
        diffusion_process (DiffusionProcess): instance of DiffusionProcess (with noise schedule)
        data_pipeline (DataProcessingPipeline): Optional data preprocessing pipeline for resize and normalization
        split (bool): if True, uses two-channel split of skull and tissue
        device (torch.device, optional): Device to run the pipeline on
    """

    def __init__(
        self,
        diffusion_model: nn.Module,
        diffusion_process: DiffusionProcess,
        data_pipeline: DataProcessingPipeline,
        split: bool = True,
        device: Optional[torch.device] = "cpu",
    ) -> None:

        self.diffusion_model = diffusion_model.to(device)
        self.diffusion_process = diffusion_process
        self.data_pipeline = data_pipeline
        self.split = split
        self.device = device


    def run(
        self,
        volume: Union[torch.Tensor, np.ndarray],
        t_start: int = 500,
        random_seed: Optional[int] = 42,
        update_kwargs: Optional[dict] = None,
        verbose: bool = False,
        visual_dir: Optional[str] = None,
        visual_iter: Optional[int] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Run the diffusion FWI pipeline on an FWI velocity map.

        Args:
            volume: velocity map in m/s
            t_start: diffusion timestep to start denoising from
            random_seed: for reproducible noise sampling
            update_kwargs: dictionary with blending parameters 
                           "alpha", "alpha_skull" and "alpha_tissue"
            verbose: log statistics and intermediate steps
            visual_dir: directory to save intermediate tensors to for visualization
            visual_iter: current FWI iteration
        
        Returns:
            Updated velocity map in m/s.
        """
        update_kwargs = update_kwargs or {}
        logger.info("(diffusionfwi) DiffusionFWIPipeline.run starting.")

        # --- basic type handling ---
        is_tensor = isinstance(volume, torch.Tensor)
        if not is_tensor:
            volume = torch.from_numpy(volume)

        original_ndim = volume.ndim
        original_vol = volume.clone()

        # Ensure volume has batch and channel dims for processing (N, C, H, W)
        while volume.ndim < 4:
            volume = volume.unsqueeze(0)

        if verbose:
            _log_stats(volume, "Input volume (m/s)")

        # Forward data preprocessing
        volume = self.data_pipeline.process_forward(volume)
        if verbose:
            _log_stats(volume, "After forward preprocessing")

        if self.split:
            volume = split_channels(volume) # shape [1, 2, x_dim, x_dim]
            volume_fwi = volume.clone() # save FWI state for blending

            # Move to device for diffusion processing
            volume = volume.to(self.device)
            volume = self._denoise(volume, t_start, random_seed, verbose, visual_dir, visual_iter)

            # Per-channel blending of FWI and diffusion output (weighted by alpha_skull and alpha_tissue)
            alpha = update_kwargs.get("alpha", 0.9)
            alpha_skull = update_kwargs.get("alpha_skull", alpha)
            alpha_tissue = update_kwargs.get("alpha_tissue", alpha)
            vfwi = volume_fwi.to(self.device)
            ch0  = (1.0 - alpha_tissue) * vfwi[:, 0:1] + alpha_tissue * volume[:, 0:1]   
            ch1  = (1.0 - alpha_skull) * vfwi[:, 1:2] + alpha_skull * volume[:, 1:2]   
            volume = torch.cat([ch0, ch1], dim=1)
            volume = merge_channels(volume) # shape [1, 1, x_dim, x_dim]
        
        else:
            volume_fwi = volume.clone() # save FWI state for blending
            # Move to device for diffusion processing
            volume = volume.to(self.device)
            volume = self._denoise(volume, t_start, random_seed, verbose, visual_dir, visual_iter)
            # Per-channel blending of FWI and diffusion output (weighted by alpha_skull and alpha_tissue)
            alpha = update_kwargs.get("alpha", 0.9)
            vfwi = volume_fwi.to(self.device)
            volume = (1.0 - alpha) * vfwi + alpha * volume
        
        # Enforce physical velocity range for FWI update stability
        volume = torch.clamp(volume, NORM_MIN, NORM_MAX)

        if verbose:
            _log_stats(volume, "After denoising and blending")

        # Inverse preprocessing
        volume = self.data_pipeline.process_inverse(volume) # back to m/s and original shape

        if verbose:
            _log_stats(volume, "After inverse preprocessing (to m/s)")

        while volume.ndim > original_ndim:
            volume = volume.squeeze(0)

        # Save final blended vp passed back to FWI
        if visual_dir:
            os.makedirs(visual_dir, exist_ok=True)
            torch.save(original_vol.cpu(), f"{visual_dir}/iter{visual_iter}_vp_before_diffusion.pt")
            torch.save(volume.cpu(), f"{visual_dir}/iter{visual_iter}_vp_blended_diffusion.pt")

        return volume.cpu() if is_tensor else volume.cpu().numpy()

    @torch.no_grad()
    def _denoise(
        self,
        volume: torch.Tensor,
        t_start: int,
        random_seed: Optional[int] = 42,
        verbose: bool = False,
        visual_dir: Optional[str] = None,
        visual_iter: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward diffuse to t_start and reverse diffuse back to t=0,
        averaging over N_SAMPLES independent diffusion trajectories.

        Args:
            volume: preprocessed tensor [1, C, H, W]
            t_start: diffusion timestep to noise to before denoising
            random_seed: seed for forward noising step
        
        Returns:
            denoised tensor [1, C, H, W], averaged over N_SAMPLES trajectories
        """
        # Choose nearest available timestep value
        t_idx = int(torch.abs(self.diffusion_process.timesteps - t_start).argmin().item())
        t_val = self.diffusion_process.timesteps[t_idx].unsqueeze(0).to(self.device)

        # --------------- Corrupt with noise ---------------
        if random_seed is not None:
            state = torch.random.get_rng_state()
            torch.manual_seed(random_seed)
            noise = torch.randn_like(volume)
            torch.random.set_rng_state(state)
        else:
            noise = torch.randn_like(volume)

        noisy_vol = self.diffusion_process.forward_diffusion(volume, t_val, noise)
        
        # Visualization: noisy vp passed to diffusion model
        if visual_dir and visual_iter is not None: 
            os.makedirs(visual_dir, exist_ok=True)
            torch.save(noisy_vol.cpu(), f"{visual_dir}/iter{visual_iter}_noisy_img.pt")

        # Expand to N_SAMPLES to batch predictions for averaging
        # same x_t with independent reverse noise per sample
        noisy_vol = noisy_vol.expand(N_SAMPLES, -1, -1, -1).contiguous()

        # --------------- Sampling ---------------
        volume = self._sample_from_start_time(
            noisy_vol, t_val, 
            visual_dir=visual_dir,
            visual_iter=visual_iter
        )

        # Averaged across samples, back to shape [1, C, H, W]
        denoised_vol = volume.mean(dim=0, keepdim=True)

        if verbose:
            _log_stats(denoised_vol, "After denoising")
        if visual_dir and visual_iter is not None:
            torch.save(denoised_vol.cpu(), f"{visual_dir}/iter{visual_iter:03d}_denoised.pt")

        return denoised_vol


    @torch.no_grad()
    def _sample_from_start_time(
        self,
        noisy_vol: torch.Tensor,
        t_start: torch.Tensor,
        visual_dir: Optional[str] = None,
        visual_iter: Optional[int] = None,
        visual_freq: int = 100
    ) -> torch.Tensor:
        """
        Reverse diffusion loop from t_start down to t=0.

        Args:
            noisy_vol: noisy tensor, shape [N_SAMPLES, C, H, W]
            t_start: starting timestep for denoising
            visual_dir: directory for saving intermediate steps
            visual_iter: FWI iteration number for naming
            visual_freq: interval for saving snapshot of reverse steps
        
        Returns: 
            denoised tensor with shape [N_SAMPLES, C, H, W]
        """
        # Choose nearest available timestep value
        timesteps = self.diffusion_process.timesteps[
            self.diffusion_process.timesteps <= t_start.item()
        ].to(self.device)
        num_steps = len(timesteps)

        self.diffusion_model.eval()
        for idx in tqdm(reversed(range(num_steps)), desc="Diffusion Sampling", total=num_steps, leave=False):
            t = timesteps[idx]
            t_batch = t.unsqueeze(0).expand(noisy_vol.shape[0]).to(self.device)
            z = torch.randn_like(noisy_vol) if t.item() > 0 else torch.zeros_like(noisy_vol)

            e_pred = self.diffusion_model(noisy_vol, t_batch)
            noisy_vol = self.diffusion_process.reverse_diffusion(noisy_vol, t_batch, e_pred, z)

            # Visualization: steps in denoising trajectory
            if visual_dir and visual_iter is not None and idx % visual_freq == 0:
                os.makedirs(visual_dir, exist_ok=True)
                torch.save(
                    noisy_vol.mean(dim=0, keepdim=True).cpu(),
                    f"{visual_dir}/iter{visual_iter}_denoise_t{int(t.item())}.pt"
                )

        return noisy_vol