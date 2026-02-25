import io
import logging
from typing import Union, Optional, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm

from torchvision import transforms
from ..dataset.buildDataset import AcousticNormalization, ReverseAcousticNormalization


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()  # Clear any existing handlers


def log_array_stats(arr: Union[torch.Tensor, np.ndarray], logger: Optional[logging.Logger] = None):
    """Utility function to log statistics of a tensor or numpy array."""
    logger = logger or logging.getLogger(__name__)
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    logger.info(
        f"Array stats - shape: {arr.shape}, dtype: {arr.dtype}, "
        f"min: {arr.min():.4f}, max: {arr.max():.4f}, mean: {arr.mean():.4f}, std: {arr.std():.4f}"
    )


class DataProcessingPipeline(torch.nn.Module):
    """Data processing pipeline for diffusion FWI.
    This includes all data preprocessing steps before running the diffusion model,
    such as gain application, skull separation, and registration (if applicable).
    The exact steps are determined by the provided configuration.
    """

    def __init__(self, x_dim: int=256, original_shape: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.transform = transforms.Compose([
            # use anitialias = True to avoid aliasing artifacts
            transforms.Resize((x_dim, x_dim), interpolation=InterpolationMode.BILINEAR, antialias=True),
            # apply the log transformation
            AcousticNormalization(),
            # normalize to -1 to 1 range
            transforms.Normalize(mean=[0.5], std=[0.5])])
        
        self.inverse_transform = transforms.Compose([
            # inverse of normalize
            transforms.Normalize(mean=[-1.0], std=[2.0]),
            # inverse of log transformation
            ReverseAcousticNormalization(),
            # inverse of resize - use nearest neighbour to avoid creating new values
            transforms.Resize(original_shape or x_dim, interpolation=InterpolationMode.NEAREST)
        ])

    def process_forward(self, x):
        return self.transform(x)

    def process_inverse(self, x):
        return self.inverse_transform(x)


class DiffusionFWIPipeline:
    """DiffusionFWIPipeline with consistent device handling and
    robust scheduler interaction.

    This pipeline encapsulates all steps related to running the diffusion model for FWI,
    including data preprocessing, model inference, and blending with the original velocity model.

    Args:
        diffusion_model (torch.nn.Module): Pre-trained diffusion model 
        data_pipeline (DataProcessingPipeline, optional): Optional data preprocessing pipeline
        update_fn (callable, optional): Optional function to blend the generated volume with the original
        device (torch.device, optional): Device to run the pipeline on
    """

    def __init__(
        self,
        diffusion_model: Optional[torch.nn.Module],
        data_pipeline: Optional[DataProcessingPipeline],
        update_fn: Optional[callable] = None,
        device: Optional[torch.device] = "cpu",
    ) -> None:

        self.diffusion_model = diffusion_model
        self.update_fn = update_fn or self._update_fn
        self.data_pipeline = data_pipeline
        self.device = device


    def run(
        self,
        volume: Union[torch.Tensor, np.ndarray],
        t_start: int = 100,
        random_seed: Optional[int] = 42,
        update_kwargs: Optional[dict] = None,
        verbose: bool = False,
    ) -> Union[torch.Tensor, np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the diffusion FWI pipeline on the given velocity model volume.
        """
        device = self.device

        logger.info("DiffusionFWIPipeline.run starting.")

        # --- basic type handling ---
        is_tensor = isinstance(volume, torch.Tensor)
        if not is_tensor:
            volume = torch.from_numpy(volume)

        original_ndim = volume.ndim
        original_vol = volume.clone()

        if verbose:
            logger.info("Volume before pipeline.")
            log_array_stats(volume)

        # Forward data preprocessing
        if self.data_pipeline is not None:
            volume = self.data_pipeline.process_forward(volume)
            if volume.ndim != 4:
                logger.error(
                    f"Volume after forward pipeline has wrong dims: {volume.shape},"
                    " reshaping now - could cause unexpected behaviour."
                )
            while volume.ndim < 4:
                volume = volume.unsqueeze(0)

        # Move to device for diffusion processing
        volume = volume.to(device)
        volume_input = volume.clone()

        if verbose:
            logger.info("Volume after data preprocessing.")
            log_array_stats(volume)


        # --- Denoised model ---
        volume = self._denoise_local(
                volume=volume,
                t_start=t_start,
                random_seed=random_seed,
                verbose=verbose,
            )

        # Inverse preprocessing
        if self.data_pipeline is not None:
            volume = self.data_pipeline.process_inverse(volume)
            if verbose:
                logger.info("Reverse preprocessed decoded, denoised volume.")
                log_array_stats(volume, logger=logger)

        # Remove extra dims added earlier, if any
        while volume.ndim > original_ndim:
            volume = volume.squeeze(0)

        # Blend with original based on update_fn
        volume = self.update_fn(volume, original_vol, **(update_kwargs or {}))

        if verbose:
            logger.info("Blended volume.")
            log_array_stats(volume)

        return volume if is_tensor else volume.cpu().numpy()

    # --------------------------------------------------------------------- #
    # Local inference (shared between pipeline & API server)
    # --------------------------------------------------------------------- #
    def _denoise_local(
        self,
        volume: torch.Tensor,
        t_start: int,
        random_seed: Optional[int] = 42,
        verbose: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor | None]]:
        """Core inference step on a *preprocessed* volume.
        """
        device = self.device

        # Move model and volume to device
        self.diffusion_model = self.diffusion_model.to(device)
        volume = volume.to(device)


        # pick nearest available timestep value
        t_idx = int(
            torch.abs(self.diffusion_process.timesteps - t_start).argmin().item()
        )
        t_val = self.diffusion_process.timesteps[t_idx].unsqueeze(0).to(device)

        if verbose:
            logger.info(f"Picked nearest available timestep value: {t_val}")

        # --------------- Corrupt with noise ---------------
        if random_seed is not None:
            state = torch.random.get_rng_state()
            torch.manual_seed(random_seed)
            noise = torch.randn_like(volume, device=device)
            torch.random.set_rng_state(state)
        else:
            noise = torch.randn_like(volume, device=device)

        if verbose:
            logger.info(f"Generated noise, shape: {noise.shape}")

        volume = self.diffusion_process.forward_diffusion(volume, t_val, noise)

        # --------------- Sampling ---------------
        volume = self._sample_from_start_time(volume, t_val, device=device)

        if verbose:
            logger.info("Volume post sampling.")
            log_array_stats(volume)

        return decoded


    @torch.no_grad()
    def _sample_from_start_time(
        self,
        noisy_volume: torch.Tensor,
        t_start: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Core sampling loop that walks scheduler timesteps from t_start down to 0."""
        
        device = device or self.device

        # Move model and scheduler to device (safe, idempotent)
        self.diffusion_model = self.diffusion_model.to(device)
        volume = noisy_volume.to(device)

        # pick nearest available timestep value
        timesteps = self.diffusion_process.timesteps[
            self.diffusion_process.timesteps <= t_start.item()
        ].to(device)

        # prepare the list of next timesteps for scheduler stepping
        all_next = torch.cat(
            (timesteps[1:], torch.tensor([0], dtype=timesteps.dtype, device=device))
        )
        n_steps = len(timesteps)
        assert n_steps == len(all_next)

        with torch.inference_mode():
            for idx in tqdm(range(n_steps), desc="Diffusion Sampling"):
                t = timesteps[idx].to(device)
                next_t = all_next[idx].to(device)

                if t.item() <= 0:
                    break  # reached final timestep

                batch = volume.shape[0]
                model_t = t.unsqueeze(0).expand(batch).to(volume.device)

                e_pred = self.diffusion_model(volume, timesteps=model_t)

                volume = self.diffusion_process.reverse_diffusion(volume, model_t, e_pred, torch.randn_like(volume))

                volume = volume.to(device=device, dtype=volume.dtype)

        return volume


   def _update_fn(
        self,
        generated_volume: Union[torch.Tensor, np.ndarray],
        original_volume: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> Union[torch.Tensor, np.ndarray]:
        mask = kwargs.get("mask", None)
        alpha = kwargs.get("alpha", 1.0)

        if mask is None:
            mask = torch.ones_like(generated_volume)
        return (1 - alpha * mask) * original_volume + alpha * mask * generated_volume


    def __repr__(self) -> str:
        return (
            f"DiffusionFWIPipeline(checkpoint={self.diffusion_checkpoint}, "
            f"device={self.device}, inference_mode={self.inference_mode})"
        )