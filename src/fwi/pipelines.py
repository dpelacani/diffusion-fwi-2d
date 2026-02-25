import io
import logging
from typing import Union, Optional, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm

from torchvision import transforms
from ..dataset.buildDataset import AcousticNormalization


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()  # Clear any existing handlers


class DataProcessingPipeline(torch.nn.Module):
    """Data processing pipeline for diffusion FWI.
    This includes all data preprocessing steps before running the diffusion model,
    such as gain application, skull separation, and registration (if applicable).
    The exact steps are determined by the provided configuration.
    """

    def __init__(self, x_dim: int=256):
        super().__init__()
        self.transform = transforms.Compose([
            # use anitialias = True to avoid aliasing artifacts
            transforms.Resize((x_dim, x_dim), interpolation=InterpolationMode.BILINEAR, antialias=True),
            # apply the log transformation
            AcousticNormalization(),
            # normalize to -1 to 1 range
            transforms.Normalize(mean=[0.5], std=[0.5])])

    def forward(self, x):
        return self.transform(x)


class DiffusionFWIPipeline:
    """DiffusionFWIPipeline with consistent device handling and
    robust scheduler interaction.

    This pipeline loads a diffusion model from checkpoint and runs
    inference on input volumes with optional preprocessing and
    postprocessing via a DataProcessingPipeline.

    Args:
        diffusion_checkpoint: Path to diffusion model checkpoint.
        data_pipeline: Optional DataProcessingPipeline for preprocessing
            and postprocessing volumes.
        config: Optional dict with subdicts for configuration:
            - "diffusion": kwargs to override when loading the diffusion Runner
            - "scheduler": kwargs to override when loading the scheduler
            - "inference": {
                  "mode": "local" or "api",
                  "api_url": "...",
                  "api_timeout": 300,
              }
        update_fn: Optional callable to update the generated volume
            based on the original volume. Should accept
            (generated_volume: torch.Tensor, original_volume: torch.Tensor, **kwargs)
        device: torch.device to run the pipeline on. Defaults to CPU.
    """

    def __init__(
        self,
        diffusion_model: Optional[torch.nn.Module] = None,
        data_pipeline: Optional[DataProcessingPipeline] = None,
        update_fn: Optional[callable] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.diffusion_model = diffusion_model

        self.update_fn = update_fn or self._update_fn
        self.data_pipeline = data_pipeline
        self.device = device or torch.device("cpu")


    def run(
        self,
        volume: Union[torch.Tensor, np.ndarray],
        t_start: int = 100,
        random_seed: Optional[int] = 42,
        update_kwargs: Optional[dict] = None,
        verbose: bool = False,
        return_auxiliary_volumes: bool = False,
    ) -> Union[torch.Tensor, np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the diffusion pipeline on an input volume.

        - Accepts numpy arrays or torch tensors.
        - For local mode:
            * Applies forward preprocessing (if data_pipeline is set)
            * Runs NN inference locally via _run_local_inference
            * Applies inverse preprocessing
        - For API mode:
            * Calls the remote API via _denoise_via_api
            * Assumes the service returns a denoised volume in the
              SAME space
              /shape as the input.
        - At the end, blends with the original via update_fn.
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

        # Forward preprocessing
        if self.data_pipeline is not None:
            volume = self.data_pipeline.process_forward(volume)
            if volume.ndim != 5:
                logger.error(
                    f"Volume after forward pipeline has wrong dims: {volume.shape},"
                    " reshaping now - could cause unexpected behaviour."
                )
            while volume.ndim < 5:
                volume = volume.unsqueeze(0)

        volume = volume.to(device)
        volume_input = volume.clone()

        if verbose:
            logger.info("Volume after data preprocessing.")
            log_array_stats(volume)

        # ------------------------------------------------------------------
        # Branch on inference mode
        # ------------------------------------------------------------------
        if self.inference_mode == "local":
            if self.runner is None:
                raise RuntimeError(
                    "Runner not initialised; call _load_diffusion_runner or "
                    "initialise with a checkpoint for local inference."
                )

            # --- Local model inference (preprocessed space) ---
            volume = self._denoise_local(
                volume=volume,
                t_start=t_start,
                random_seed=random_seed,
                verbose=verbose,
            )

        elif self.inference_mode == "api":
            if verbose:
                logger.info(
                    f"Running inference via API at {self.api_url} "
                    f"(timeout={self.api_timeout}s)"
                )
            volume = self._denoise_via_api(
                volume,
                t_start=t_start,
                random_seed=random_seed,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown inference_mode: {self.inference_mode}")

        if return_auxiliary_volumes:
            volume_reg = volume.clone()
        else:
            volume_reg = None

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

        if volume_reg is None:
            return volume
        else:
            # volume_pre_reg is the decoded volume before inverse pipeline
            return volume, volume_reg, volume_input

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
        """Core LDM inference step on a *preprocessed* volume.

        This is what you can also call from your API endpoint, e.g.:

            def api_handler(volume):
                # volume already in (N, C, D, H, W) and correct scaling
                denoised = pipeline._run_local_inference(volume, t_start=100)

        It:
        - Assumes volume has shape (N, C, D, H, W)
        - Uses self.runner.{autoencoder, model, scheduler}
        - Returns the decoded, denoised volume in the same (preprocessed) space.
        """
        if self.inference_mode != "local":
            raise RuntimeError("Local inference requested but inference_mode='api'.")

        assert self.runner is not None, "Runner not initialized"
        device = self.device

        # Move model and scheduler to device (safe, idempotent)
        self.runner.model = self.runner.model.to(device)
        self.runner.scheduler = self._move_scheduler_tensors(
            self.runner.scheduler, device
        )

        volume = volume.to(device)

        # ---------------- Encode ----------------
        self.runner.autoencoder.eval()
        with torch.inference_mode():
            _ = self.runner.autoencoder.encode(volume)
            latents = self.runner.autoencoder.sample() * self.latent_scaling_factor
            if verbose:
                logger.info("Encoded volume (latents).")
                log_array_stats(latents)

        # pick nearest available timestep value
        t_idx = int(
            torch.abs(self.runner.scheduler.timesteps - t_start).argmin().item()
        )
        t_val = self.runner.scheduler.timesteps[t_idx].unsqueeze(0).to(device)

        if verbose:
            logger.info(f"Picked nearest available timestep value: {t_val}")

        # --------------- Corrupt with noise ---------------
        if random_seed is not None:
            state = torch.random.get_rng_state()
            torch.manual_seed(random_seed)
            noise = torch.randn_like(latents, device=device)
            torch.random.set_rng_state(state)
        else:
            noise = torch.randn_like(latents, device=device)

        if verbose:
            logger.info(f"Generated noise, shape: {noise.shape}")

        latents = self.runner.scheduler.add_noise(latents, noise, t_val)

        # --------------- Sampling ---------------
        latents = self._sample_from_start_time(latents, t_val, device=device)

        if verbose:
            logger.info("Latents post sampling.")
            log_array_stats(latents)

        # --------------- Decode ---------------
        with torch.no_grad():
            decoded = self.runner.autoencoder.decode(
                latents / self.latent_scaling_factor
            )

            if verbose:
                logger.info("Decoded, denoised volume (pre-inverse-pipeline).")
                log_array_stats(decoded)

        return decoded

    def _denoise_via_api(
        self,
        volume: Union[torch.Tensor, np.ndarray],
        t_start: int,
        random_seed: Optional[int] = 42,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Call the new FastAPI denoise endpoint:
            POST {self.api_url}/denoise
        using multipart/form-data:
            - volume: .npy file
            - t_start, random_seed, verbose, return_auxiliary_volumes: form fields
        Returns:
            denoised volume as torch.Tensor (float32) on CPU.
        """
        import requests

        api_url = getattr(self, "api_url", None)
        if not api_url:
            raise ValueError("self.api_url is not set (e.g. 'http://localhost:8000').")

        url = api_url.rstrip("/") + "/denoise"

        # ---- Convert to numpy (CPU) ----
        if isinstance(volume, torch.Tensor):
            vol_np = volume.detach().cpu().numpy()
        else:
            vol_np = np.asarray(volume)

        # ---- Serialize to .npy bytes (no temp file needed) ----
        buf = io.BytesIO()
        np.save(buf, vol_np)
        buf.seek(0)

        # ---- Multipart upload (matches curl -F ...) ----
        files = {
            # (filename, fileobj/bytes, content_type)
            "volume": ("volume.npy", buf, "application/octet-stream"),
        }
        data = {
            "t_start": str(int(t_start)),
            "random_seed": "" if random_seed is None else str(int(random_seed)),
            "verbose": "true" if verbose else "false",
        }

        try:
            resp = requests.post(url, files=files, data=data, timeout=600)
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {e}") from e

        if resp.status_code != 200:
            # FastAPI usually returns JSON {"detail": "..."} on errors
            detail = None
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:2000]
            raise RuntimeError(
                f"API returned {resp.status_code} from {url}. Detail: {detail}"
            )

        # ---- Response is raw .npy bytes ----
        try:
            denoised_np = np.load(io.BytesIO(resp.content), allow_pickle=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse API response as .npy. "
                f"Got {len(resp.content)} bytes. Error: {e}"
            ) from e

        denoised_tensor = (
            torch.from_numpy(denoised_np).to(torch.float32).to(self.device)
        )

        return denoised_tensor

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

    @torch.no_grad()
    def _sample_from_start_time(
        self,
        noisy_volume: torch.Tensor,
        t_start: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Core sampling loop that walks scheduler timesteps from t_start down to 0."""
        assert self.runner is not None, "Runner not initialized"
        device = device or self.device

        # Move model and scheduler to device (safe, idempotent)
        self.runner.model = self.runner.model.to(device)
        self.runner.scheduler = self._move_scheduler_tensors(
            self.runner.scheduler, device
        )

        volume = noisy_volume.to(device)
        timesteps = self.runner.scheduler.timesteps[
            self.runner.scheduler.timesteps <= t_start.item()
        ].to(device)

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

                output = self.runner.model(volume, timesteps=model_t)

                if not isinstance(self.runner.scheduler, RFlowScheduler):
                    volume, _ = self.runner.scheduler.step(output, t, volume)
                else:
                    volume, _ = self.runner.scheduler.step(output, t, volume, next_t)

                volume = volume.to(device=device, dtype=volume.dtype)

        return volume


    def __repr__(self) -> str:
        return (
            f"DiffusionFWIPipeline(checkpoint={self.diffusion_checkpoint}, "
            f"device={self.device}, inference_mode={self.inference_mode})"
        )