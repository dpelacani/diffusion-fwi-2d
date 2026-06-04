import os
import logging
from typing import Optional
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from diffusionfwi.dataset.buildDataset import (
    AcousticNormalization, 
    LogMinMaxNormalization
)
from diffusionfwi.diffusion import DiffusionProcess
from diffusionfwi.utils import split_channels, merge_channels

logger = logging.getLogger(__name__)
# Number of averaged score estimates per guidance step
N_SAMPLES = 8

class GradientOperator:
    """
    Diffusion gradient guidance operator for FWI.
    At every FWI iteration, diffusion score ∇log p_t(vp) computed and added to FWI gradient.
    - split=True: skull and soft tissue scores are computed separately and added with independent weighting
    - N_SAMPLES=8 score estimates are averaged in merge and split mode

    Args:
        diffusion_model (torch.nn.Module): Pre-trained diffusion model for FWI.
        diffusion_process (DiffusionProcess): DiffusionProcess with cosine schedule
        vp_ref (ScalarField): Stride ScalarField of the current velocity model before FWI update
        input_dim (int): Dimension of the velocity model volume (128 x 128)
        original_dim (Tuple): Shape of FWI grid (320, 256)
        split: if True, split skull and tissue channels
        max_freqs: List of frequency bands used in FWI run
        t_start: Diffusion time at lowest frequency
        t_end: Diffusion time at highest frequency
        lambda_skull: Starting relative skull gradient strength for split channels
        lambda_tissue: Starting relative tissue gradient strength for split channels
        lambda_end: Final relative gradient strength for merged and split channel case
        device (str, optional): Device to run the diffusion model on ("cpu" or "cuda").
        visual_dir (optional): Optional directory to save diagnostics
    """

    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        diffusion_process: DiffusionProcess,
        vp_ref,
        input_dim: int = 128,
        original_dim: tuple = (320, 256), 
        split: bool = True,
        max_freqs: list = None,
        t_start: int = 700,
        t_end: int = 100,
        lambda_skull: float = 0.3,
        lambda_tissue: float = 0.3,
        lambda_end: float = 0.01,
        use_brain_mask: bool = False,
        init_seed: int = 42,
        device: str = "cuda",
        visual_dir: Optional[str] = None
    ):

        self.init_seed         = init_seed
        self.device            = device
        self.visual_dir        = visual_dir

        if max_freqs is None:
            raise ValueError("max_freqs must be provided, part of scenario definition")

        self.diffusion_model   = diffusion_model.to(device)
        self.diffusion_process = diffusion_process
        self.vp                = vp_ref
        self.f_min             = max_freqs[0]
        self.f_max_all         = max_freqs[-1]
        self.t_start           = t_start
        self.t_end             = t_end
        self.lambda_skull      = lambda_skull
        self.lambda_tissue     = lambda_tissue
        self.lambda_end        = lambda_end
        self.split             = split
        self.use_brain_mask    = use_brain_mask
        self.original_dim      = original_dim
        self.iteration         = 0

        self.preprocess = transforms.Compose([
            transforms.Resize(
                (input_dim, input_dim),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            AcousticNormalization(),
            LogMinMaxNormalization()
        ])

        # Postprocessing is only resize, score is gradient centered at 0
        self.postprocess_grad = transforms.Resize(
            original_dim,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        self.brain_mask = None
        if self.use_brain_mask:
            spacing = 0.5e-3  # m/pixel
            rx_m = (original_dim[0] * spacing - 7e-3) / 2   # 0.0765m
            ry_m = (original_dim[1] * spacing - 5e-3) / 2   # 0.0615m
            rx = rx_m / spacing   # 153 pixels
            ry = ry_m / spacing   # 123 pixels
            cx = original_dim[0] / 2   # 160
            cy = original_dim[1] / 2   # 128

            yy, xx = np.mgrid[0:original_dim[0], 0:original_dim[1]]
            elliptical_dist = np.sqrt((yy - cx)**2 / rx**2 + (xx - cy)**2 / ry**2)

            sigma = 0.05   # transition over ~5% of the ring radius
            self.brain_mask = np.exp(-np.maximum(elliptical_dist - 1.0, 0)**2 / (2 * sigma**2))

        logger.info(
            "GradientOperator initialised (split=%s, t=%d→%d, device=%s), "
            "lambda_skull=%.3f, lambda_tissue=%.3f, lambda_end=%.3f",
            split, t_start, t_end, device, lambda_skull, lambda_tissue, lambda_end
        )

    def _t_for_freq(self, f_current):
        """
        Cosine t from t_start (at lowest freq) to t_end (at highest freq).
        f_current comes from kwargs["f_max"] at each FWI iteration.
        """
        f_lo, f_hi = self.f_min, self.f_max_all
        if f_hi == f_lo:
            return self.t_start
        frac = (f_current - f_lo) / (f_hi - f_lo)
        cosine_frac = 0.5 * (1 - np.cos(np.pi * frac))
        t = int(self.t_start - (self.t_start - self.t_end) * cosine_frac)
        return t

    def _lambda_for_freq(self, f_current, lambda_start):
        """
        Cosine lambda from lambda_start (at lowest freq) to lambda_end (at highest freq).
        f_current comes from kwargs["f_max"] at each FWI iteration.
        """
        f_lo, f_hi = self.f_min, self.f_max_all
        if f_hi == f_lo:
            return lambda_start
        frac = (f_current - f_lo) / (f_hi - f_lo)
        cosine_frac = 0.5 * (1 - np.cos(np.pi * frac))
        return lambda_start - (lambda_start - self.lambda_end) * cosine_frac

    def _preprocess_vp(self, vp_np):
        """ numpy (H, W) float32 m/s → tensor (1, C, x_dim, x_dim) normalized. """
        x = torch.tensor(vp_np).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
        x = self.preprocess(x)                              # (1, 1, x_dim, x_dim)
        if self.split:
            x = split_channels(x)                           # (1, 2, x_dim, x_dim)
        return x.to(self.device)

    @torch.no_grad()
    def _compute_score(self, vp_np, t_val, random_seed=None):
        """
        Compute score direction averaged over N_SAMPLES estimates.

        Returns:
            Gradient array in FWI grid space (H, W) to add to FWI gradient after scaling
        """
        volume = self._preprocess_vp(vp_np) # [1, C, x_dim, x_dim]

        # Forward diffusion
        t_tensor = torch.tensor(t_val).unsqueeze(0).to(self.device)
        t_batch = t_tensor.expand(N_SAMPLES)
        if random_seed is not None:
            state = torch.random.get_rng_state()
            torch.manual_seed(random_seed)
            noise_batch = torch.randn(N_SAMPLES, *volume.shape[1:], device=self.device)
            torch.random.set_rng_state(state)
        else:
            noise_batch = torch.randn(N_SAMPLES, *volume.shape[1:], device=self.device)
        
        # Expand to N_SAMPLES for independent denoising predictions
        volume_batch = volume.expand(N_SAMPLES, -1, -1, -1).contiguous()
        noisy_vol_batch = self.diffusion_process.forward_diffusion(volume_batch, t_batch, noise_batch)

        self.diffusion_model.eval()
        e_pred = self.diffusion_model(noisy_vol_batch, t_batch) # [N_SAMPLES, C, x_dim, x_dim]
        score = (e_pred - noise_batch).mean(dim=0, keepdim=True) # [1, C, x_dim, x_dim]

        if self.split:
            # Compute per-channel scores separately when for split channel guidance            
            skull_score = score[:, 1:2]
            tissue_score = score[:, 0:1]
            # Resize scores to FWI grid (H, W)
            skull_grad = self.postprocess_grad(skull_score).squeeze().cpu().numpy()
            tissue_grad = self.postprocess_grad(tissue_score).squeeze().cpu().numpy()
            # Normalize gradients to unit max
            tissue_grad = tissue_grad / (np.abs(tissue_grad).max() + 1e-8)
            skull_grad = skull_grad / (np.abs(skull_grad).max() + 1e-8)
            return skull_grad, tissue_grad
        else:
            merged_score = merge_channels(score) # [1, 1, x_dim, x_dim]
            # Resize score to FWI grid (H, W)
            merged_grad = self.postprocess_grad(merged_score).squeeze().cpu().numpy()
            # Normalize gradient to unit max
            merged_grad = merged_grad / (np.abs(merged_grad).max() + 1e-8)
            return merged_grad, None

    def forward(self, gradient, **kwargs):
        """
        Called by Pipeline.forward() once per FWI iteration via process_grad.

        Args:
            gradient : Stride ScalarField or numpy array with shape (H, W) 
            **kwargs : includes f_max (current frequency cap), iteration, problem, etc.
        
        Returns:
            FWI gradient with added diffusion score
        """
        self.iteration += 1

        is_scalar_field = hasattr(gradient, "data")
        gradient_np = gradient.data.copy() if is_scalar_field else np.array(gradient)

        f_current = kwargs.get("f_max", self.f_max_all)
        t_val = self._t_for_freq(f_current)
        
        logger.info(
            f"(gradient_op) iter={self.iteration} f={f_current/1e6:.2f}MHz t={t_val} "
        )

        # Current velocity model
        vp_np = self.vp.data.copy()
        random_seed = self.init_seed + self.iteration
        skull_score, tissue_score = self._compute_score(vp_np, t_val, random_seed) # in merged case: skull score stores total score
        skull_grad = skull_score * self.brain_mask if self.use_brain_mask else skull_score
        tissue_grad = tissue_score * self.brain_mask if (self.use_brain_mask and tissue_score is not None) else tissue_score
        grad_scale = np.abs(gradient_np).max() + 1e-8

        lbd_skull = self._lambda_for_freq(f_current, self.lambda_skull)
        lbd_tissue = self._lambda_for_freq(f_current, self.lambda_tissue)
        logger.info(f"lambda_skull={lbd_skull:.4f} lambda_tissue={lbd_tissue:.4f}")


        if self.split:
            grad_update = gradient_np - lbd_skull * grad_scale * skull_grad - lbd_tissue * grad_scale * tissue_grad
        else:
            grad_update = gradient_np - lbd_skull * grad_scale * skull_grad

        if self.visual_dir:
            os.makedirs(self.visual_dir, exist_ok=True)
            np.save(f"{self.visual_dir}/iter{self.iteration}_score.npy",  skull_score)
            np.save(f"{self.visual_dir}/iter{self.iteration}_fwi_grad.npy",  gradient_np)
            np.save(f"{self.visual_dir}/iter{self.iteration}_grad_update.npy",  grad_update)

        # Add modified gradient back into ScalarField for Stride iteration
        if is_scalar_field:
            gradient.data[:] = grad_update
            return gradient

        return grad_update

    def adjoint(self, *args, **kwargs):
        return args[0] if args else None
    
    def reset_iteration(self):
        self.iteration = 0