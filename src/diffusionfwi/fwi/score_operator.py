import os
import logging
from typing import Optional
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from diffusionfwi.dataset.buildDataset import AcousticNormalization
from diffusionfwi.diffusion import DiffusionProcess
from diffusionfwi.utils import split_channels, merge_channels
from diffusionfwi.constants import SKULL_THRESH_MS

logger = logging.getLogger(__name__)
# Number of averaged score estimates per guidance step
N_SAMPLES = 8

class ScoreGuidanceOperator:
    """
    Diffusion score guidance (SG/SG-C) operator for FWI.
    Implements the GuidanceOperator protocol (see base_operator.py) as a
    process_grad pipeline step.
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
        warmstart_steps: int = 1,
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
        self.max_freqs         = sorted(max_freqs)
        self.t_start           = t_start
        self.t_end             = t_end
        self.lambda_skull      = lambda_skull
        self.lambda_tissue     = lambda_tissue
        self.lambda_end        = lambda_end
        self.split             = split
        self.original_dim      = original_dim
        self.iteration         = 0
        self.warmstart_steps   = warmstart_steps

        self.preprocess = transforms.Compose([
            transforms.Resize(
                (input_dim, input_dim),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            AcousticNormalization(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        # Postprocessing is only resize, score is gradient centered at 0
        self.postprocess_grad = transforms.Resize(
            original_dim,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        logger.info(
            "ScoreGuidanceOperator initialised (split=%s, t=%d→%d, device=%s), "
            "lambda_skull=%.3f, lambda_tissue=%.3f, lambda_end=%.3f",
            split, t_start, t_end, device, lambda_skull, lambda_tissue, lambda_end
        )
    def _band_index(self, f_current):
        """ Return 0-indexed band number for current frequency cap. """
        return int(np.argmin(np.abs(np.array(self.max_freqs) - f_current)))

    def _t_for_freq(self, f_current):
        """
        Cosine t from t_start (at lowest freq) to t_end (at highest freq).
        f_current comes from kwargs["f_max"] at each FWI iteration.
        """
        f_lo, f_hi = self.max_freqs[0], self.max_freqs[-1]
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
        f_lo, f_hi = self.max_freqs[0], self.max_freqs[-1]
        if f_hi == f_lo:
            return lambda_start
        frac = (f_current - f_lo) / (f_hi - f_lo)
        cosine_frac = 0.5 * (1 - np.cos(np.pi * frac))
        return lambda_start - (lambda_start - self.lambda_end) * cosine_frac

    def _preprocess_vp(self, vp_np, zero_tissue=False):
        """ numpy (H, W) float32 m/s → tensor (1, C, x_dim, x_dim) normalized. """
        x = torch.tensor(vp_np).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
        x = self.preprocess(x)                              # (1, 1, x_dim, x_dim)
        if self.split:
            if zero_tissue:
                # Warmstart: tissue channel = zeros, skull channel = full normalised FWI
                x = torch.cat([torch.zeros_like(x), x.clone()], dim=1)
            else:
                x = split_channels(x)                           # (1, 2, x_dim, x_dim)
        return x.to(self.device)

    @torch.no_grad()
    def _compute_score(self, vp_np, t_val, random_seed=None, zero_tissue=False,
                        visual_dir: Optional[str] = None, iteration: Optional[int] = None,):
        """
        Compute score direction averaged over N_SAMPLES estimates.

        Returns:
            Gradient array in FWI grid space (H, W) to add to FWI gradient after scaling
        """
        volume = self._preprocess_vp(vp_np, zero_tissue=zero_tissue) # [1, C, x_dim, x_dim]

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

        if visual_dir and iteration is not None:
            os.makedirs(visual_dir, exist_ok=True)
            avg_noise  = noise_batch.mean(dim=0, keepdim=True).cpu()
            avg_noisy  = noisy_vol_batch.mean(dim=0, keepdim=True).cpu()
            avg_e_pred = e_pred.mean(dim=0, keepdim=True).cpu()
            score_dir  = (noisy_vol_batch - e_pred).mean(dim=0, keepdim=True).cpu()
            torch.save(avg_noise,  f"{visual_dir}/iter{iteration:03d}_noise_added.pt")
            torch.save(avg_noisy,  f"{visual_dir}/iter{iteration:03d}_noisy_vol.pt")
            torch.save(avg_e_pred, f"{visual_dir}/iter{iteration:03d}_noise_predicted.pt")
            torch.save(score_dir,  f"{visual_dir}/iter{iteration:03d}_score_direction.pt")

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
        try:
            self.iteration += 1

            is_scalar_field = hasattr(gradient, "data")
            gradient_np = gradient.data.copy() if is_scalar_field else np.array(gradient)

            f_current = kwargs.get("f_max", self.max_freqs[-1])
            t_val = self._t_for_freq(f_current)

            # Determine warmstart: apply to split model runs in first K iterations
            zero_tissue = (self.split and self.warmstart_steps is not None and self.warmstart_steps > 0
                            and self._band_index(f_current) < self.warmstart_steps)
            
            logger.info(
                f"(score_op) iter={self.iteration} f={f_current/1e6:.2f}MHz t={t_val} zero_tissue={zero_tissue}"
            )

            # Current velocity model
            vp_np = self.vp.data.copy()
            
            if self.visual_dir:
                os.makedirs(self.visual_dir, exist_ok=True)
                np.save(f"{self.visual_dir}/iter{self.iteration:03d}_vp.npy", vp_np)
                with open(f"{self.visual_dir}/iter{self.iteration:03d}_t_val.txt", "w") as fh:
                    fh.write(str(t_val))

            random_seed = self.init_seed + self.iteration
            skull_grad, tissue_grad = self._compute_score(vp_np, t_val, random_seed, zero_tissue=zero_tissue, 
                visual_dir=self.visual_dir, iteration=self.iteration,) # in merged case: skull score stores total score
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

                it = self.iteration
                # 3-digit padded (primary)
                np.save(f"{self.visual_dir}/iter{it:03d}_diff_grad_skull.npy",  skull_grad)
                np.save(f"{self.visual_dir}/iter{it:03d}_fwi_grad.npy",         gradient_np)
                np.save(f"{self.visual_dir}/iter{it:03d}_grad_update.npy",      grad_update)
                if self.split and tissue_grad is not None:
                    np.save(f"{self.visual_dir}/iter{it:03d}_diff_grad_tissue.npy", tissue_grad)

                np.save(f"{self.visual_dir}/iter{self.iteration}_diff_grad.npy", skull_grad)
                np.save(f"{self.visual_dir}/iter{self.iteration}_fwi_grad.npy", gradient_np)
                np.save(f"{self.visual_dir}/iter{self.iteration}_grad_update.npy", grad_update)

            # Add modified gradient back into ScalarField for Stride iteration
            if is_scalar_field:
                gradient.data[:] = grad_update
                return gradient

            return grad_update
        
        except Exception as e:
            import traceback
            traceback.print_exc()   
            raise

    def adjoint(self, *args, **kwargs):
        return args[0] if args else None
    
    def reset_iteration(self):
        self.iteration = 0