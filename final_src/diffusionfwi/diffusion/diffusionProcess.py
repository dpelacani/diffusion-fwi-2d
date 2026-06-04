import numpy as np
import torch
from tqdm import tqdm

from diffusionfwi.utils import merge_channels

class DiffusionProcess:
    def __init__(self, device, T=1000, s=0.001):
        """
        Initialize the diffusion process with the number of timesteps T and a scaling factor s.

        Args:
            device: device to run the computations on (e.g., 'cpu' or 'cuda').
            T: number of timesteps for the diffusion process.
            s: scaling factor for the cosine schedule.
        """
        self.T = T
        self.s = s
        self.device = device
        self.set_variables()

    def set_variables(self):
        """
        Set the beta and alpha variables for the diffusion process. 
        """
        self.timesteps = torch.arange(0, self.T + 1)
        self.betas = self.cosine_beta_schedule().to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def cosine_beta_schedule(self):
        """
        Compute the beta schedule using a cosine function.
        Using the cosine schedule from Nichol & Dhariwal (2021), https://arxiv.org/abs/2102.09672.
        The betas are computed based on a cosine function that varies over the timesteps.

        Returns:
            A tensor of betas for each timestep.
        """
        # Calculate the variables using the cosine function
        alphas_cumprod = (
            np.cos(((self.timesteps / self.T) + self.s) / (1 + self.s) * np.pi * 0.5)
            ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to 1
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.tensor(np.clip(betas, 1e-6, 0.999), dtype=torch.float32)

    def combine_timestep(self, param, t, device):
        """
        Combine the parameter with the timestep t to get the noise variance.

        Args:
            param: parameter tensor to combine with the timestep.
            t: timestep tensor.
            device: device to run the computations on.

        Returns:
            A tensor representing the noise variance for the given timestep.
        """
        # Get noise variance for given timestep t, shape [B, 1, 1, 1]
        return param.to(device)[t.to(device)].view(-1, 1, 1, 1)

    def forward_diffusion(self, x_0, t, e):
        """
        Perform the forward diffusion process.

        Args:
            x_0: original data tensor.
            t: timestep tensor.
            e: noise tensor.

        Returns:
            Noisy data tensor at timestep t.
        """
        # Calculate variables needed for forward diffusion
        sqrt_alpha = self.combine_timestep(torch.sqrt(self.alphas_bar), t, x_0.device)
        sqrt_one_minus_alpha = self.combine_timestep(
            torch.sqrt(1.0 - self.alphas_bar), t, x_0.device
        )
        # Forward diffusion equation
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * e

    def reverse_diffusion(self, x, t, e, z):
        """
        Perform a step of the reverse diffusion process.

        Args:
            x: noisy data tensor at timestep t.
            t: timestep tensor.
            e: noise tensor.
            z: noise tensor for the reverse process.

        Returns:
            The denoised data tensor at timestep t-1.
        """
        # Calculate variables needed for reverse diffusion
        sigma = self.combine_timestep(torch.sqrt(self.betas), t, x.device)
        if t[0].item() == 0:
            sigma = 0  # no noise at the final step
        sqrt_recip_alphas = self.combine_timestep(
            torch.sqrt(1.0 / self.alphas), t, x.device
        )
        scale = self.combine_timestep(
            (1 - self.alphas) / torch.sqrt(1.0 - self.alphas_bar), t, x.device
        )
        # Reverse diffusion equation
        return sqrt_recip_alphas * (x - scale * e) + sigma * z

    def sample_diffusion(self, model, data_loader, batch_size, device, num_channels):
        """
        Sample from the diffusion model.

        Args:
            model: trained diffusion model to sample from.
            data_loader: data loader to get shape of data.
            batch_size: batch size for sampling.
            device: device to run the computations on.
            num_channels: number of channels to sample (2 for split model, 1 otherwise)

        Returns:
            Tensor of shape [B, 1, H, W] in normalized training space.
        """
        model.eval()

        # Randomly sample an xT batch from a normal distribution, we will simply refer to xT as x
        sample_shape = next(iter(data_loader)).shape[1:]
        H, W = sample_shape[-2], sample_shape[-1]
        x = torch.randn((batch_size, num_channels, H, W), device=device) # [N, C, H, W]

        # Context management to ensure gradients are not tracked for any torch tensors or parameters
        # since we are only interested in sampling, not training
        with torch.no_grad():

            # Loop over time from T to 0
            for t in tqdm(reversed(range(0, self.T)), desc="Sampling Steps", total=self.T):
                 
                # Sample z from normal distribution if condition met
                z = torch.randn_like(x) if t > 0 else 0

                # Convert t to a tensor, send to the GPU, and expand its first dimension to be equal to batch size
                t_tensor = torch.tensor(t, device=device).expand(x.shape[0])
                
                # Denoise x
                x = self.reverse_diffusion(x, t_tensor, model(x, t_tensor), z)

            # Conditional merge: split models [N, 2, H, W] -> [N, 1, H, W]
            if num_channels > 1:
                x = merge_channels(x)

        return x