import numpy as np
import torch
from tqdm import tqdm


class DiffusionProcess:
    def __init__(self, device, T=1000, s=0.008):
        """
        Initialize the diffusion process with the number of timesteps T and a scaling factor s.

        Args:
            device: The device to run the computations on (e.g., 'cpu' or 'cuda').
            T: The number of timesteps for the diffusion process.
            s: A scaling factor for the cosine schedule.
        """
        self.T = T
        self.s = s
        self.device = device
        self.set_variables()

    def set_variables(self):
        """
        Set the beta and alpha variables for the diffusion process. So we can use them in the forward and reverse diffusion processes.
        """
        self.timesteps = np.arange(0, self.T + 1)
        self.betas = self.cosine_beta_schedule().to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def cosine_beta_schedule(self):
        """
        Compute the beta schedule using a cosine function.
        Using the cosine schedule explained in https://arxiv.org/abs/2102.09672.
        The betas are computed based on a cosine function that varies over the timesteps.

        Returns:
            A tensor of betas for each timestep.
        """
        # calculate the variables using the cosine function
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
            param: The parameter tensor to combine with the timestep.
            t: The timestep tensor.
            device: The device to run the computations on.

        Returns:
            A tensor representing the noise variance for the given timestep.
        """
        # get the noise variance for the given timestep t
        # it will be in the shape of batch_size, 1, 1, 1
        return param.to(device)[t.to(device)].view(-1, 1, 1, 1)

    def forward_diffusion(self, x_0, t, e):
        """
        Perform the forward diffusion process.

        Args:
            x_0: The original data tensor.
            t: The timestep tensor.
            e: The noise tensor.

        Returns:
            The noisy data tensor at timestep t.
        """
        # calculate the variables needed for the forward diffusion
        sqrt_alpha = self.combine_timestep(torch.sqrt(self.alphas_bar), t, x_0.device)
        sqrt_one_minus_alpha = self.combine_timestep(
            torch.sqrt(1.0 - self.alphas_bar), t, x_0.device
        )
        # the forward diffusion equation
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * e

    def reverse_diffusion(self, x, t, e, z):
        """
        Perform the reverse diffusion process.

        Args:
            x: The noisy data tensor at timestep t.
            t: The timestep tensor.
            e: The noise tensor.
            z: The noise tensor for the reverse process.

        Returns:
            The denoised data tensor at timestep t-1.
        """
        # calculate the variables needed for the reverse diffusion
        sigma = self.combine_timestep(torch.sqrt(self.betas), t, x.device)
        sqrt_recip_alphas = self.combine_timestep(
            torch.sqrt(1.0 / self.alphas), t, x.device
        )
        scale = self.combine_timestep(
            (1 - self.alphas) / torch.sqrt(1.0 - self.alphas_bar), t, x.device
        )
        # the reverse diffusion equation
        return sqrt_recip_alphas * (x - scale * e) + sigma * z

    def sample_diffusion(self, model, data_loader, batch_size, device):
        """
        Sample from the diffusion model.

        Args:
            model: The diffusion model to sample from.
            data_loader: The data loader to get the shape of the data.
            batch_size: The batch size for sampling.
            device: The device to run the computations on.

        Returns:
            A tensor of the final sample from the diffusion model.
        """
        model.eval()

        # Randomly sample an xT batch from a normal distribution
        # We will simply refer to xT as x
        sample_shape = next(iter(data_loader)).shape[1:]
        x = torch.randn((batch_size, *sample_shape)).to(device)

        # send to the GPU
        x = x.to(device)

        # Container to store sample throughout the reverse diffusion
        # Context management to ensure gradients are not tracked for any torch tensors or parameters
        # since we are only interested in sampling, not training
        with torch.no_grad():

            # Loop over time from T to 0
            for t in tqdm(
                reversed(range(0, self.T)), desc="Sampling Steps", total=self.T
            ):

                # sample z from a normal distribution if condition met
                if t > 0:
                    z = torch.randn_like(x)
                else:
                    z = 0

                # convert t to a tensor, send to the GPU, and expand its first dimension to be equal to batch size
                t_tensor = torch.tensor(t).to(device).expand(x.shape[0])

                # denoise x
                x = self.reverse_diffusion(x, t_tensor, model(x, t_tensor), z)

            # clipping to ensure the sample values remain in the range -0.4172 to 1 at the last step
            # -0.4172 is the min value of the training data after normalization
            x = torch.clamp(x, -0.4172, 1)
        return x
