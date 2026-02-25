import torch
import torch.nn as nn
import numpy as np


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for time steps.
    This module generates sinusoidal embeddings for the given time steps, allowing the model to get information about the relative position of each time step.
    The embeddings are computed using sine and cosine functions.

    Args:
        dim: The dimension of the embedding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device) * -(np.log(10000.0) / half_dim)
        )
        emb = time[:, None] * emb[None, :]
        # combine sine and cosine embeddings
        emb = torch.stack((emb.sin(), emb.cos()), dim=-1)  # (B, half_dim, 2)
        emb = emb.view(len(time), self.dim)
        return emb


class TimeEmbedding(nn.Module):
    """
    Time embedding module that applies a sinusoidal positional embedding followed by a linear transformation.
    This module aims to encode the time step information into a higher-dimensional space, making it learnable by the model.

    Args:
        dim: The dimension of the embedding.
        hidden_dim: The dimension of the hidden layer in the MLP.
    """

    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t):
        x = self.sin_emb(t)
        return self.mlp(x)
