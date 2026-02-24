import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadedAttention(nn.Module):
    """
    Multiheaded Attention module.
    This module implements multiheaded attention like in the Transformer architecture.
    It splits the input into multiple heads, applies attention to each head, and then concatenates the results.

    Args:
        dim: Dimension of the input features.
        num_heads: Number of attention heads.
        attn_dropout: Dropout rate for the attention weights.
        proj_dropout: Dropout rate for the output projection.
    """
    def __init__(self, dim, num_heads=8, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape
        # seperate q, k, v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        qkv = qkv.reshape(3, B * self.num_heads, N, C // self.num_heads) # (3, B * num_heads, N, head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=0)  # each is (1, B * num_heads, N, head_dim)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)  # (B * num_heads, N, head_dim)
        # calculate dot product of q and k
        attn = torch.bmm(q, k.transpose(-2, -1)) * self.scale  # (B * num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        # Multiply with v
        x = torch.bmm(attn, v)  # (B * num_heads, N, head_dim)
        x = x.reshape(B, self.num_heads, N, C // self.num_heads).permute(0, 2, 1, 3)
        x = x.reshape(B, N, C)  # (B, N, C)
        # Linear layer
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x
    
class AttentionBlock(nn.Module):
    """
    Attention Block that applies multiheaded attention to the input tensor.
    This block is designed to be used in our UNet architecture later.
    It applies layer normalization and GELU activation after the attention.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = MultiheadedAttention(dim=channels)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_ = x.flatten(2).permute(0, 2, 1) # (B, N, C)
        x_ = self.norm(x_)
        x_ = self.act(x_)
        x_ = self.mha(x_)
        x_ = x_.permute(0, 2, 1).reshape(B, C, H, W)
        return x + x_  # residual connection
    
