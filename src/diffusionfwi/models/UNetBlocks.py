import torch
import torch.nn as nn


# Define Convolutional Block
class ConvBlock(nn.Module):
    """
    Convolutional Block with two convolutional layers, group normalization, and time embedding.
    Using two convolutional layers because it allows for more complex feature extraction.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        time_emb_dim: Dimension of the time embedding.
        dropout: Dropout rate.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb1 = nn.Linear(time_emb_dim, out_channels)
        self.gn1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb2 = nn.Linear(time_emb_dim, out_channels)
        self.gn2 = nn.GroupNorm(8, out_channels)
        # only need to apply residual connection if in_channels != out_channels
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

        # Apply dropout and activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.silu = nn.SiLU()

    def forward(self, x, t):
        res = self.residual_conv(x)
        x = self.conv1(x)
        x += self.time_emb1(t)[:, :, None, None]
        x = self.gn1(x)
        x = self.silu(x)

        x = self.conv2(x)
        x += self.time_emb2(t)[:, :, None, None]
        x = self.gn2(x)
        x = self.silu(x)
        x = self.dropout(x)
        return x + res


# Define the encoder
class Encoder(nn.Module):
    """
    Encoder block that applies two convolutional blocks followed by a max pooling layer.
    This is a basic block for U-Net. It downsamples the input while extracting features. With the skip connection, it allows the decoder to access high-resolution features.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        time_emb_dim: Dimension of the time embedding.
        dropout: Dropout rate for the first ConvBlock.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, time_emb_dim, dropout=dropout)
        self.conv2 = ConvBlock(out_channels, out_channels, time_emb_dim)
        self.res = nn.Conv2d(in_channels, out_channels, 1)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x, t):
        h = self.conv1(x, t)
        h = self.conv2(h, t)
        h += self.res(x)
        p = self.pool(h)
        return h, p  # return the skip connection and pooled output


# Define the decoder
class Decoder(nn.Module):
    """
    Decoder block that applies a transposed convolution followed by two convolutional blocks.
    This block upsamples the input and combines it with the skip connection from the encoder.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        time_emb_dim: Dimension of the time embedding.
        dropout: Dropout rate for the first ConBlock.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Consider using skip connections
        # to combine features from the encoder
        self.conv1 = ConvBlock(
            out_channels * 2, out_channels, time_emb_dim, dropout=dropout
        )
        self.conv2 = ConvBlock(out_channels, out_channels, time_emb_dim)

    def forward(self, x, t, skip):
        x = self.up(x)
        # Concatenate the skip connection
        h = torch.cat((x, skip), axis=1)
        h = self.conv1(h, t)
        h = self.conv2(h, t)
        return h
