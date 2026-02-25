import torch
import torch.nn as nn
from .Attention import AttentionBlock
from .TimeEmbedding import TimeEmbedding
from .UNetBlocks import ConvBlock, Encoder, Decoder


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super(SimpleUNet, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.enc1 = Encoder(in_channels, 64, time_emb_dim)
        self.enc2 = Encoder(64, 128, time_emb_dim)
        self.enc3 = Encoder(128, 256, time_emb_dim)
        self.enc4 = Encoder(256, 512, time_emb_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024, time_emb_dim)
        self.bottleneck2 = ConvBlock(1024, 1024, time_emb_dim)

        # Decoder
        self.dec1 = Decoder(1024, 512, time_emb_dim)
        self.dec2 = Decoder(512, 256, time_emb_dim)
        self.dec3 = Decoder(256, 128, time_emb_dim)
        self.dec4 = Decoder(128, 64, time_emb_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)

        # Encoder
        skip1, x = self.enc1(x, t)
        skip2, x = self.enc2(x, t)
        skip3, x = self.enc3(x, t)
        skip4, x = self.enc4(x, t)

        # Bottleneck
        x = self.bottleneck(x, t)
        x = self.bottleneck2(x, t)

        # Decoder
        x = self.dec1(x, t, skip4)
        x = self.dec2(x, t, skip3)
        x = self.dec3(x, t, skip2)
        x = self.dec4(x, t, skip1)

        # Final convolution
        x = self.final_conv(x)

        return x


class Simple5UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super(Simple5UNet, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.enc1 = Encoder(in_channels, 64, time_emb_dim)
        self.enc2 = Encoder(64, 128, time_emb_dim)
        self.enc3 = Encoder(128, 256, time_emb_dim)
        self.enc4 = Encoder(256, 512, time_emb_dim)
        self.enc5 = Encoder(512, 1024, time_emb_dim)

        # Bottleneck
        self.bottleneck1 = ConvBlock(1024, 2048, time_emb_dim)
        self.bottleneck2 = ConvBlock(2048, 2048, time_emb_dim)

        # Decoder
        self.dec1 = Decoder(2048, 1024, time_emb_dim)
        self.dec2 = Decoder(1024, 512, time_emb_dim)
        self.dec3 = Decoder(512, 256, time_emb_dim)
        self.dec4 = Decoder(256, 128, time_emb_dim)
        self.dec5 = Decoder(128, 64, time_emb_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)

        # Encoder
        skip1, x = self.enc1(x, t)
        skip2, x = self.enc2(x, t)
        skip3, x = self.enc3(x, t)
        skip4, x = self.enc4(x, t)
        skip5, x = self.enc5(x, t)

        # Bottleneck
        x = self.bottleneck1(x, t)
        x = self.bottleneck2(x, t)

        # Decoder
        x = self.dec1(x, t, skip5)
        x = self.dec2(x, t, skip4)
        x = self.dec3(x, t, skip3)
        x = self.dec4(x, t, skip2)
        x = self.dec5(x, t, skip1)

        # Final output
        return self.final_conv(x)


class UNetAttn(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super(UNetAttn, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.enc1 = Encoder(in_channels, 64, time_emb_dim)
        self.enc2 = Encoder(64, 128, time_emb_dim)
        self.enc3 = Encoder(128, 256, time_emb_dim)
        self.attn1 = AttentionBlock(256)  # add attention at 16*16
        self.enc4 = Encoder(256, 512, time_emb_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024, time_emb_dim)
        self.attn2 = AttentionBlock(1024)  # add attention in bottleneck
        self.bottleneck2 = ConvBlock(1024, 1024, time_emb_dim)

        # Decoder
        self.dec1 = Decoder(1024, 512, time_emb_dim)
        self.dec2 = Decoder(512, 256, time_emb_dim)
        self.attn3 = AttentionBlock(256)
        self.dec3 = Decoder(256, 128, time_emb_dim)
        self.dec4 = Decoder(128, 64, time_emb_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)

        # Encoder
        skip1, x = self.enc1(x, t)
        skip2, x = self.enc2(x, t)
        skip3, x = self.enc3(x, t)
        x = self.attn1(x)
        skip4, x = self.enc4(x, t)

        # Bottleneck
        x = self.bottleneck(x, t)
        x = self.attn2(x)
        x = self.bottleneck2(x, t)

        # Decoder
        x = self.dec1(x, t, skip4)
        x = self.dec2(x, t, skip3)
        x = self.attn3(x)
        x = self.dec3(x, t, skip2)
        x = self.dec4(x, t, skip1)

        # Final convolution
        x = self.final_conv(x)

        return x


class UNetAttn5(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super(UNetAttn5, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.enc1 = Encoder(in_channels, 64, time_emb_dim)
        self.enc2 = Encoder(64, 128, time_emb_dim)
        self.enc3 = Encoder(128, 256, time_emb_dim)
        self.attn1 = AttentionBlock(256)  # add attention at 16*16
        self.enc4 = Encoder(256, 512, time_emb_dim)
        self.enc5 = Encoder(512, 1024, time_emb_dim)

        # Bottleneck
        self.bottleneck1 = ConvBlock(1024, 2048, time_emb_dim)
        self.attn2 = AttentionBlock(2048)  # add attention in bottleneck
        self.bottleneck2 = ConvBlock(2048, 2048, time_emb_dim)

        # Decoder
        self.dec1 = Decoder(2048, 1024, time_emb_dim)
        self.dec2 = Decoder(1024, 512, time_emb_dim)
        self.dec3 = Decoder(512, 256, time_emb_dim)
        self.attn3 = AttentionBlock(256)
        self.dec4 = Decoder(256, 128, time_emb_dim)
        self.dec5 = Decoder(128, 64, time_emb_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)

        # Encoder
        skip1, x = self.enc1(x, t)
        skip2, x = self.enc2(x, t)
        skip3, x = self.enc3(x, t)
        x = self.attn1(x)
        skip4, x = self.enc4(x, t)
        skip5, x = self.enc5(x, t)

        # Bottleneck
        x = self.bottleneck1(x, t)
        x = self.attn2(x)
        x = self.bottleneck2(x, t)

        # Decoder
        x = self.dec1(x, t, skip5)
        x = self.dec2(x, t, skip4)
        x = self.dec3(x, t, skip3)
        x = self.attn3(x)
        x = self.dec4(x, t, skip2)
        x = self.dec5(x, t, skip1)

        # Final output
        return self.final_conv(x)


class UNetAttnD(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super(UNetAttnD, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.enc1 = Encoder(in_channels, 64, time_emb_dim)
        self.enc2 = Encoder(64, 128, time_emb_dim)
        self.enc3 = Encoder(128, 256, time_emb_dim)
        self.attn1 = AttentionBlock(256)  # add attention at 16*16
        self.enc4 = Encoder(256, 512, time_emb_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024, time_emb_dim, dropout=0.1)
        self.attn2 = AttentionBlock(1024)  # add attention in bottleneck
        self.bottleneck2 = ConvBlock(1024, 1024, time_emb_dim, dropout=0.1)

        # Decoder
        self.dec1 = Decoder(1024, 512, time_emb_dim)
        self.dec2 = Decoder(512, 256, time_emb_dim)
        self.attn3 = AttentionBlock(256)
        self.dec3 = Decoder(256, 128, time_emb_dim)
        self.dec4 = Decoder(128, 64, time_emb_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)

        # Encoder
        skip1, x = self.enc1(x, t)
        skip2, x = self.enc2(x, t)
        skip3, x = self.enc3(x, t)
        x = self.attn1(x)
        skip4, x = self.enc4(x, t)

        # Bottleneck
        x = self.bottleneck(x, t)
        x = self.attn2(x)
        x = self.bottleneck2(x, t)

        # Decoder
        x = self.dec1(x, t, skip4)
        x = self.dec2(x, t, skip3)
        x = self.attn3(x)
        x = self.dec3(x, t, skip2)
        x = self.dec4(x, t, skip1)

        # Final convolution
        x = self.final_conv(x)

        return x


class UNetAttnD5(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super(UNetAttnD5, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.enc1 = Encoder(in_channels, 64, time_emb_dim)
        self.enc2 = Encoder(64, 128, time_emb_dim)
        self.enc3 = Encoder(128, 256, time_emb_dim)
        self.attn1 = AttentionBlock(256)  # add attention at 16*16
        self.enc4 = Encoder(256, 512, time_emb_dim)
        self.enc5 = Encoder(512, 1024, time_emb_dim)

        # Bottleneck
        self.bottleneck1 = ConvBlock(1024, 2048, time_emb_dim, dropout=0.1)
        self.attn2 = AttentionBlock(2048)  # add attention in bottleneck
        self.bottleneck2 = ConvBlock(2048, 2048, time_emb_dim, dropout=0.1)

        # Decoder
        self.dec1 = Decoder(2048, 1024, time_emb_dim)
        self.dec2 = Decoder(1024, 512, time_emb_dim)
        self.dec3 = Decoder(512, 256, time_emb_dim)
        self.attn3 = AttentionBlock(256)
        self.dec4 = Decoder(256, 128, time_emb_dim)
        self.dec5 = Decoder(128, 64, time_emb_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)

        # Encoder
        skip1, x = self.enc1(x, t)
        skip2, x = self.enc2(x, t)
        skip3, x = self.enc3(x, t)
        x = self.attn1(x)
        skip4, x = self.enc4(x, t)
        skip5, x = self.enc5(x, t)

        # Bottleneck
        x = self.bottleneck1(x, t)
        x = self.attn2(x)
        x = self.bottleneck2(x, t)

        # Decoder
        x = self.dec1(x, t, skip5)
        x = self.dec2(x, t, skip4)
        x = self.dec3(x, t, skip3)
        x = self.attn3(x)
        x = self.dec4(x, t, skip2)
        x = self.dec5(x, t, skip1)

        # Final output
        return self.final_conv(x)


class UNetAttnD5M(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super(UNetAttnD5M, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.enc1 = Encoder(in_channels, 64, time_emb_dim)
        self.enc2 = Encoder(64, 128, time_emb_dim)
        self.enc3 = Encoder(128, 256, time_emb_dim)
        self.attn1 = AttentionBlock(256)  # add attention at 16*16
        self.enc4 = Encoder(256, 512, time_emb_dim)
        self.enc5 = Encoder(512, 1024, time_emb_dim)

        # Bottleneck
        self.bottleneck1 = ConvBlock(1024, 2048, time_emb_dim, dropout=0.1)
        self.attn2 = AttentionBlock(2048)  # add attention in bottleneck
        self.bottleneck2 = ConvBlock(2048, 2048, time_emb_dim, dropout=0.1)

        # Decoder
        self.dec1 = Decoder(2048, 1024, time_emb_dim)
        self.dec2 = Decoder(1024, 512, time_emb_dim, dropout=0.1)
        self.dec3 = Decoder(512, 256, time_emb_dim)
        self.attn3 = AttentionBlock(256)
        self.dec4 = Decoder(256, 128, time_emb_dim)
        self.dec5 = Decoder(128, 64, time_emb_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)

        # Encoder
        skip1, x = self.enc1(x, t)
        skip2, x = self.enc2(x, t)
        skip3, x = self.enc3(x, t)
        x = self.attn1(x)
        skip4, x = self.enc4(x, t)
        skip5, x = self.enc5(x, t)

        # Bottleneck
        x = self.bottleneck1(x, t)
        x = self.attn2(x)
        x = self.bottleneck2(x, t)

        # Decoder
        x = self.dec1(x, t, skip5)
        x = self.dec2(x, t, skip4)
        x = self.dec3(x, t, skip3)
        x = self.attn3(x)
        x = self.dec4(x, t, skip2)
        x = self.dec5(x, t, skip1)

        # Final output
        return self.final_conv(x)


class UNetAttnD5E(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super(UNetAttnD5E, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.enc1 = Encoder(in_channels, 64, time_emb_dim)
        self.enc2 = Encoder(64, 128, time_emb_dim)
        self.enc3 = Encoder(128, 256, time_emb_dim)
        self.attn1 = AttentionBlock(256)  # add attention at 16*16
        self.enc4 = Encoder(256, 512, time_emb_dim, dropout=0.05)
        self.enc5 = Encoder(512, 1024, time_emb_dim)

        # Bottleneck
        self.bottleneck1 = ConvBlock(1024, 2048, time_emb_dim, dropout=0.1)
        self.attn2 = AttentionBlock(2048)  # add attention in bottleneck
        self.bottleneck2 = ConvBlock(2048, 2048, time_emb_dim, dropout=0.1)

        # Decoder
        self.dec1 = Decoder(2048, 1024, time_emb_dim)
        self.dec2 = Decoder(1024, 512, time_emb_dim)
        self.dec3 = Decoder(512, 256, time_emb_dim)
        self.attn3 = AttentionBlock(256)
        self.dec4 = Decoder(256, 128, time_emb_dim)
        self.dec5 = Decoder(128, 64, time_emb_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)

        # Encoder
        skip1, x = self.enc1(x, t)
        skip2, x = self.enc2(x, t)
        skip3, x = self.enc3(x, t)
        x = self.attn1(x)
        skip4, x = self.enc4(x, t)
        skip5, x = self.enc5(x, t)

        # Bottleneck
        x = self.bottleneck1(x, t)
        x = self.attn2(x)
        x = self.bottleneck2(x, t)

        # Decoder
        x = self.dec1(x, t, skip5)
        x = self.dec2(x, t, skip4)
        x = self.dec3(x, t, skip3)
        x = self.attn3(x)
        x = self.dec4(x, t, skip2)
        x = self.dec5(x, t, skip1)

        # Final output
        return self.final_conv(x)


class UNetAttnMoreD(nn.Module):
    """
    UNet with more attention blocks and dropout in the bottleneck.
    This model is designed to handle more complex features and improve performance on tasks requiring detailed feature extraction.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        time_emb_dim: Dimension of the time embedding.
    """

    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super(UNetAttnMoreD, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.enc1 = Encoder(in_channels, 64, time_emb_dim)
        self.enc2 = Encoder(64, 128, time_emb_dim)
        self.enc3 = Encoder(128, 256, time_emb_dim)
        self.attn1 = AttentionBlock(256)  # add attention at 16*16
        self.enc4 = Encoder(256, 512, time_emb_dim, dropout=0.02)
        self.enc5 = Encoder(512, 1024, time_emb_dim, dropout=0.02)

        # Bottleneck
        self.bottleneck1 = ConvBlock(1024, 2048, time_emb_dim, dropout=0.1)
        self.attn2 = AttentionBlock(2048)  # add attention in bottleneck
        self.bottleneck2 = ConvBlock(2048, 2048, time_emb_dim, dropout=0.1)

        # Decoder
        self.dec1 = Decoder(2048, 1024, time_emb_dim)
        self.dec2 = Decoder(1024, 512, time_emb_dim, dropout=0.02)
        self.dec3 = Decoder(512, 256, time_emb_dim)
        self.attn3 = AttentionBlock(256)
        self.dec4 = Decoder(256, 128, time_emb_dim)
        self.dec5 = Decoder(128, 64, time_emb_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)

        # Encoder
        skip1, x = self.enc1(x, t)
        skip2, x = self.enc2(x, t)
        skip3, x = self.enc3(x, t)
        x = self.attn1(x)
        skip4, x = self.enc4(x, t)
        skip5, x = self.enc5(x, t)

        # Bottleneck
        x = self.bottleneck1(x, t)
        x = self.attn2(x)
        x = self.bottleneck2(x, t)

        # Decoder
        x = self.dec1(x, t, skip5)
        x = self.dec2(x, t, skip4)
        x = self.dec3(x, t, skip3)
        x = self.attn3(x)
        x = self.dec4(x, t, skip2)
        x = self.dec5(x, t, skip1)

        # Final output
        return self.final_conv(x)
