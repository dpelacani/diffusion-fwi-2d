import torch
import torch.nn as nn
from .TimeEmbedding import TimeEmbedding
from .UNetBlocks import ConvBlock, Encoder, Decoder


class UNet(nn.Module):
    def __init__(
        self, 
        in_channels=2, 
        out_channels=2, 
        time_emb_dim=256, 
        num_layers=4, 
        base_channels=64
    ):
        super().__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)

        enc_channels = [base_channels * (2 ** i) for i in range(num_layers)]

        # Encoders
        self.encoders = nn.ModuleList()
        ch = in_channels
        for ch_out in enc_channels:
            self.encoders.append(Encoder(ch, ch_out, time_emb_dim))
            ch = ch_out

        # Bottleneck
        bottleneck_ch = enc_channels[-1] * 2
        self.bottleneck = ConvBlock(enc_channels[-1], bottleneck_ch, time_emb_dim)
        self.bottleneck2 = ConvBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)

        # Decoder
        self.decoders = nn.ModuleList()
        ch = bottleneck_ch
        for ch_out in reversed(enc_channels):
            self.decoders.append(Decoder(ch, ch_out, time_emb_dim))
            ch = ch_out

        # Final convolution
        self.final_conv = nn.Conv2d(enc_channels[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)

        # Encoder: store skip connection
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x, t)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x, t)
        x = self.bottleneck2(x, t)

        # Decoder: consume skip in reverse
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, t, skip)

        # Final convolution
        x = self.final_conv(x)

        return x