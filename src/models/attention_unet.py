"""
Module 2: Baseline Reproduction — Attention U-Net
====================================================
Reproduces the base paper's architecture: a standard U-Net encoder/decoder
with Oktay-style additive attention gates (Oktay et al., 2018) on every
skip connection, so the decoder learns to focus on fire-relevant spatial
regions before fusing encoder features. This is the accuracy benchmark
our lightweight model (src/models/lightweight_unet.py) is compared against.

Input:  (B, 12, 64, 64)  — the 12 NDWS channels
Output: (B, 1, 64, 64)   — per-pixel next-day fire probability (post-sigmoid)
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 conv -> BN -> ReLU layers."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018).

    Gates the encoder skip features `x` using the coarser decoder features
    `g`, so only spatial regions relevant to the current decoding step pass
    through to the concatenation.
    """

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=True), nn.BatchNorm2d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=True), nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        attn = self.relu(self.W_g(g) + self.W_x(x))
        attn = self.psi(attn)  # (B, 1, H, W) attention coefficients in [0, 1]
        return x * attn


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=12, base_ch=32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16

        # Encoder
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(c4, c5)

        # Decoder + attention gates
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.att4 = AttentionGate(c4, c4, c4 // 2)
        self.dec4 = ConvBlock(c5, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.att3 = AttentionGate(c3, c3, c3 // 2)
        self.dec3 = ConvBlock(c4, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.att2 = AttentionGate(c2, c2, c2 // 2)
        self.dec2 = ConvBlock(c3, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.att1 = AttentionGate(c1, c1, c1 // 2)
        self.dec1 = ConvBlock(c2, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                  # (B, c1, 64, 64)
        e2 = self.enc2(self.pool(e1))      # (B, c2, 32, 32)
        e3 = self.enc3(self.pool(e2))      # (B, c3, 16, 16)
        e4 = self.enc4(self.pool(e3))      # (B, c4, 8, 8)
        b = self.bottleneck(self.pool(e4))  # (B, c5, 4, 4)

        # Decoder
        d4 = self.up4(b)                           # (B, c4, 8, 8)
        d4 = torch.cat([self.att4(d4, e4), d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)                          # (B, c3, 16, 16)
        d3 = torch.cat([self.att3(d3, e3), d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                          # (B, c2, 32, 32)
        d2 = torch.cat([self.att2(d2, e2), d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                          # (B, c1, 64, 64)
        d1 = torch.cat([self.att1(d1, e1), d1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out_conv(d1))    # (B, 1, 64, 64)


if __name__ == "__main__":
    model = AttentionUNet(in_channels=12, base_ch=32)
    x = torch.randn(2, 12, 64, 64)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output shape: {tuple(y.shape)}  (expect (2, 1, 64, 64))")
    print(f"parameters: {n_params:,}")
