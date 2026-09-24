"""
Module 3: Lightweight Model — Depthwise-Separable + Linear-Attention U-Net
============================================================================
This is the project's efficiency contribution (targets Gap 1: no paper we
reviewed reports parameters/FLOPs/latency alongside accuracy).

Two changes relative to the Attention U-Net baseline:

1. Every standard 3x3 convolution is replaced with a depthwise-separable
   convolution (a 3x3 depthwise conv + a 1x1 pointwise conv), which cuts
   the parameter count of a conv layer by roughly `1/out_ch + 1/9` of the
   original — the same trick MobileNet and ShearFuse-UNet [7] use.
2. Instead of an Oktay-style attention gate on every skip connection (4
   gates, expensive), a single linear-attention block (Katharopoulos et
   al., 2020 formulation — O(N) instead of O(N^2) in the number of spatial
   tokens) is applied once at the bottleneck, in the spirit of LinU-Mamba
   [5]. This keeps a global-context mechanism while avoiding the repeated
   full-resolution attention-gate cost.

Same I/O contract as the baseline: (B, 12, 64, 64) -> (B, 1, 64, 64).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class LiteConvBlock(nn.Module):
    """Two depthwise-separable convs — the lightweight drop-in for ConvBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            DepthwiseSeparableConv(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class LinearAttention(nn.Module):
    """Linear attention (Katharopoulos et al., 2020): softmax attention's
    O(N^2) similarity matrix is replaced with a kernel feature map
    phi(x) = elu(x) + 1, letting attention be computed as
    phi(Q) (phi(K)^T V), i.e. O(N * C^2) instead of O(N^2 * C) — linear in
    the number of spatial tokens N = H*W, which is what makes this cheap
    enough to use at the bottleneck of a segmentation U-Net.
    """

    def __init__(self, channels, heads=4):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.heads = heads
        self.head_dim = channels // heads
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)  # each (B, C, H, W)
        q, k, v = [
            t.reshape(b, self.heads, self.head_dim, h * w) for t in qkv
        ]  # (B, heads, head_dim, N)

        q, k = self._phi(q), self._phi(k)
        # linear attention: out = q (k^T v) / (q sum(k))
        kv = torch.einsum("bhdn,bhen->bhde", k, v)          # (B, heads, head_dim, head_dim)
        k_sum = k.sum(dim=-1, keepdim=True)                  # (B, heads, head_dim, 1)
        denom = torch.einsum("bhdn,bhdk->bhn", q, k_sum).clamp_min(1e-6)  # (B, heads, N)
        out = torch.einsum("bhdn,bhde->bhen", q, kv) / denom.unsqueeze(2)  # (B, heads, head_dim, N)

        out = out.reshape(b, c, h, w)
        return x + self.to_out(out)  # residual


class LightweightUNet(nn.Module):
    """
    `dropout_p` inserts nn.Dropout2d after the bottleneck and each decoder
    stage. It does double duty: light regularization during normal training,
    and — left switched on at inference via `model.enable_mc_dropout()` —
    the stochasticity that Module 6 (src/uncertainty.py) needs to run
    Monte Carlo Dropout and get a per-pixel confidence estimate. A model
    with `dropout_p=0` (the default-off baseline) cannot produce MC-Dropout
    uncertainty; the risk-tier layer requires `dropout_p > 0`.
    """

    def __init__(self, in_channels=12, base_ch=24, attn_heads=4, dropout_p: float = 0.2):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16

        self.enc1 = LiteConvBlock(in_channels, c1)
        self.enc2 = LiteConvBlock(c1, c2)
        self.enc3 = LiteConvBlock(c2, c3)
        self.enc4 = LiteConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = LiteConvBlock(c4, c5)
        self.bottleneck_attn = LinearAttention(c5, heads=attn_heads)
        self.drop_b = nn.Dropout2d(dropout_p)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = LiteConvBlock(c4 * 2, c4)
        self.drop4 = nn.Dropout2d(dropout_p)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = LiteConvBlock(c3 * 2, c3)
        self.drop3 = nn.Dropout2d(dropout_p)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = LiteConvBlock(c2 * 2, c2)
        self.drop2 = nn.Dropout2d(dropout_p)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = LiteConvBlock(c1 * 2, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        b = self.drop_b(self.bottleneck_attn(b))

        d4 = self.drop4(self.dec4(torch.cat([self.up4(b), e4], dim=1)))
        d3 = self.drop3(self.dec3(torch.cat([self.up3(d4), e3], dim=1)))
        d2 = self.drop2(self.dec2(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))

    def enable_mc_dropout(self):
        """Keep Dropout2d layers stochastic at eval time (everything else
        — BatchNorm etc. — stays in eval mode), which is what MC-Dropout
        requires: dropout masks that vary run-to-run without also letting
        BatchNorm statistics drift on a single-image forward pass."""
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()


if __name__ == "__main__":
    model = LightweightUNet(in_channels=12, base_ch=24)
    x = torch.randn(2, 12, 64, 64)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output shape: {tuple(y.shape)}  (expect (2, 1, 64, 64))")
    print(f"parameters: {n_params:,}")
