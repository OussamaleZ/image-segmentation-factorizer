import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Basic 3D Conv Block (Conv3d → GN → LeakyReLU) × 2
# ------------------------------------------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups=8, num_channels=out_c),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups=8, num_channels=out_c),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------
# nnU-Net with factorizer deep supervision (P1, P2, P3)
# ------------------------------------------------------------
class NNUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_channels=32, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # ---------------- Encoder ----------------
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = ConvBlock3D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(2)

        # ---------------- Bottleneck ----------------
        self.bottleneck = ConvBlock3D(base_channels * 8, base_channels * 16)

        # ---------------- Decoder ----------------
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 2, 2)
        self.dec4 = ConvBlock3D(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        # ---------------- Heads ----------------
        self.out_conv = nn.Conv3d(base_channels, out_channels, 1)
        self.ds2_conv = nn.Conv3d(base_channels * 2, out_channels, 1)
        self.ds3_conv = nn.Conv3d(base_channels * 4, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder
        d4 = self.up4(bn)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Heads
        p_full = self.out_conv(d1)
        p_half = self.ds2_conv(d2)
        p_quarter = self.ds3_conv(d3)

        if self.training and self.deep_supervision:
            return [p_full, p_half, p_quarter]

        return p_full
