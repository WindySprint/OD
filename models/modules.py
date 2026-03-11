import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

##############--- Multi-blocks ---################
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

##############--- Residual Block (RB) ---################
class RB(nn.Module):
    def __init__(self, n_feats=16):
        super(RB, self).__init__()
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        x1 = self.conv3_2(self.conv3_1(x))
        x = x + x1
        return x

class ChannelTransfer(nn.Module):
    r"""Transfer 2D feature channel"""

    def __init__(self, n_feats=16):
        super().__init__()
        self.norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x

class ChannelReturn(nn.Module):
    r"""Return 2D feature channel"""

    def __init__(self, n_feats=16):
        super().__init__()
        self.norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

##############--- Channel Fusion Block (CFB) ---################
class CFB(nn.Module):
    def __init__(self, n_feats=16):
        super(CFB, self).__init__()
        self.linear = nn.Linear(n_feats, 1, bias=False)
        self.norm = nn.LayerNorm(n_feats)

    def forward(self, f, i):
        # b c h*w
        ori_f = f.clone()
        f = f.permute(0, 2, 1).contiguous()# b h*w c
        i = i.permute(0, 2, 1).contiguous()# b h*w c
        fi = self.linear(f + i)# b h*w 1
        fi = torch.matmul(ori_f, fi).permute(0, 2, 1).contiguous()# b 1 c
        fic = f * fi # b h*w c
        fic = self.norm(fic)
        fic = fic.permute(0, 2, 1).contiguous() # b c h*w

        return fic

def get_BL(imgs):
    """
    Calculate the background light (BL) for images.

    Args:
        img (torch.Tensor): Input tensor of shape [b, 3, h, w].

    Returns:
        torch.Tensor: Background light tensor of shape [b, 3, h, w].
    """
    b, c, h, w = imgs.shape
    img_np = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    img_np = img_np * 255.0 

    BL = np.zeros((b, h, w, c))

    for i in range(b): # Calculate BL for each channel (R, G, B) OR (B, G, R)
        BL[i, :, :, 0] = 140 / (1 + 14.4 * np.exp(-0.034 * np.median(img_np[i, :, :, 0])))  # R
        BL[i, :, :, 1] = (1.13 * np.mean(img_np[i, :, :, 1])) + 1.11 * np.std(img_np[i, :, :, 1]) - 25.6  # G
        BL[i, :, :, 2] = (1.13 * np.mean(img_np[i, :, :, 2])) + 1.11 * np.std(img_np[i, :, :, 2]) - 25.6  # B

    BL = np.clip(BL, 5, 250)  # Clip values to [5, 250]
    BL = BL / 255.0  # Normalize to [0, 1]

    BL = torch.from_numpy(BL).float().permute(0, 3, 1, 2).cuda()
    return BL

##############--- Background Light and Transmission Estimation Block (BLTEB) ---################
class BLTEB(nn.Module):
    def __init__(self, n_feats=16, num_frames=5):
        super(BLTEB, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d((num_frames, 1, 1))
        self.bl_ca = nn.Sequential(
            nn.Conv3d(3, n_feats, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv3d(n_feats, 3, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        self.t_ca = nn.Sequential(
            nn.Conv3d(3, n_feats, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv3d(n_feats, 1, 3, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # learnable parameter
        self.w1 = torch.nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
        self.t_bias = nn.Parameter(torch.zeros(1, 1, num_frames, 1, 1), requires_grad=True)

    def forward(self, i, j):
        # B, C, N, H, W & B, C, H, W
        bl = get_BL(i[:, :, i.size(2)//2, :, :])
        bl = bl.unsqueeze(2).repeat(1, 1, i.size(2), 1, 1).contiguous()

        j = j.unsqueeze(2).repeat(1, 1, i.size(2), 1, 1).contiguous()
        diff = self.gap(i - j)
        bl_diff = self.bl_ca(diff).expand_as(i)

        bl = bl + self.w1 * bl_diff

        # imaging formula
        ep = 1e-8  # A small constant used to prevent zero division
        t3 = torch.mul((i - bl), 1 / (j - bl + ep))
        t = self.t_ca(i + self.w2 * t3)

        t = t + self.t_bias

        return bl, t

def add_mixed_noise(x, add_std=0.05, mul_strength=0.3):
    gaussian = torch.randn_like(x) * add_std
    factor = 1 + mul_strength * (torch.rand_like(x) - 0.5)
    noisy = x * factor + gaussian
    return torch.clamp(noisy, 0.0, 1.0)

##############--- Imaging-guided Enhancement Block (IGEB) ---################
class IGEB(nn.Module):
    def __init__(self, n_feats=16, r=4):
        super(IGEB, self).__init__()
        # eliminate background light
        self.bl_conv = nn.Sequential(
            nn.Conv3d(3, n_feats, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2)
        )
        self.clrc = nn.Sequential(
            nn.Conv3d(n_feats, n_feats // r, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv3d(n_feats // r, n_feats, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # transmission enhance
        self.tx_rb1 = nn.Sequential(
            nn.Conv3d(n_feats, n_feats, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2)
        )
        self.tx_rb2 = nn.Sequential(
            nn.Conv3d(n_feats, n_feats, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2)
        )

        # fuse dual-branch
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(n_feats * 2, n_feats, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv3d(n_feats, n_feats, 3, 1, 1, bias=True)
        )

    def forward(self, x, bl, t):
        # B, C, N, H, W & B, 3, N, H, W & B, 1, N, H, W
        B, C, N, H, W = x.size()
        bl = F.interpolate(bl, size=(N, H, W), mode='trilinear', align_corners=True)
        t = F.interpolate(t, size=(N, H, W), mode='trilinear', align_corners=True)

        bl = self.bl_conv(bl)
        gap = nn.AdaptiveAvgPool3d((N, 1, 1))
        blx = self.sigmoid(self.clrc(gap(x - bl)))
        blx = x * blx

        tx = x + self.tx_rb1(x * t)
        tx = tx + self.tx_rb2(tx)

        out = self.fuse_conv(torch.cat([blx, tx], dim=1))

        return out
