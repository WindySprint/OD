import numpy as np
import math
import torch
import torch.nn as nn
import functools
from models.mamba_arch import SCFM
from models.modules import BLTEB, IGEB

class net_video(nn.Module):
    def __init__(self, n_feats, num_frames):
        super(net_video, self).__init__()
        self.n_feats = n_feats

        # Estimate Background Light and Transmission
        self.blteb = BLTEB(n_feats, num_frames)

        # Encode_net
        self.conv_ini = nn.Sequential(
            nn.Conv3d(3, self.n_feats, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2)
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(self.n_feats, self.n_feats * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats * 2, affine=True)
        )
        self.down_conv2 = nn.Sequential(
            nn.Conv3d(self.n_feats * 2, self.n_feats * 4, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats * 4, affine=True)
        )

        # Skip
        self.skip = nn.Sequential(
            nn.Conv3d(self.n_feats * 4, self.n_feats * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Image-Frames Fusion
        self.scfm = SCFM(dim=self.n_feats * 4,
                        d_state=math.ceil(self.n_feats * 4 / 6),
                        attn_drop=0., drop_path=0.,
                        norm_layer=nn.LayerNorm)

        # Decode_net
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(self.n_feats * 4, self.n_feats * 2, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats * 2, affine=True)
        )
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(self.n_feats * 2, self.n_feats, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats, affine=True)
        )

        # Imaging-guided Enhance
        self.igeb2 = IGEB(self.n_feats * 2)
        self.igeb1 = IGEB(self.n_feats)

        self.conv_last = nn.Sequential(
            nn.Conv3d(self.n_feats, self.n_feats, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.n_feats, 3, 3, 1, 1, bias=True)
        )

    def forward(self, frames, img_out):
        # B, C, N, H, W
        bl, t = self.blteb(frames, img_out[0])
        e1 = self.conv_ini(frames)
        e2 = self.down_conv1(e1)
        e3 = self.down_conv2(e2)

        skip = self.skip(e3)
        skip = self.scfm(skip, img_out[1])

        d2 = self.up_conv2(skip) + e2
        d2 = self.igeb2(d2, bl, t)
        d1 = self.up_conv1(d2) + e1
        d1 = self.igeb1(d1, bl, t)

        out_frames = self.conv_last(d1)

        return out_frames, bl, t