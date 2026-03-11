import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat

from models.modules import ChannelTransfer, ChannelReturn, CFB
import numpy as np
import matplotlib.pyplot as plt
import os

class CFSS(nn.Module):
    r"""Channel Fusion Selective Scan"""
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.d_model * self.expand)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj1 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.cfbs = nn.ModuleList([CFB(n_feats=self.d_inner) for i in range(4)])

        self.f_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.f_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.f_proj], dim=0))  # (K=4, N, inner)
        del self.f_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.fusion_proj = nn.Linear(self.d_inner*2, self.d_inner, bias=bias, **factory_kwargs)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, f, i):
        B, C, H, W = f.shape
        L = H * W
        K = 4
        f_hwwh = torch.stack([f.view(B, -1, L), torch.transpose(f, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        sf = torch.cat([f_hwwh, torch.flip(f_hwwh, dims=[-1])], dim=1)  # [4, 4, 128, 4800]

        i_hwwh = torch.stack([i.view(B, -1, L), torch.transpose(i, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        si = torch.cat([i_hwwh, torch.flip(i_hwwh, dims=[-1])], dim=1)  # [4, 4, 128, 4800]

        for i, cfb in enumerate(self.cfbs):
            sf_slice = sf[:, i, :, :].clone()
            si_slice = si[:, i, :, :].clone()
            sf[:, i, :, :] = cfb(sf_slice, si_slice)
        # sf = sf + si

        f_dbl = torch.einsum("b k d l, k c d -> b k c l", sf.view(B, K, -1, L), self.f_proj_weight)
        dts, Bs, Cs = torch.split(f_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        sf = sf.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            sf, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, f, i):
        B, H, W, C = f.shape
        # 2 linear
        fz = self.in_proj1(f)
        f, z = fz.chunk(2, dim=-1)
        f = f.permute(0, 3, 1, 2).contiguous()
        f = self.act(self.conv1(f))

        iz = self.in_proj2(i)
        i, z = iz.chunk(2, dim=-1)
        i = i.permute(0, 3, 1, 2).contiguous()
        i = self.act(self.conv2(i))

        # self.show_feature_map(H, f.view(B, -1, H, W), './results/Attention/v')
        # self.show_feature_map(H, i.view(B, -1, H, W), './results/Attention/i')

        # 2D-SSM
        y1, y2, y3, y4 = self.forward_core(f, i)
        assert y1.dtype == torch.float32
        # # Pre_Fusion
        # y = y1 + y2 + y3 + y4
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        # Channel_Fusion
        v = (y1+y2).view(B, -1, H, W)
        h = (y3+y4).view(B, -1, H, W)
        # self.show_feature_map(H, v.view(B, -1, H, W), './results/Attention/v')
        # self.show_feature_map(H, h.view(B, -1, H, W), './results/Attention/h')
        y = self.fusion_proj(torch.cat((v, h), 1).permute(0, 2, 3, 1).contiguous())
        # self.show_feature_map(H, y.permute(0, 3, 1, 2), './results/Attention/out')

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    def show_feature_map(self, size, feature_map, path):
        feature_map = feature_map.squeeze(0)
        feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1],
                                       feature_map.shape[2])
        upsample = torch.nn.UpsamplingBilinear2d(size=(240, 320))
        feature_map = upsample(feature_map)
        feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])
        feature_map_num = feature_map.shape[0]
        for index in range(1, feature_map_num + 1):
            plt.axis('off')
            folder_name = path
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            plt.imsave(folder_name + '//' + str(size) + "_" + str(index) + ".png",
                       feature_map[index - 1].detach().cpu().numpy(), cmap='jet')

class SCFB(nn.Module):
    r"""Selective Channel Fusion State-Space Block"""
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cfss = CFSS(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop = DropPath(drop_path)

        self.p = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, frame, image):
        ori_frame = frame
        frame = self.cfss(self.norm1(frame), self.norm2(image))
        out = ori_frame*self.p + self.drop(frame)
        return out

class SCFM(nn.Module):
    """ Selective Channel Fusion Mamba"""
    def __init__(
            self,
            dim,
            d_state,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.cf = ChannelTransfer(dim)
        self.scfb = SCFB(
            hidden_dim=dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop,
            d_state=d_state)
        self.cr = ChannelReturn(dim)

    def forward(self, frames, image):
        """
            Forward pass through the SCFM module.
            Args:
                frames (torch.Tensor): Input tensor with shape (batch_size, channels, num_frames, height, width).
                image (torch.Tensor): Single image tensor with shape (batch_size, channels, height, width).
            Returns:
                torch.Tensor: Processed frames tensor with the same shape as the input.
        """
        b, c, n, h, w = frames.shape
        image = self.cf(image)
        # Process each frame using the SCF blocks
        for i in range(n):
            frame = self.cf(frames[:, :, i, :, :].clone())
            frame = self.scfb(frame, image)
            frames[:, :, i, :, :] = self.cr(frame)
        return frames