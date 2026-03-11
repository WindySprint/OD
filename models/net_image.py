import functools
import torch
import torch.nn as nn
import models.modules as ms

##### input:256*256*3, 256*256*1|output:256*256*3
class net_image(nn.Module):
    def __init__(self, n_feats=16):
        super(net_image, self).__init__()
        self.n_feats = n_feats
        self.conv_ini = nn.Sequential(nn.Conv2d(3, self.n_feats, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1, bias=True))
        self.rb1 = ms.RB(self.n_feats)

        self.down_conv1 = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats*2, 3, 2, 1),
                                        nn.LeakyReLU(0.1, True))
        self.down_conv2 = nn.Sequential(nn.Conv2d(self.n_feats*2, self.n_feats*4, 3, 2, 1),
                                        nn.LeakyReLU(0.1, True))

        basic_block1 = functools.partial(ms.RB, self.n_feats * 2)
        basic_block2 = functools.partial(ms.RB, self.n_feats * 4)
        basic_block3 = functools.partial(ms.RB, self.n_feats * 2)
        self.mrb1 = ms.make_layer(basic_block1, 2)
        self.mrb2 = ms.make_layer(basic_block2, 4)
        self.mrb3 = ms.make_layer(basic_block3, 2)

        self.up_conv1 = nn.Sequential(nn.PixelShuffle(2),
                                      nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(self.n_feats, self.n_feats * 2, 3, 1, 1))
        self.up_conv2 = nn.Sequential(nn.PixelShuffle(2),
                                      nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(self.n_feats//2, self.n_feats, 3, 1, 1))

        self.rb2 = ms.RB(self.n_feats)
        self.conv_last = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1, bias=True),
                                       nn.LeakyReLU(0.1, True),
                                       nn.Conv2d(self.n_feats, 3, 3, 1, 1, bias=True))

    def forward(self, x):
        # B, C, H, W
        e1 = self.conv_ini(x)
        e1 = self.rb1(e1)

        e2 = self.down_conv1(e1)
        e2 = self.mrb1(e2)

        e3 = self.down_conv2(e2)
        skip = self.mrb2(e3)
        ####################################

        d2 = self.up_conv1(skip) + e2
        d2 = self.mrb3(d2)

        d1 = self.up_conv2(d2) + e1
        d1 = self.rb2(d1)

        out = self.conv_last(d1)

        return (out, skip)