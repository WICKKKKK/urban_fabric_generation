#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/4
"""


import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from network_utils import BaseNetwork
from pix2pixHD_generator import get_nonspade_norm_layer
from torchsummary import summary


class ConvEncoder(BaseNetwork):
    """
    与判别器是一样的架构
    """

    def __init__(self, opt):
        # (image_size=256, z_dim, fnc=64, norm_type="batch", apply_spectral=True):
        super().__init__()

        self.image_size = opt.image_size

        kw = 3
        pw = (kw-1) // 2
        fnc = opt.fnc_e
        norm_layer = get_nonspade_norm_layer(norm_type=opt.norm_E, apply_spectral=opt.apply_spectral)
        self.layer1 = norm_layer(nn.Conv2d(3, fnc, kernel_size=kw, stride=2, padding=pw, padding_mode=opt.padding_mode))
        self.layer2 = norm_layer(nn.Conv2d(fnc * 1, fnc * 2, kernel_size=kw, stride=2, padding=pw, padding_mode=opt.padding_mode))
        self.layer3 = norm_layer(nn.Conv2d(fnc * 2, fnc * 4, kernel_size=kw, stride=2, padding=pw, padding_mode=opt.padding_mode))
        self.layer4 = norm_layer(nn.Conv2d(fnc * 4, fnc * 8, kernel_size=kw, stride=2, padding=pw, padding_mode=opt.padding_mode))
        self.layer5 = norm_layer(nn.Conv2d(fnc * 8, fnc * 8, kernel_size=kw, stride=2, padding=pw, padding_mode=opt.padding_mode))
        if self.image_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(fnc * 8, fnc * 8, kernel_size=kw, stride=2, padding=pw, padding_mode=opt.padding_mode))

        self.s0 = 4
        self.fc_mu = nn.Linear(fnc * 8 * self.s0 * self.s0, opt.z_dim)
        self.fc_var = nn.Linear(fnc * 8 * self.s0 * self.s0, opt.z_dim)

        self.actvn = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=True)

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.image_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar


def main():
    from options import BaseOptions
    opt = BaseOptions()
    opt.initialize()
    e = ConvEncoder(opt).to(torch.device("cuda:0"))

    summary(e, input_size=(3, 512, 512), batch_size=-1)


if __name__ == '__main__':
    main()