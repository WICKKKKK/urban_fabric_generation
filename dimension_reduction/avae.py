#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/20
"""

import torch
from torch import nn


class AVAE(nn.Module):
    
    def __init__(self, opt):
        super(AVAE, self).__init__()

        self.layers = opt.layers
        self.s0 = opt.image_size//(2**(len(self.layers)-1))

        encoder_modules = []
        for i in range(len(self.layers)-1):
            encoder_modules += [nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i+1], kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
                                nn.BatchNorm2d(self.layers[i+1]),   ## batch normalization 可加可不加
                                nn.ReLU(inplace=False),]
        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_z = nn.Linear(self.layers[-1] * self.s0 * self.s0, opt.z_dim)

        self.fc_z_up = nn.Linear(opt.z_dim, self.layers[-1] * self.s0 * self.s0)

        decoder_modules = []
        for i in range(len(self.layers) - 1, 1, -1):
            decoder_modules += [
                nn.ConvTranspose2d(in_channels=self.layers[i], out_channels=self.layers[i - 1], kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(self.layers[i-1]),
                nn.ReLU(inplace=False), ]

        decoder_modules += [nn.ConvTranspose2d(in_channels=self.layers[1], out_channels=self.layers[0], kernel_size=4, stride=2, padding=1),
                            nn.Tanh(), ]
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        h = self.encoder(x).view(-1, self.layers[-1] * self.s0 * self.s0)
        z = self.fc_z(h)
        z_up = self.fc_z_up(z).view(-1, self.layers[-1], self.s0, self.s0)
        x_hat = self.decoder(z_up)

        return x_hat, z


class Discriminatr_z(nn.Module):
    def __init__(self, opt):
        super(Discriminatr_z, self).__init__()

        self.discriminatr = nn.Sequential(
            nn.Linear(opt.z_dim, opt.inter_dim),
            nn.ReLU(inplace=False),
            nn.Linear(opt.inter_dim, opt.inter_dim),
            nn.ReLU(inplace=False),
            # nn.Linear(opt.inter_dim, opt.inter_dim),
            # nn.ReLU(inplace=False),
            nn.Linear(opt.inter_dim, 1)
        )

    def forward(self, z):
        out = self.discriminatr(z)
        return out

