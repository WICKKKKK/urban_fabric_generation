#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/19
"""


import torch
from torch import nn
from torchsummary import summary


# 第二版 autoencoder, 借鉴了gaugan中encoder的设计
class AE(nn.Module):
    def __init__(self, opt):
        super(AE, self).__init__()

        self.layers = opt.layers
        self.s0 = opt.image_size//(2**(len(self.layers)-1))
        # print(self.s0)

        encoder_modules = []
        for i in range(len(self.layers)-1):
            encoder_modules += [nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i+1], kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
                                # nn.BatchNorm2d(self.layers[i+1]),   ## batch normalization 可加可不加
                                nn.ReLU(inplace=False),]

        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_z = nn.Linear(self.layers[-1] * self.s0 * self.s0, opt.z_dim)

        self.fc_z_up = nn.Linear(opt.z_dim, self.layers[-1] * self.s0 * self.s0)

        decoder_modules = []
        for i in range(len(self.layers)-1,1,-1):
            decoder_modules += [nn.ConvTranspose2d(in_channels=self.layers[i], out_channels=self.layers[i-1], kernel_size=4, stride=2, padding=1),
                                # nn.BatchNorm2d(self.layers[i-1]),
                                nn.ReLU(inplace=False), ]

        decoder_modules += [nn.ConvTranspose2d(in_channels=self.layers[1], out_channels=self.layers[0], kernel_size=4, stride=2, padding=1),
                            nn.Tanh(),]
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        h = self.encoder(x).view(-1, self.layers[-1]*self.s0*self.s0)
        z = self.fc_z(h)
        z_up = self.fc_z_up(z).view(-1, self.layers[-1], self.s0, self.s0)
        x_hat = self.decoder(z_up)

        return x_hat, z


def main():
    from options import Opt
    opt = Opt()
    ae = AE(opt).to(opt.device)
    summary(ae, input_size=(3, 256, 256), batch_size=-1)



if __name__ == '__main__':
    main()