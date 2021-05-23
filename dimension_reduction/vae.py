#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/19
"""

import torch
from torch import nn
from torchsummary import summary


class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()

        self.layers = opt.layers
        self.s0 = opt.image_size//(2**(len(self.layers)-1))

        encoder_modules = []
        for i in range(len(self.layers)-1):
            encoder_modules += [nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i+1], kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
                                # nn.BatchNorm2d(self.layers[i+1]),   ## batch normalization 可加可不加
                                nn.ReLU(inplace=False),]

        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Linear(self.layers[-1] * self.s0 * self.s0, opt.z_dim)
        self.fc_var = nn.Linear(self.layers[-1] * self.s0 * self.s0, opt.z_dim)

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
        h = self.encoder(x).view(x.size(0), self.layers[-1]*self.s0*self.s0)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        kld = self.kld(mu, logvar)
        z_up = self.fc_z_up(z).view(x.size(0), self.layers[-1], self.s0, self.s0)
        x_hat = self.decoder(z_up)

        return x_hat, z, mu, logvar, kld

    def reparameterize(self, mu, logvar):
        """
        从 mu 和 logvar所代表的高斯分布下随机采样一个z向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def kld(self, mu, logvar):
        kld_loss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld_loss



def main():
    from options import Opt
    opt = Opt()
    vae = VAE(opt).to(opt.device)
    x = torch.randn(2, 3, 256, 256).to(opt.device)
    summary(vae, input_size=(3, 256, 256), batch_size=-1)



if __name__ == '__main__':
    main()