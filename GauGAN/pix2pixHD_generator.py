#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/4
"""


import torch
from torch import nn
from network_utils import BaseNetwork
from torch.nn.utils import spectral_norm


class Pix2pixHDGenerator(BaseNetwork):
    def __init__(self, opt):
        # (label_nc, fnc, out_nc, resnet_n_downsample=4, resnet_n_blocks=9,
        #          resnet_kernel_size=3, resnet_initial_kernel_size=7,
        #          norm_type="instance", apply_spectral=True,)

        super().__init__()
        input_nc = opt.label_nc

        norm_layer = get_nonspade_norm_layer(norm_type=opt.norm_G, apply_spectral=opt.apply_spectral)
        activation = nn.ReLU(inplace=False)

        model = []

        # 初始化卷积层
        model += [nn.ReflectionPad2d(padding=opt.resnet_initial_kernel_size//2),
                  norm_layer(nn.Conv2d(input_nc, opt.fnc_g, kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # 下采样
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.fnc_g * mult, opt.fnc_g * mult * 2,
                                           kernel_size=3, stride=2, padding=1, padding_mode=opt.padding_mode)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.fnc_g * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # 上采样
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.fnc_g * mult)
            nc_out = int((opt.fnc_g * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # 最外层 conv 输出层
        model += [nn.ReflectionPad2d(padding=opt.resnet_initial_kernel_size//2),
                  nn.Conv2d(nc_out, opt.out_nc, kernel_size=opt.resnet_initial_kernel_size, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)




# 当使用pix2pixHD时, 生成一个不基于语义图的Normalization layer,
# 下面的函数返回一个可以输入previous layer 并返回指定类型Normalization layer的函数
def get_nonspade_norm_layer(norm_type='instance', apply_spectral=True):
    """
    norm_type:       返回的 Normalization layer 的类型, 请选择 instance | batch | none 其中一种
    apply_spectral:  选择是否对前一层使用 spectral Normalization
    """
    # 用来提取上一层 channel 数
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # 返回下面的函数
    def add_norm_layer(layer):
        nonlocal norm_type, apply_spectral
        if apply_spectral:
            layer = spectral_norm(layer)

        if norm_type == 'none':
            return layer

        # 将前一层中的 bias 删除, 因为这一参数在Normalize之后将会失效
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % norm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer




# 残差块单元, gaugan中的残差块与pix2pixHD的一致
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(inplace=False), kernel_size=3):
        super(ResnetBlock, self).__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


def main():
    from options import BaseOptions
    opt = BaseOptions()
    opt.initialize()
    g = Pix2pixHDGenerator(opt)
    x = torch.randn(2, 11, 256, 256)
    print(g(x).shape)


if __name__ == '__main__':
    main()