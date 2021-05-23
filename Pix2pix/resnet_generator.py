#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/3/30
"""

import torch
from torch import nn
from torchsummary import summary


class ResNetGenerator(nn.Module):
    """
    按照CycleGAN的ResNetGenerator做了一定的微调(效果上差不多), 在经过一开始的通道转换之后, 会有两个下采样层,
    然后传入 n 个残差块, 接着两个上采样, 最后输出为指定的数据样式
    """
    def __init__(self, in_ch, out_ch, fch=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_block=6, padding_mode="reflect"):
        """
        构建一个 ResNet
        :param in_ch:          -- 输入通道数
        :param out_ch:         -- 输出通道数
        :param fch:            -- 第一层卷积输出的通道数
        :param norm_layer:     -- Normalization layer类型
        :param use_dropout:    -- 是否在残差块中添加dropout层
        :param n_block:        -- 残差块数量
        :param padding_mode:   -- padding的类型, 默认是reflect
        """
        super(ResNetGenerator, self).__init__()

        use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(in_ch, fch, kernel_size=7, padding=3, padding_mode=padding_mode, bias=use_bias),
                 norm_layer(fch),
                 nn.ReLU(True)]
        # 添加两个下采样层
        model += [nn.Conv2d(fch, fch*2, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=use_bias),
                  norm_layer(fch*2),
                  nn.ReLU(True)]
        model += [nn.Conv2d(fch*2, fch*2*2, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=use_bias),
                  norm_layer(fch*2*2),
                  nn.ReLU(True)]
        # 添加残差块
        for i in range(n_block):
            model += [ResNetBlock(fch*2*2, padding_mode=padding_mode, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]
        # 添加两个上采样层
        model += [nn.ConvTranspose2d(fch*2*2, fch*2,
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(fch*2),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(fch*2, fch,
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(fch),
                  nn.ReLU(True)]

        model += [nn.Conv2d(fch, out_ch, kernel_size=7, padding=3, padding_mode=padding_mode),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResNetBlock(nn.Module):
    """
    定义残差块
    """

    def __init__(self, dim, padding_mode="reflect", norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        """
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        在 build_conv_block 中实现残差块的主体部分
        在 forward 中实现skip connection

        :param dim: 输入的通道数
        :param padding_type: padding的类型
        :param norm_layer: Normalization的类型
        :param use_drop:  是否使用dropout层
        :param use_bias:  是否在Conv2d层中增加bias参数
        """
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_mode, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_mode, norm_layer, use_dropout, use_bias):
        """

        :param dim: 输入的通道数
        :param padding_mode: padding的类型
        :param norm_layer: Normalization的类型
        :param use_dropout: 是否使用dropout层
        :param use_bias: 是否在Conv2d层中增加bias参数
        :return:
        """

        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_mode),
                       norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_mode),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """

        :param x: 输入图片
        :return:
        """
        out  = x + self.conv_block(x)    # 添加 skip connection
        return out


def main():
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    net = ResNetGenerator(3, 3).to(device)
    summary(net, input_size=(3, 256, 256), batch_size=-1)



if __name__ == '__main__':
    main()