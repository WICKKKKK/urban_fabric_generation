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
from torchsummary import summary
from network_utils import BaseNetwork
from pix2pixHD_generator import get_nonspade_norm_layer

class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, opt):
        # (fnc=64, label_nc=10, output_nc=3, n_layers_D=4, norm_type="instance", ganFeat_loss=True, num_D=2, )
        super().__init__()

        self.ganFeat_loss = opt.ganFeat_loss

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module("discriminator_%d" % i, subnetD)


    def create_single_discriminator(self, opt):
        return NLayerDiscriminator(opt)

    # 对图像进行下采样, 生成不同尺度的图像, 给不同的判别器进行判别
    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=1,
                            count_include_pad=False)

    # 返回每个判别器的输出, 以list形式
    # 最终的list length 为 num_D * n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = self.ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result




class NLayerDiscriminator(BaseNetwork):
    """
    与pix2pix中一致的patchGAN架构
    """
    def __init__(self, opt):
        """
        fnc:            第一层卷积层输出的通道数
        label_nc:       语义图的通道数
        output_nc:      生成图/真实图 的通道数
        n_layers_D:     判别器卷积层的数目, 默认为4
        norm_type:      Normalization层的类型
        ganFeat_loss:   是否计算判别器中不同层的 feature matching loss
        """
        super().__init__()

        self.ganFeat_loss = opt.ganFeat_loss

        kw = 4
        padw = (kw-1) // 2
        fnc = opt.fnc_d
        input_nc = opt.label_nc + opt.out_nc

        norm_layer = get_nonspade_norm_layer(norm_type=opt.norm_D, apply_spectral=opt.apply_spectral)
        sequence = [[nn.Conv2d(input_nc, fnc, kernel_size=kw, stride=2, padding=padw, padding_mode=opt.padding_mode),
                     nn.LeakyReLU(0.2, inplace=False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = fnc
            fnc = min(fnc * 2, 512)
            stride = 1 if (n == (opt.n_layers_D-1)) else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, fnc, kernel_size=kw,
                                               stride=stride, padding=padw, padding_mode=opt.padding_mode)),
                          nn.LeakyReLU(0.2, inplace=False)]]

        sequence += [[nn.Conv2d(fnc, 1, kernel_size=kw, stride=1, padding=padw, padding_mode=opt.padding_mode)]]

        # 上面的 sequence 模型按照卷积层打成组的原因是后面需要抽取中间层的输出结果来计算 feature matching loss
        for n in range(len(sequence)):
            self.add_module("model"+str(n), nn.Sequential(*sequence[n]))


    def forward(self, x):
        results = [x]
        for submodule in self.children():
            intermediate_output = submodule(results[-1])
            results.append(intermediate_output)
        # 选择是否输出中间层feature, 用来计算ganFeat_loss
        get_intermediate_features = self.ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


def main():
    from options import TrainOptions
    opt = TrainOptions()
    opt.initialize()
    D = MultiscaleDiscriminator(opt).cuda()

    summary(D, input_size=(14, 256, 256), batch_size=-1)
    x = torch.randn(2, 14, 256, 256).cuda()
    print(D(x)[-1][0].shape)



if __name__ == '__main__':
    main()