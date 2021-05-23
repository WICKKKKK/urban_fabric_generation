#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/3/30
"""

import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class MultiscaleDiscriminator(nn.Module):

    def __init__(self, in_ch=6, fnc=64, n_layers_D=4, norm_layer=nn.BatchNorm2d, padding_mode="reflect", ganFeat_loss=True, num_D=2):
        # (fnc=64, label_nc=10, output_nc=3, n_layers_D=4, norm_type="instance", ganFeat_loss=True, num_D=2, )
        super().__init__()

        self.ganFeat_loss = ganFeat_loss

        for i in range(num_D):
            subnetD = PatchDiscriminator(in_ch=in_ch, fch=fnc, n_layers_D=n_layers_D, norm_layer=norm_layer,
                                         padding_mode=padding_mode, ganFeat_loss=ganFeat_loss)
            self.add_module("discriminator_%d" % i, subnetD)


    # 对图像进行下采样, 生成不同尺度的图像, 给不同的判别器进行判别
    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=1,
                            count_include_pad=False)

    # 返回每个判别器的输出, 以list形式
    # 最终的list length 为 [num_D, n_layers_D]
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




## 判别器 PatchGAN
## 输入: 条件图像 和 生成图像/真实图像 在通道维度上进行拼合[b, 6, 256, 256]
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6, fch=64, n_layers_D=4, norm_layer=nn.BatchNorm2d, padding_mode="reflect", ganFeat_loss=True):
        """
        :param in_ch: 输入通道数
        :param fch: 第一层卷积的通道数
        """
        super(PatchDiscriminator, self).__init__()

        use_bias = norm_layer == nn.InstanceNorm2d

        self.ganFeat_loss = ganFeat_loss

        # 输出是一个 N*N的Patch矩阵(这里为30*30), 其中每一块对应输入数据的一小块
        # in_ch 表示将 条件图像 和 生成图像/真实图像 在通道维度上进行拼合后的图像输入
        # [b, 6, 256, 256]

        sequence = [[nn.Conv2d(in_ch, fch, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode),
                     nn.LeakyReLU(0.2, inplace=False)]]

        for n in range(1, n_layers_D):
            nf_prev = fch
            fch = min(fch * 2, 512)
            stride = 1 if (n == (n_layers_D - 1)) else 2
            sequence += [[nn.Conv2d(nf_prev, fch, kernel_size=4,
                                    stride=stride, padding=1, padding_mode=padding_mode),
                          norm_layer(fch),
                          nn.LeakyReLU(0.2, inplace=False)]]

        sequence += [[nn.Conv2d(fch, 1, kernel_size=4, stride=1, padding=1, padding_mode=padding_mode)]]

        # 上面的 sequence 模型按照卷积层打成组的原因是后面需要抽取中间层的输出结果来计算 feature matching loss
        for n in range(len(sequence)):
            self.add_module("model" + str(n), nn.Sequential(*sequence[n]))


    def forward(self, x):
        """
        :param x: 输入图像 [b, 3, 256, 256]
        :return:
        """
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
    ### 测试代码
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    D = MultiscaleDiscriminator().to(device)
    summary(D, input_size=(6, 256, 256), batch_size=-1)

if __name__ == '__main__':
    main()