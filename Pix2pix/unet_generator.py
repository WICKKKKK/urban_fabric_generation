#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/3/30
"""

import torch
from torch import nn
from torchsummary import summary


## 生成器 U-Net (输入: [b, 3, 256, 256])
class UNetGenerator(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, fch=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_mode="reflect"):
        """
        定义生成器的网络结构
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param fch: 第一层卷积的通道数 number of first conv channels
        """
        super(UNetGenerator, self).__init__()

        use_bias = norm_layer == nn.InstanceNorm2d

        # 激活函数放在每个模块的第一步, 为了skip-connect方便
        # U-Net, encoder
        # 每一层图像长宽折半, 通道数量增加一倍
        # [b, 3, 256, 256]
        self.en1 = nn.Sequential(
            nn.Conv2d(in_ch, fch, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode),
            # 输入图片已归一化, 不需要BatchNorm
        )
        # [b, 64, 128, 128]
        self.en2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fch, fch * 2, kernel_size=4, stride=2, padding=1, bias=use_bias, padding_mode=padding_mode),
            norm_layer(fch * 2),
        )
        # [b, 128, 64, 64]
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fch * 2, fch * 4, kernel_size=4, stride=2, padding=1, bias=use_bias, padding_mode=padding_mode),
            norm_layer(fch * 4),
        )
        # [b, 256, 32, 32]
        self.en4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fch * 4, fch * 8, kernel_size=4, stride=2, padding=1, bias=use_bias, padding_mode=padding_mode),
            norm_layer(fch * 8),
        )
        # [b, 512, 16, 16]
        self.en5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fch * 8, fch * 8, kernel_size=4, stride=2, padding=1, bias=use_bias, padding_mode=padding_mode),
            norm_layer(fch * 8),
        )
        # [b, 512, 8, 8]
        self.en6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fch * 8, fch * 8, kernel_size=4, stride=2, padding=1, bias=use_bias, padding_mode=padding_mode),
            norm_layer(fch * 8),
        )
        # [b, 512, 4, 4]
        self.en7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fch * 8, fch * 8, kernel_size=4, stride=2, padding=1, bias=use_bias, padding_mode=padding_mode),
            norm_layer(fch * 8),
        )
        # [b, 512, 2, 2]
        self.en8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fch * 8, fch * 8, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode),
            # Encoder输出不需要BatchNorm
        )


        # U-Net decoder
        # skip-connect

        de1_model = [nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(fch * 8, fch * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(fch * 8)]

        de2_model = [nn.ReLU(inplace=True),
                     # 加入skip-connection, 通道数量加倍
                    nn.ConvTranspose2d(fch * 8 * 2, fch * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(fch * 8)]

        de3_model = [nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(fch * 8 * 2, fch * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(fch * 8)]

        if use_dropout:
            de1_model.append(nn.Dropout(p=0.5))
            de2_model.append(nn.Dropout(p=0.5))
            de3_model.append(nn.Dropout(p=0.5))

        # [b, 512, 1, 1]
        self.de1 = nn.Sequential(*de1_model)
        #[b, 1024, 2, 2]
        self.de2 = nn.Sequential(*de2_model)
        # [b, 1024, 4, 4]
        self.de3 = nn.Sequential(*de3_model)
        # [b, 1024, 8, 8]
        self.de4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fch * 8 * 2, fch * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(fch * 8),
            # nn.Dropout(p=0.5),
        )
        # [b, 1024, 16, 16]
        self.de5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fch * 8 * 2, fch * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(fch * 4),
            # nn.Dropout(p=0.5),
        )
        # [b, 512, 32, 32]
        self.de6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fch * 4 * 2, fch * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(fch * 2),
            # nn.Dropout(p=0.5),
        )
        # [b, 256, 64, 64]
        self.de7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fch * 2 * 2, fch, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(fch),
            # nn.Dropout(p=0.5),
        )
        # [b, 128, 128, 128]
        self.de8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fch * 2, out_ch, kernel_size=4, stride=2, padding=1),
            # Decoder输出不需要BatchNorm
            nn.Tanh(),
        )
        # 输出: [b, 3, 256, 256]

    def forward(self, x):
        """
        生成器前向传播
        :param x: [b, 3, 256, 256]
        :return:
        """
        # Encoder
        en1_out = self.en1(x)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)
        en4_out = self.en4(en3_out)
        en5_out = self.en5(en4_out)
        en6_out = self.en6(en5_out)
        en7_out = self.en7(en6_out)
        en8_out = self.en8(en7_out)

        # print(en8_out.shape)
        # Decoder
        de1_out = self.de1(en8_out)
        de1_cat = torch.cat([de1_out, en7_out], dim=1)
        de2_out = self.de2(de1_cat)
        de2_cat = torch.cat([de2_out, en6_out], dim=1)
        de3_out = self.de3(de2_cat)
        de3_cat = torch.cat([de3_out, en5_out], dim=1)
        de4_out = self.de4(de3_cat)
        de4_cat = torch.cat([de4_out, en4_out], dim=1)
        de5_out = self.de5(de4_cat)
        de5_cat = torch.cat([de5_out, en3_out], dim=1)
        de6_out = self.de6(de5_cat)
        de6_cat = torch.cat([de6_out, en2_out], dim=1)
        de7_out = self.de7(de6_cat)
        de7_cat = torch.cat([de7_out, en1_out], dim=1)
        de8_out = self.de8(de7_cat)

        return de8_out


def main():
    ### 测试代码
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    G = UNetGenerator(3, 3, fch=64, use_dropout=False).to(device)
    summary(G, input_size=(3, 256, 256), batch_size=-1)

if __name__ == '__main__':
    main()