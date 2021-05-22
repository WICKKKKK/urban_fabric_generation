#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/3
"""


import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn.utils import spectral_norm
from network_utils import BaseNetwork


class SPADEGenerator(BaseNetwork):
    """
    构建SPADE生成器

    """
    def __init__(self, opt):
        # (fnc, image_size, z_dim, label_nc=3, apply_spectral=True, norm_type="batch",
        #          kernel_size=3, num_up_layers="normal", use_vae=True)
        """
        会用到的 opt 中的参数:
        fnc:                 最后一层的输出通道数
        image_size:           输出的图像尺寸
        z_dim:               z向量的维度
        label_nc:            语义图的通道数
        apply_spectral:      是否在SPADEResBlock中的卷积层后添加spectral Normalization
        norm_type:           SPADE中Normalization层的类型
        kernel_size:         SPADE中的所有卷积层的kernel大小
        num_up_layers:       上采样的层数, 请选择 normal | more | most 其中一种, 如果是normal, 则上采样只有5层;
                             如果是more, 则上采样有6层, 会在两个middle层之间添加;
                             如果是most, 则上采样有7层, 跟原paper上的框架一致, 在6层的基础上, 最后添加一个SPADEResBlock和上采样层
        use_vae:             是否使用vae来生成z向量, 如果是则会一起训练encoder网络, 训练好之后可以随意输入一个N(0, 1)的z向量来作为输入;
                             如果不是则使用语义图downsampling和卷积之后的feature map作为输入
        """
        super().__init__()

        self.fnc = opt.fnc_g
        self.use_vae = opt.use_vae
        self.z_dim = opt.z_dim
        self.device = opt.device

        self.num_up_layers = 5
        if opt.num_up_layers == "normal":
            self.num_up_layers = 5
        elif opt.num_up_layers == "more":
            self.num_up_layers = 6
        elif opt.num_up_layers == "most":
            self.num_up_layers = 7
        else:
            raise ValueError("num_upsampling_layers [%s] not recognized" %
                             opt.num_up_layers)

        self.s = opt.image_size // (2**self.num_up_layers)          # s为upsampling之前的latent feature map尺寸

        if self.use_vae:
            # 如果使用了vae, 将会从一个z向量中采样, 然后转化为指定维度的输入
            self.fc = nn.Linear(self.z_dim, 16*self.fnc*self.s*self.s)
        else:
            # 如果不使用vae, 则网络将从语义图经过下采样卷积之后得到指定维度的feature map作为输入
            self.fc = nn.Conv2d(opt.label_nc, 16*self.fnc, kernel_size=3, padding=1, padding_mode=opt.padding_mode)

        self.head_0 = SPADEResnetBlock(16*self.fnc, 16*self.fnc, label_nc=opt.label_nc, apply_spectral=opt.apply_spectral,
                                       norm_type=opt.norm_G, kernel_size=opt.kernel_size, padding_mode=opt.padding_mode)

        self.G_middle_0 = SPADEResnetBlock(16*self.fnc, 16*self.fnc, label_nc=opt.label_nc, apply_spectral=opt.apply_spectral,
                                       norm_type=opt.norm_G, kernel_size=opt.kernel_size, padding_mode=opt.padding_mode)
        self.G_middle_1 = SPADEResnetBlock(16*self.fnc, 16*self.fnc, label_nc=opt.label_nc, apply_spectral=opt.apply_spectral,
                                       norm_type=opt.norm_G, kernel_size=opt.kernel_size, padding_mode=opt.padding_mode)

        self.up_0 = SPADEResnetBlock(16*self.fnc, 8*self.fnc, label_nc=opt.label_nc, apply_spectral=opt.apply_spectral,
                                       norm_type=opt.norm_G, kernel_size=opt.kernel_size, padding_mode=opt.padding_mode)
        self.up_1 = SPADEResnetBlock(8*self.fnc, 4*self.fnc, label_nc=opt.label_nc, apply_spectral=opt.apply_spectral,
                                       norm_type=opt.norm_G, kernel_size=opt.kernel_size, padding_mode=opt.padding_mode)
        self.up_2 = SPADEResnetBlock(4*self.fnc, 2*self.fnc, label_nc=opt.label_nc, apply_spectral=opt.apply_spectral,
                                       norm_type=opt.norm_G, kernel_size=opt.kernel_size, padding_mode=opt.padding_mode)
        self.up_3 = SPADEResnetBlock(2*self.fnc, 1*self.fnc, label_nc=opt.label_nc, apply_spectral=opt.apply_spectral,
                                       norm_type=opt.norm_G, kernel_size=opt.kernel_size, padding_mode=opt.padding_mode)

        final_nc = self.fnc

        if self.num_up_layers == 7:
            self.up_4 = SPADEResnetBlock(1*self.fnc, self.fnc//2, label_nc=opt.label_nc, apply_spectral=opt.apply_spectral,
                                         norm_type=opt.norm_G, kernel_size=opt.kernel_size, padding_mode=opt.padding_mode)
            final_nc = self.fnc // 2

        self.conv_img = nn.Conv2d(final_nc, 3, kernel_size=3, padding=1, padding_mode=opt.padding_mode)

        self.up = nn.Upsample(scale_factor=2)
        self.actv = nn.Tanh()


    def forward(self, segmap, z=None):

        if self.use_vae:
            # 使用vae的情况, 将z向量经过linear层并且reshape成指定维度的tensor作为后续的输入
            if z is None:
                z = torch.randn(segmap.size(0), self.z_dim,
                                dtype=torch.float32, device=self.device)
            x = self.fc(z)
            x = x.view(-1, 16*self.fnc, self.s, self.s)
        else:
            # 不使用vae的情况, 从语义图中下采样, 并且通过卷积得到指定维度的tensor作为后续的输入
            x = F.interpolate(segmap, size=(self.s, self.s))
            x = self.fc(x)

        x = self.head_0(x, segmap)
        x = self.up(x)

        x = self.G_middle_0(x, segmap)

        if self.num_up_layers == 6 or self.num_up_layers == 7:
            x = self.up(x)

        x = self.G_middle_1(x, segmap)
        x = self.up(x)

        x = self.up_0(x, segmap)
        x = self.up(x)
        x = self.up_1(x, segmap)
        x = self.up(x)
        x = self.up_2(x, segmap)
        x = self.up(x)
        x = self.up_3(x, segmap)

        if self.num_up_layers == 7:
            x = self.up(x)
            x = self.up_4(x, segmap)

        x = self.conv_img(F.leaky_relu(x, 2e-1, inplace=False))
        x = self.actv(x)

        return x



# 使用了SPADE的残差块
# 与 pix2pixHD中的残差块不同之处在于这里将语义图作为输入, 将首先其应用于归一化层, 然后再进行卷积, 并且在必要的时候可以学习skip-connection
# 可以作为 unconditional or class-conditional GAN 架构的标准单元
# https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, in_nc, out_nc, label_nc, apply_spectral=True, norm_type="batch", kernel_size=3, padding_mode="reflect"):
        """
        in_nc:  输入的通道数
        out_nc: 输出的通道数
        label_nc: 语义图的通道数
        apply_spectral: 是否在卷积层之后应用spectral Normalization, 主要是为了增加训练稳定性, 因此默认打开
        norm_type: 指定 SPADE 单元中的 Normalization 类型
        kernel_size: 卷积层上的 kernel 大小
        """
        super().__init__()
        self.learned_shortcut = (in_nc != out_nc)    # 如果输入通道不等于输出通道, 则在短接通道中增加一层卷积以匹配
        middle_nc = min(in_nc, out_nc)

        self.conv_0 = nn.Conv2d(in_nc, middle_nc, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_1 = nn.Conv2d(middle_nc, out_nc, kernel_size=3, padding=1, padding_mode=padding_mode)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=False)      #使用1*1的kernel来匹配输入和输出通道数

        # 设置spectral norm, 限制卷积层的权重变化, 保证 Lipschitz 连续性, 增加生成器的训练稳定性
        if apply_spectral:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # 定义 Normalization 层
        self.norm_0 = SPADE(label_nc=label_nc, in_nc=in_nc, norm_type=norm_type, kernel_size=kernel_size, padding_mode=padding_mode)
        self.norm_1 = SPADE(label_nc=label_nc, in_nc=middle_nc, norm_type=norm_type, kernel_size=kernel_size, padding_mode=padding_mode)
        if self.learned_shortcut:
            self.norm_s = SPADE(label_nc=label_nc, in_nc=in_nc, norm_type=norm_type, kernel_size=kernel_size, padding_mode=padding_mode)


    def forward(self, x, segmap):
        # 需要用 语义图 作为 SPADE 的输入
        x_s = self.shortcut(x, segmap)

        dx = self.conv_0(self.actv(self.norm_0(x, segmap)))
        dx = self.conv_1(self.actv(self.norm_1(dx, segmap)))

        out = x_s + dx

        return out


    def shortcut(self, x, segmap):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actv(self.norm_s(x, segmap)))                  ## paper上在SPADE与conv之间有一个激活层, 但是source code中没有
        else:
            x_s = x
        return x_s

    def actv(self, x):
        return F.leaky_relu(x, 2e-1, inplace=False)



class SPADE(nn.Module):
    """
    包含两个步骤:
    首先是对已经激活的feature map进行归一化
    然后通过语义图得到的 γ和β, 用来对归一化的结果进行仿射变换
    """
    def __init__(self, label_nc, in_nc, norm_type="batch", kernel_size=3, padding_mode="reflect"):
        """
        label_nc: 输入语义图的通道数
        in_nc:    输入的用来归一化的激活的feature map的通道数
        norm_type: 指定归一化层的类型, 这里归一化层将不学习仿射变换, 请选择 batch | instance | none 其中一种
        kernel_size: SPADE中的kernel的大小
        """

        super().__init__()

        if norm_type == "instance":
            self.norm_layer = nn.InstanceNorm2d(in_nc, affine=False)
        elif norm_type == "batch":
            self.norm_layer = nn.BatchNorm2d(in_nc, affine=False)
        elif norm_type == "none":
            self.norm_layer = nn.Sequential()
        else:
            raise ValueError('%s is not a recognized norm type in SPADE'
                             % norm_type)


        # 中间embedding层的通道数. Yes, hardcoded.
        nhidden = 128
        pw = (kernel_size-1) // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=pw, padding_mode=padding_mode),
            nn.ReLU(inplace=False),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, in_nc, kernel_size=kernel_size, padding=pw, padding_mode=padding_mode)
        self.mlp_beta = nn.Conv2d(nhidden, in_nc, kernel_size=kernel_size, padding=pw, padding_mode=padding_mode)

    def forward(self, x, segmap):
        # Part 1. 生成 parameter-free normalized activations
        normalized = self.norm_layer(x)

        # Part 2. 从 semantic map 上得到 scaling 和 bias
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # 应用 scaling 和 bias
        out = normalized * (1 + gamma) + beta

        return out


def main():
    from options import BaseOptions
    opt = BaseOptions()
    opt.initialize()
    opt.apply_spectral = False
    opt.num_up_layers = "most"
    g = SPADEGenerator(opt).to(opt.device)
    summary(g, input_size=[(11, 256, 256)], batch_size=-1)

    num_params = 0
    for param in g.parameters():
        num_params += param.numel()  ## 返回当前层的参数总数
    print('Network [%s] was created. Total number of parameters: %0.3f m '
          'To see the architecture, do print(network).'
          % (type(g).__name__, num_params/1000000))

if __name__ == '__main__':
    main()