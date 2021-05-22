#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/3

部署基础网络, 包含一些网络架构中的小功能, 如打印网络参数, 网络权重初始化
载入VGG19网络
定义GAN损失函数(包含常规GAN 和 LSGAN)
定义VGG损失函数
定义KLD损失函数
"""


import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn import init
from torchsummary import summary
import torchvision
import os


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()                           ## 返回当前层的参数总数
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        self.apply(init_func)




class GANLoss(nn.Module):
    """
    定义GAN损失函数, 包含 LSGAN 和 常规的GAN损失函数
    LSGAN损失函数, 主体是 MSELoss, 主要的改变是不需要创建与 input 相同size的 output
    """
    def __init__(self, gan_loss_mode="hinge", target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_loss_mode = gan_loss_mode

        if gan_loss_mode in ["lsgan", "original", "hinge"]:
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_loss_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_loss_mode == "original":
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_loss_mode == "lsgan":
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.mse_loss(input, target_tensor)
            return loss
        elif self.gan_loss_mode == "hinge":
            # print(input.shape, target_is_real, for_discriminator)
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input-1, self.get_zero_tensor(input))        ## 大于0还不够, 需要大于1, loss才能等于0
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input-1, self.get_zero_tensor(input))       ## 这里fake_pred需要足够小于-1才能够收敛
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"     # 对于生成器, 计算hinge损失时, 必须设置target_is_real为true
                loss = -torch.mean(input)                                           ## loss值这里设置越大越好
            return loss
        else:
            #wgangp, 未写完
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # 可能会有两种类型的 输入, 一种是单个tensor, 另一种是包含在list中tensor(来自multiscale discriminator)
        # 这里注意, GAN_feat loss并不在这里计算, 而是在gaugan_model中通过L1Loss计算, 这里只计算最后一层的输出(如果是多个判别器, 则求平均)
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                batch_size = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(batch_size, -1), dim=1)
                loss += new_loss
            return loss/len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)




#VGG19网络, 用来计算perceptual loss
class VGG19(nn.Module):
    def __init__(self, model_path="./", requires_grad=False):
        super(VGG19, self).__init__()
        # 先导入vgg19模型,
        vgg = torchvision.models.vgg19(pretrained=False, progress=True)
        # 如果本地有下好的模型, 可以根据 model_path 载入预训练的模型参数
        if os.path.isfile(model_path):
            vgg.load_state_dict(torch.load(model_path))
            print("load pretrained vgg19 model parameters successfully!")
        elif os.path.isdir(model_path):
            url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
            model_name = url.split("/")[-1]
            new_model_path = os.path.join(model_path, model_name)
            if os.path.exists(new_model_path):
                vgg.load_state_dict(torch.load(new_model_path))
                print("load pretrained vgg19 model parameters successfully!")
            else:
                # 如果本地没有训练好的vgg19, 则自动将模型下载到 model_path 目录下
                print("no pretrained vgg19 model in '%s', starting download model from %s" % (model_path, url))
                state_dict = torch.utils.model_zoo.load_url(url, model_dir=model_path, map_location=None, progress=True)
                # state_dict = torch.hub.load_state_dict_from_url(url, model_dir=model_path, map_location=None, progress=True)
                vgg.load_state_dict(state_dict)
                print("finish downloading and load pretrained vgg19 model parameters successfully!")
        else:
            print('Unexpected model_path: {}'.format(model_path))

        # print(vgg.features[0].weight)
        vgg_pretrained_features = vgg.features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):                      ## 将VGG19中的0-1层加入进来  c3s1p1-64 => ReLU
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):                   ## 将VGG19中的2-6层加入进来  c3s1p1-64 => ReLU => MaxP => c3s1p1-128 => ReLU
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):                  ## 将VGG19中的7-11层加入进来  c3s1p1-128 => ReLU => MaxP => c3s1p1-256 => ReLU
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):                 ## 将VGG19中的12-20层加入进来   c3s1p1-256 => ReLU => c3s1p1-256 => ReLU => c3s1p1-256 => ReLU => MaxP => c3s1p1-512 => ReLU
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):                 ## 将VGG19中的21-29层加入进来   c3s1p1-512 => ReLU => c3s1p1-512 => ReLU => c3s1p1-512 => ReLU => MaxP => c3s1p1-512 => ReLU
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:                   ## 设置不需要计算梯度
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out



class VGGLoss(nn.Module):
    def __init__(self, opt):
        """
        利用vgg来计算不同层的 L1Loss
        """
        super().__init__()
        self.vgg = VGG19(opt.vgg_model_path).to(opt.device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())



def main():
    vgg = VGG19(model_path="./").to(torch.device("cuda: 0"))
    summary(vgg, input_size=(3, 256, 256), batch_size=-1)


if __name__ == '__main__':
    main()