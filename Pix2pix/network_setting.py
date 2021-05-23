#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/3/30

定义生成器和判别器

"""

import torch
from torch import nn, autograd
from torch.nn import init
import functools
from torch.optim import lr_scheduler



def get_norm_layer(norm_type='instance'):
    """返回一个Normalization层

    Parameters:
        norm_type (str) -- Normalization层的名称: batch | instance | none

    BatchNorm, 学习mean和std参数
    InstanceNorm, 只是简单的归一化
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = nn.Sequential()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """初始化网络权重.

    Parameters:
        net (network)   -- 用于初始化的网络
        init_type (str) -- 初始化方法: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- 对于 normal, xavier and orthogonal方法的scale系数.

    原论文中默认使用normal, 但可能xavier和kaiming在某些方面会更加优越
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNormalization的参数并不是一个矩阵, 因此只需要normal distribution即可
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)                  ##递归地对网络进行参数初始化



def init_net(net, init_type='normal', init_gain=0.02, device=torch.device("cpu")):
    """初始化网络: 1. 将网络配置到cuda上; 2. 初始化网络权重
    Parameters:
        net (network)      -- 网络
        init_type (str)    -- 初始化权值方法: normal | xavier | kaiming | orthogonal
        gain (float)       -- 对于 normal, xavier and orthogonal方法的scale系数.
        device             -- torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    """
    net.to(device)

    init_weights(net, init_type, init_gain=init_gain)
    return net



def get_scheduler(optimizer, lr_policy="linear", start_epoch=1, n_epochs=100, n_epochs_decay=100, lr_decay_iters=50):
    """优化器中的学习率调节器

    Parameters:
        optimizer          -- 网络中的优化器
        lr_policy          -- 学习率变化策略 : linear | step | plateau | cosine
        start_epoch        -- 开始的 epoch (之前有训练过的模型载入时, 即并非从第 1 epoch开始, 需要设定这个参数)
        n_epochs           -- 保持初始学习率的 epoch 数
        n_epochs_decay     -- 学习率衰减的 epoch 数
        lr_decay_iters     -- step衰减中, 会阶梯式地对学习率进行衰减(乘以gamma值), 这个参数用于每个阶梯的 epoch 数


    默认是 linear , 一开始在 n_epochs 时间段内会保持初始的学习率, 然后在 n_epoch_decay 时间段内会以线性的方式进行衰减
    其他的 学习率策略, 保持pytorch默认设定(https://pytorch.org/docs/stable/optim.html)
    """
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + start_epoch - n_epochs) / float(n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


# 判别器训练, 损失函数计算
def D_train(netD, netG, X, GANLoss, optimizer_D, device=torch.device("cpu")):
    """
    训练判别器
    :param D: 判别器
    :param G: 生成器
    :param X: 未分割的图像
    :param BCELoss: 二分交叉熵损失函数
    :param optimizer_D: 判别器优化器
    :return: 判别器损失值
    """
    # 标签转实物 (左转右)
    image_size = X.size(3) // 2
    x = X[:, :, :, :image_size]         # 标签图 (右半部分)
    y = X[:, :, :, image_size:]         # 实物图 (左半部分)
    xy = torch.cat([x, y], dim=1)       # 在channel维将标签和实物叠加

    for param in netD.parameters():
        param.requires_grad = True
    netD.zero_grad()   # 梯度初始化为0

    # 在真实数据上
    D_output_r = netD(xy)
    if isinstance(D_output_r, list):
        loss = 0
        for pred_i in D_output_r:
            if isinstance(pred_i, list):
                pred_i = pred_i[-1]
            loss_tensor = GANLoss(pred_i, torch.ones(pred_i.size()).to(device))
            batch_size = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
            new_loss = torch.mean(loss_tensor.view(batch_size, -1), dim=1)
            loss += new_loss
        D_real_loss = loss / len(D_output_r)
    else:
        D_real_loss = GANLoss(D_output_r, torch.ones(D_output_r.size()).to(device))
    # 在生成数据上
    G_output = netG(x)
    xy_fake = torch.cat([x, G_output], dim=1)
    D_output_f = netD(xy_fake.detach())
    if isinstance(D_output_f, list):
        loss = 0
        for pred_i in D_output_f:
            if isinstance(pred_i, list):
                pred_i = pred_i[-1]
            loss_tensor = GANLoss(pred_i, torch.zeros(pred_i.size()).to(device))
            batch_size = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
            new_loss = torch.mean(loss_tensor.view(batch_size, -1), dim=1)
            loss += new_loss
        D_fake_loss = loss / len(D_output_f)
    else:
        D_fake_loss = GANLoss(D_output_f, torch.zeros(D_output_f.size()).to(device))
    # 反向传播并优化
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()

    return D_loss.data.item()

# 生成器训练, 损失函数计算
def G_train(netD, netG, X, GANLoss, L1, optimizer_G, num_D, gan_feat_loss=True, lamb=100, device=torch.device("cpu")):
    """
    :param netD: 判别器
    :param netG: 生成器
    :param X: 未分割的图片
    :param BCELoss: 二分交叉熵损失函数
    :param L1: L1损失函数
    :param optimizer_G: 生成器优化器
    :param lamb: L1损失权重
    :return: 生成器损失值
    """
    # 标签转实物 (左转右)
    image_size = X.size(3) // 2
    x = X[:, :, :, :image_size]      #标签图 (左半部分)
    y = X[:, :, :, image_size:]      #实物图 (右半部分)
    # 梯度初始化为0
    for param in netD.parameters():
        param.requires_grad = False
    netG.zero_grad()
    # 在假数据上
    G_output = netG(x)
    xy_fake = torch.cat([x, G_output], dim=1)
    xy_real = torch.cat([x, y], dim=1)
    D_output_f = netD(xy_fake)
    D_output_r = netD(xy_real)
    if isinstance(D_output_f, list):
        loss = 0
        for pred_i in D_output_f:
            if isinstance(pred_i, list):
                pred_i = pred_i[-1]
            loss_tensor = GANLoss(pred_i, torch.ones(pred_i.size()).to(device))
            batch_size = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
            new_loss = torch.mean(loss_tensor.view(batch_size, -1), dim=1)
            loss += new_loss
        GAN_loss = loss / len(D_output_f)
    else:
        GAN_loss = GANLoss(D_output_f, torch.ones(D_output_f.size()).to(device))

    if gan_feat_loss:
        G_L1_loss = torch.FloatTensor(1).fill_(0).to(device)
        if num_D > 1 and isinstance(num_D, int):
            for i in range(num_D):
                # 中间层不包含最后的预测层, 因此减一
                num_intermediate_outputs = len(D_output_f[i]) - 1
                for j in range(num_intermediate_outputs):
                    unweighted_loss = L1(D_output_f[i][j], D_output_r[i][j].detach())
                    G_L1_loss += unweighted_loss / num_D
        elif num_D == 1:
            # 中间层不包含最后的预测层, 因此减一
            num_intermediate_outputs = len(D_output_f) - 1
            for j in range(num_intermediate_outputs):
                unweighted_loss = L1(D_output_f[j], D_output_r[j].detach())
                G_L1_loss += unweighted_loss / num_D
        else:
            raise NotImplementedError('num_D is not corrected')
    else:
        G_L1_loss = L1(G_output, y)


    # 反向传播并优化
    G_loss = GAN_loss + lamb * G_L1_loss
    G_loss.backward()
    optimizer_G.step()

    return G_loss.data.item()


# 定义WGAN-GP损失函数(用于后面更换L1损失, 但实验证明效果并没有那么好)
def gradient_penalty(netD, x_r, x_f, device=torch.device("cpu")):
    """
    x_r: [b, 2]
    x_f: [b, 2]
    """
    # 随机sample 一个均值分布, [b,1]
    t = torch.rand(x_r.size(0), 1).to(device)
    # [b, 1] => [b, 2]
    t = t.expand_as(x_r)
    # interpolate
    mid = t * x_r + (1- t) * x_f
    # 设置mid需要计算导数信息
    mid.requires_grad_(True)

    pred = netD(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred).to(device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
                        ##create_graph, 是否要二级求导; retain_graph, 如果还需要backpropagate一次, 就需要把图保留下来
    grads = grads.view(x_r.size(0), -1)
    gp = torch.pow((grads + 1e-16).norm(2, dim=1)-1., 2).mean()

    return gp




def main():
    ### 测试代码
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    main()