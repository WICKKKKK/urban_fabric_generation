#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/3/29

网络架构
UNet
PatchGAN
"""

import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim, autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchsummary import summary
import visdom

from dataset import MyDataset
from unet_generator import UNetGenerator
from resnet_generator import ResNetGenerator
from discriminator import PatchDiscriminator, MultiscaleDiscriminator
from network_setting import init_net, get_norm_layer, get_scheduler, gradient_penalty, G_train, D_train
from pix2pix_options import Opt


# 主函数: 训练Pix2Pix网络 #
def main():
    # torch.manual_seed(23)
    # np.random.seed(23)

    opt = Opt()

    if torch.cuda.is_available():
        print(" -- 使用GPU进行训练 -- ")


    if not os.path.exists(opt.train_img_save_path):
        os.mkdir(opt.train_img_save_path)
    if not os.path.exists(opt.test_img_save_path):
        os.mkdir(opt.test_img_save_path)

    # 加载训练数据
    train_dataset = MyDataset(opt.data_root, opt.train_subfolder, device=opt.device)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    # 加载测试数据
    test_dataset = MyDataset(opt.data_root, opt.test_subfolder, device=opt.device)

    # 生成器, 判别器
    if opt.G_type == "UNet":
        G = UNetGenerator(in_ch=opt.in_ch, out_ch=opt.out_ch, fch=opt.fch_g,
                          norm_layer=get_norm_layer(opt.norm_type), use_dropout=opt.use_dropout,
                          padding_mode=opt.padding_mode)
    elif opt.G_type == "ResNet":
        G = ResNetGenerator(in_ch=opt.in_ch, out_ch=opt.out_ch, fch=opt.fch_g,
                            norm_layer=get_norm_layer(opt.norm_type), use_dropout=opt.use_dropout,
                            n_block = opt.n_block, padding_mode = opt.padding_mode)
    else:
        raise NotImplementedError('generator method [%s] is not implemented' % opt.G_type)
    G = init_net(G, init_type=opt.init_type, init_gain=opt.init_gain, device=opt.device)
    D = MultiscaleDiscriminator(in_ch=opt.in_ch*2, fnc=opt.fch_d, n_layers_D=4, norm_layer=get_norm_layer(opt.norm_type),
                                padding_mode=opt.padding_mode, ganFeat_loss=opt.gan_feat_loss, num_D=opt.num_D)
    D = init_net(D, init_type=opt.init_type, init_gain=opt.init_gain, device=opt.device)

    # 目标损失函数, 优化器
    GANLoss = nn.BCEWithLogitsLoss().to(opt.device)   ## discriminator最后没有加sigmoid, 因此这里需要用BCEWithLogitsLoss()
    # GANLoss = nn.MSELoss().to(opt.device)           ## lsgan(CycleGAN)
    L1 = nn.L1Loss().to(opt.device)    # Pix2Pix论文中在传统GAN目标函数加上了L1
    optimizer_G = optim.Adam(G.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))

    D_Loss, G_Loss, Epochs = [], [], range(1, opt.epochs + 1)  # 记录数据的容器

    # 预训练模型加载
    if opt.preload:
        print("loading model from " + opt.model_name)
        checkpoint = torch.load(opt.preload_model_path)
        G.load_state_dict(checkpoint["G_state_dict"])
        D.load_state_dict(checkpoint["D_state_dict"])
        optimizer_G.load_state_dict((checkpoint["optimizerG_state_dict"]))
        optimizer_D.load_state_dict((checkpoint["optimizerD_state_dict"]))
        opt.start_epoch = checkpoint["epoch"]
        opt.step = checkpoint["step"]
        D_Loss = checkpoint["D_Loss"]
        G_Loss = checkpoint["G_Loss"]
        print("preload successfully!")

    # 定义学习率调节器
    scheduler_G = get_scheduler(optimizer=optimizer_G, lr_policy=opt.lr_policy, start_epoch=opt.start_epoch,
                                n_epochs=opt.keep_lr_epochs,
                                n_epochs_decay= opt.epochs-opt.keep_lr_epochs)
    scheduler_D = get_scheduler(optimizer=optimizer_D, lr_policy=opt.lr_policy, start_epoch=opt.start_epoch,
                                n_epochs=opt.keep_lr_epochs,
                                n_epochs_decay= opt.epochs-opt.keep_lr_epochs)

    # 第一批次输入数据 & ground-truth & 初始生成器的输出
    if not opt.preload:
        X = next(iter(train_loader))
        g = G(X[:, :, :, :opt.image_size])
        g = g.view(opt.batch_size, opt.in_ch, opt.image_size, opt.image_size)
        img_all = train_dataset.denormalize(torch.cat([X, g], dim=3))
        save_image(img_all, opt.train_img_save_path + 'epoch_0.png', nrow=1)

    # 随机选择测试集中的n(8)张来进行生成测试
    imgs_list = []
    for i in np.random.choice(len(test_dataset), opt.test_pick):
        imgs_list.append(test_dataset[i][0].unsqueeze(0))
    test_imgs = torch.cat(imgs_list, dim=0)


    #训练
    G.train()  #
    D.train()  # 训练模式(区分 eval())

    for epoch in range(opt.start_epoch, opt.epochs+1):
        D_losses, G_losses, batch, d_l, g_l = [], [], 0, 0, 0  # 每个epoch的数据记录容器
        for X in train_loader:
            # 每个批次
            batch += 1
            opt.step += X.size(0)
            D_losses.append(D_train(D, G, X, GANLoss, optimizer_D, device=opt.device))        # 计算D损失
            for i in range(opt.G_per_steps):
                G_losses.append(G_train(D, G, X, GANLoss, L1, optimizer_G, num_D=opt.num_D, gan_feat_loss=opt.gan_feat_loss,lamb=opt.lamb, device=opt.device))  # 计算G损失
            d_l, g_l = np.array(D_losses).mean(), np.array(G_losses).mean()     # 记录并打印每个批次的平均损失

            if opt.step % opt.print_freq < opt.batch_size:
                print("[%d / %d]: batch#%d / steps#%d loss_d= %.3f loss_g= %.3f" %(epoch, opt.epochs, batch, opt.step, d_l, g_l))

        scheduler_G.step()
        scheduler_D.step()

        # 保存每个epoch的loss
        D_Loss.append(d_l)
        G_Loss.append(g_l)

        if (epoch) % opt.model_latest_save_freq == 0:
            print("saving model: " + "model_latest.jpp")
            torch.save({
                "epoch": epoch+1,
                "step": opt.step,
                "D_Loss": D_Loss,
                "G_Loss": G_Loss,
                "G_state_dict": G.state_dict(),
                "D_state_dict": D.state_dict(),
                "optimizerG_state_dict": optimizer_G.state_dict(),
                "optimizerD_state_dict": optimizer_D.state_dict(),
            }, os.path.join(opt.model_path, "model_latest.jpp"))

        # 保存一定epoch下的训练集生成效果
        if (epoch) % opt.train_generated_freq == 0:
            X, _ = next(iter(train_loader))
            with torch.no_grad():
                g = G(X[:, :, :, :opt.image_size])
            g = g.view(opt.batch_size, opt.in_ch, opt.image_size, opt.image_size)
            img_all = train_dataset.denormalize(torch.cat([X, g], dim=3))
            save_image(img_all, opt.train_img_save_path + 'epoch_%d.png' %(epoch), nrow=1)
        # 保存一定epoch下的测试集生成效果
        if (epoch) % opt.test_generated_freq == 0:
            with torch.no_grad():
                g = G(test_imgs[:, :, :, :opt.image_size])
            g = g.view(opt.test_pick, opt.in_ch, opt.image_size, opt.image_size)
            img_all = train_dataset.denormalize(torch.cat([test_imgs, g], dim=3))
            save_image(img_all, opt.test_img_save_path + 'epoch_%d.png' %(epoch), nrow=1)
        # 保存一定epoch下的模型
        if (epoch) % opt.model_save_freq == 0:
            print("saving model: " + "model_epoch_%d.jpp" %(epoch))
            torch.save({
                "epoch": epoch+1,
                "step": opt.step,
                "D_Loss": D_Loss,
                "G_Loss": G_Loss,
                "G_state_dict": G.state_dict(),
                "D_state_dict": D.state_dict(),
                "optimizerG_state_dict": optimizer_G.state_dict(),
                "optimizerD_state_dict": optimizer_D.state_dict(),
            }, os.path.join(opt.model_path, "model_epoch_%d.jpp" %(epoch)))

    print("Done!")
    # 保存训练结果
    print("saving model: " + "model_latest.jpp")
    torch.save({
        "epoch": opt.epochs+1,
        "step": opt.step,
        "D_Loss": D_Loss,
        "G_Loss": G_Loss,
        "G_state_dict": G.state_dict(),
        "D_state_dict": D.state_dict(),
        "optimizerG_state_dict": optimizer_G.state_dict(),
        "optimizerD_state_dict": optimizer_D.state_dict(),
    }, os.path.join(opt.model_path, "model_latest.jpp"))

    """
    G = torch.load("generator.jpp")
    D = torch.load("discriminator.jpp")
    """
    # 画出loss图
    # G的loss包含L1, 相比D loss过大, 因此除以100用来缩放
    plt.plot(Epochs, D_Loss, label="Discriminator Losses")
    plt.plot(Epochs, np.array(G_Loss)/100, label="Generator Losses/100")
    plt.legend()
    plt.savefig(opt.model_path + "loss.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()