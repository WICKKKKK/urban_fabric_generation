#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/2
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from options import TrainOptions
from datasets import MyDataset
from gaugan_model import GauganModel, GauganTrainer

# torch.manual_seed(23)
# np.random.seed(23)

if torch.cuda.is_available():
    print(" -- 使用GPU进行训练 -- ")

opt = TrainOptions()
opt.initialize()

opt.apply_spectral = True
opt.ganFeat_loss = True
opt.vgg_loss = True
opt.fnc_g = 32
opt.fnc_d = 64
opt.fnc_e = 64
opt.G_per_steps = 4
opt.lamb_feat = 10.0  # feature matching loss 的权重
opt.lamb_vgg = 10.0  # vgg loss 的权重
opt.print_freq = 10          # 每 n 个step 打印一次信息 (会受到批次的影响而产生不同)

# 加载训练数据
train_dataset = MyDataset(opt, mode="train")
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
# 加载测试数据
test_dataset = MyDataset(opt, mode="test")
test_loader = DataLoader(test_dataset, batch_size=opt.test_pick, shuffle=True)
print(len(train_dataset), len(test_dataset))

trainer = GauganTrainer(opt)

start_epoch = 1
step = 0
losses = {}
losses["D_fake"] = []
losses["D_real"] = []
losses["GAN"] = []
if opt.use_vae:
    losses["KLD"] = []
if opt.ganFeat_loss:
    losses["GAN_Feat"] = []
else:
    losses["L1"] = []
if opt.vgg_loss:
    losses["VGG"] = []
# 预训练模型加载
if opt.continue_train:
    print("Resuming step&losses information from " + opt.model_name)
    checkpoint = torch.load(opt.preload_model_path)
    start_epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    losses = checkpoint["losses"]

# 还未训练时的输出数据
data = next(iter(train_loader))
if not opt.continue_train:
    trainer.save_imgs(data, epoch=0, path=opt.train_img_save_path)
# 随机选择测试集中的n(8)张来进行生成测试
test_data = next(iter(test_loader))

for epoch in range(start_epoch, opt.epochs+1):
    epoch_losses, epoch_mean_losses = {}, {}  # 每个epoch的数据记录容器
    batch = 0
    for i, data_i in enumerate(train_loader):
        batch += 1
        step += data_i["image"].size(0)

        trainer.run_discriminator_one_step(data_i)

        for i in range(opt.G_per_steps):
            trainer.run_generate_one_step(data_i)
        # if i % opt.D_steps_per_G == 0:
        #     trainer.run_generate_one_step(data_i)

        epoch_losses = trainer.get_latest_losses()

        for key, value in epoch_losses.items():
            epoch_mean_losses[key] = np.array(value).mean()

        if step%opt.print_freq < opt.batch_size:
            trainer.print_info(epoch=epoch, batch=batch, step=step, opt=opt, losses=epoch_mean_losses)

        # if step%opt.model_save_latest_freq < opt.batch_size:
        #     trainer.save_latest(epoch=epoch, step=step, losses=losses)

        # torch.cuda.empty_cache()

    trainer.update_learning_rate(epoch)

    for key, value in epoch_mean_losses.items():
        losses[key].append(value)

    if (epoch) % opt.model_save_latest_freq == 0:
        trainer.save_latest(epoch=epoch, step=step, losses=losses)
    if (epoch) % opt.train_generated_freq == 0:
        data = next(iter(train_loader))
        trainer.save_imgs(data, epoch=epoch, path=opt.train_img_save_path)
    if (epoch) % opt.test_generated_freq == 0:
        trainer.save_imgs(test_data, epoch=epoch, path=opt.test_img_save_path)
    if (epoch) % opt.model_save_epoch_freq == 0:
        trainer.save_epoch(epoch=epoch, step=step, losses=losses)


print('Training was successfully finished.')

# 画出loss图
# 数值介绍:
# loss_D_fake:      判别器对生成器生成的fake image的损失值, 因为用了hinge loss, 因此判别器输出的结果小于-1才会等于0, 否则loss会线性增长, loss在(0, +∞)之间浮动, 由于初始化影响, 一般情况会在(0, 2)浮动
# loss_D_real:      判别器对real image的损失值, 同样因为用了hinge loss, 因此判别器输出的结果大于1才会等于0, 否则loss会线性增长, loss在(0, +∞)之间浮动, 由于初始化, 一般情况会在(0, 2)浮动
# loss_GAN:         生成器训练时对fake image的损失值, 同样因为用了hinge loss, 因此判别器输出的结果越大越好, 否则loss会线性增长, loss在(-∞, +∞)之间浮动, 由于初始化, 一般情况会在(-1, 1)浮动
# loss_KLD:         KL Divergence的损失值, encoder输出的隐空间向量分布与 N(0,1)分布的差异, 这里用了reparameter trick,具体可以查看vae原paper
# loss_GAN_Feat:    生成器生成的fake image与real image在判别器各层输出的feature值之间的L1 loss, 如果将选项中的gan_feature_loss关闭, 则计算原pix2pix中的生成图与真实图之间的L1 loss
# loss_VGG:         算是GAN_feature_loss的补强, 因为一开始判别器提取特征的能力并不强, 用预训练的VGG网络可以补充训练开始阶段的生成损失.
Epochs = range(1, opt.epochs + 1)
for key, value in losses.items():
    if key == "VGG" or key == "GAN_Feat":
        value /= 10.0
    plt.plot(Epochs, value, label=key)
plt.legend()
plt.savefig(opt.model_path + "loss.png", dpi=300)
plt.show()