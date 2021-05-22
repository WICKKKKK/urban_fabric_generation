#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/3/31
"""

import torch
import numpy as np
import os, glob
from torch import nn
from torch.utils.data import DataLoader
from dataset import MyDataset
from unet_generator import UNetGenerator
from resnet_generator import ResNetGenerator
from network_setting import init_net, get_norm_layer
from torchvision.utils import save_image

from pix2pix_options import Opt




def main():
    # torch.manual_seed(23)
    # np.random.seed(23)

    opt = Opt()

    if torch.cuda.is_available():
        print(" -- 使用GPU进行训练 -- ")

    if not os.path.exists(opt.test_img_infer_path):
        os.mkdir(opt.test_img_infer_path)

    # 加载测试集数据
    test_dataset = MyDataset(opt.data_root, opt.test_subfolder,device=opt.device)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    test_size = len(test_dataset)

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

    # 加载模型
    checkpoint = torch.load(opt.infer_model_path)
    G.load_state_dict(checkpoint["G_state_dict"])

    step = 0

    for X in test_loader:
        with torch.no_grad():
            g = G(X[:, :, :, :opt.image_size].to(opt.device))
        g = g.view(-1, opt.in_ch, opt.image_size, opt.image_size)
        img_all = test_dataset.denormalize(torch.cat([X.to(opt.device), g], dim=3))
        step += 1
        save_image(img_all, opt.test_img_infer_path + '%d.png' % step, pad_value=255, nrow=1)
        print("image: %d / %d" % (step, test_size))
    print("Done!")

if __name__ == '__main__':
    main()