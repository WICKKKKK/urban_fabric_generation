#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/5/3
"""

import torch
import os
import numpy as np
from skimage import io
from torch.utils.data import DataLoader
from dataset_city import MyDataset
from pix2pix_options import Opt
from unet_generator import UNetGenerator
from resnet_generator import ResNetGenerator
from network_setting import get_norm_layer, init_net
from tqdm import tqdm


def main():

    if torch.cuda.is_available():
        print("-- 使用GPU训练 --")

    opt = Opt()

    root = "../data/city_xiongan/"
    filename = "rongxi.png"
    dataset = MyDataset(root=root, filename=filename, device=opt.device)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)


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

    print("loading model from " + opt.model_name)
    checkpoint = torch.load(opt.preload_model_path)
    G.load_state_dict(checkpoint["G_state_dict"])
    print("preload successfully!")

    generated_imgs = []

    for i, data_i in tqdm(enumerate(dataloader)):
        g = G(data_i)
        g = dataset.denormalize(g.detach())
        generated_imgs.append(g.cpu().numpy())

    img_all = np.concatenate(generated_imgs, axis=0)
    img_all = img_all.transpose(0, 2, 3, 1)

    image_size = dataset.image_size
    cols = ((image_size[0] - 256) // 128) + 1
    rows = ((image_size[1] - 256) // 128) + 1
    image_generated = np.zeros(image_size)

    index = 0
    for i in range(cols):
        for j in range(rows):
            image_generated[i*128 : (i * 128 + 256), j*128:(j * 128 + 256)] = img_all[index]
            index += 1

    io.imsave(os.path.join(opt.model_path, "./city_generated.png"), image_generated)


if __name__ == '__main__':
    main()
