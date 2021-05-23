#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/5/2
"""

import torch
import os
import numpy as np
from ae import AE
from pix2pix_options import Opt

from torch.utils.data import DataLoader
from dataset import MyDataset

from unet_generator import UNetGenerator
from resnet_generator import ResNetGenerator
from inception import Inception
from torchvision.utils import save_image

from network_setting import get_norm_layer, init_net
from tqdm import tqdm


def normalization(data):
    _range = data.max(axis=0) - data.min(axis=0)
    return (data - data.min(axis=0)) / _range

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist


def main():
    # torch.manual_seed(23)
    # np.random.seed(23)

    opt = Opt()

    if torch.cuda.is_available():
        print(" -- 使用GPU进行训练 -- ")

    # 加载训练数据
    train_dataset = MyDataset(opt.data_root, opt.train_subfolder, device=opt.device)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    # 加载测试数据
    test_dataset = MyDataset(opt.data_root, opt.test_subfolder, device=opt.device)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

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

    ae = AE(opt).to(opt.device)
    ae_checkpoint = torch.load("../models/ae.jpp")                 ## 将训练好的自编码器模型放到这个位置
    ae.load_state_dict(ae_checkpoint["model_state_dict"])


    inception = Inception(model_path="../models/").to(opt.device)

    # 预训练模型加载
    print("loading model from " + opt.model_name)
    checkpoint = torch.load(opt.preload_model_path)
    G.load_state_dict(checkpoint["G_state_dict"])
    print("preload successfully!")


    # 整个数据集上的相似度
    # 计算训练集数据上的得分
    train_image_vector_ae = None
    train_generated_vector_ae = None
    train_image_vector_inception = None
    train_generated_vector_inception = None
    for X in tqdm(train_loader):
        with torch.no_grad():
            image = X[:, :, :, opt.image_size:]
            generated = G(X[:, :, :, :opt.image_size])

            _, image_z_ae = ae(image)
            _, generated_z_ae = ae(generated)

            image_z_inception = inception(image)
            generated_z_inception = inception(generated)
        image_z_ae = image_z_ae.detach().view(X.size(0), -1)
        generated_z_ae = generated_z_ae.detach().view(X.size(0), -1)

        image_z_inception = image_z_inception.detach().view(X.size(0), -1)
        generated_z_inception = generated_z_inception.detach().view(X.size(0), -1)
        if train_image_vector_ae is None:
            train_image_vector_ae = image_z_ae.cpu().numpy()
        else:
            train_image_vector_ae = np.concatenate((train_image_vector_ae, image_z_ae.cpu().numpy()), axis=0)

        if train_generated_vector_ae is None:
            train_generated_vector_ae = generated_z_ae.cpu().numpy()
        else:
            train_generated_vector_ae = np.concatenate((train_generated_vector_ae, generated_z_ae.cpu().numpy()), axis=0)

        if train_image_vector_inception is None:
            train_image_vector_inception = image_z_inception.cpu().numpy()
        else:
            train_image_vector_inception = np.concatenate((train_image_vector_inception, image_z_inception.cpu().numpy()), axis=0)

        if train_generated_vector_inception is None:
            train_generated_vector_inception = generated_z_inception.cpu().numpy()
        else:
            train_generated_vector_inception = np.concatenate((train_generated_vector_inception, generated_z_inception.cpu().numpy()), axis=0)

    # 自编码器 得分
    train_image_vector_ae = train_image_vector_ae.reshape(-1, train_image_vector_ae.shape[-1])
    train_image_vector_norm_ae = normalization(train_image_vector_ae)

    train_generated_vector_ae = train_generated_vector_ae.reshape(-1, train_generated_vector_ae.shape[-1])
    train_generated_vector_norm_ae = normalization(train_generated_vector_ae)

    train_score_L2_norm_ae = np.linalg.norm(train_image_vector_norm_ae - train_generated_vector_norm_ae, axis=1).mean()

    # inception 得分
    train_image_vector_inception = train_image_vector_inception.reshape(-1, train_image_vector_inception.shape[-1])
    train_image_vector_norm_inception = normalization(train_image_vector_inception)

    train_generated_vector_inception = train_generated_vector_inception.reshape(-1, train_generated_vector_inception.shape[-1])
    train_generated_vector_norm_inception = normalization(train_generated_vector_inception)

    train_score_L2_norm_inception = np.linalg.norm(train_image_vector_norm_inception - train_generated_vector_norm_inception, axis=1).mean()


    print("训练集Inception相似度得分：", train_score_L2_norm_inception)
    print("训练集自编码器相似度得分：", train_score_L2_norm_ae)


    # 计算测试集数据上的得分
    test_image_vector_ae = None
    test_generated_vector_ae = None
    test_image_vector_inception = None
    test_generated_vector_inception = None
    for X in tqdm(test_loader):
        with torch.no_grad():
            image = X[:, :, :, opt.image_size:]
            generated = G(X[:, :, :, :opt.image_size])

            _, image_z_ae = ae(image)
            _, generated_z_ae = ae(generated)

            image_z_inception = inception(image)
            generated_z_inception = inception(generated)
        image_z_ae = image_z_ae.detach().view(X.size(0), -1)
        generated_z_ae = generated_z_ae.detach().view(X.size(0), -1)

        image_z_inception = image_z_inception.detach().view(X.size(0), -1)
        generated_z_inception = generated_z_inception.detach().view(X.size(0), -1)
        if test_image_vector_ae is None:
            test_image_vector_ae = image_z_ae.cpu().numpy()
        else:
            test_image_vector_ae = np.concatenate((test_image_vector_ae, image_z_ae.cpu().numpy()), axis=0)

        if test_generated_vector_ae is None:
            test_generated_vector_ae = generated_z_ae.cpu().numpy()
        else:
            test_generated_vector_ae = np.concatenate((test_generated_vector_ae, generated_z_ae.cpu().numpy()),
                                                       axis=0)

        if test_image_vector_inception is None:
            test_image_vector_inception = image_z_inception.cpu().numpy()
        else:
            test_image_vector_inception = np.concatenate(
                (test_image_vector_inception, image_z_inception.cpu().numpy()), axis=0)

        if test_generated_vector_inception is None:
            test_generated_vector_inception = generated_z_inception.cpu().numpy()
        else:
            test_generated_vector_inception = np.concatenate(
                (test_generated_vector_inception, generated_z_inception.cpu().numpy()), axis=0)

    test_image_vector_ae = test_image_vector_ae.reshape(-1, test_image_vector_ae.shape[-1])
    test_image_vector_norm_ae = normalization(test_image_vector_ae)

    test_generated_vector_ae = test_generated_vector_ae.reshape(-1, test_generated_vector_ae.shape[-1])
    test_generated_vector_norm_ae = normalization(test_generated_vector_ae)

    test_score_L2_norm_ae = np.linalg.norm(test_image_vector_norm_ae - test_generated_vector_norm_ae, axis=1).mean()

    test_image_vector_inception = test_image_vector_inception.reshape(-1, test_image_vector_inception.shape[-1])
    test_image_vector_norm_inception = normalization(test_image_vector_inception)

    test_generated_vector_inception = test_generated_vector_inception.reshape(-1, test_generated_vector_inception.shape[-1])
    test_generated_vector_norm_inception = normalization(test_generated_vector_inception)

    test_score_L2_norm_inception = np.linalg.norm(test_image_vector_norm_inception - test_generated_vector_norm_inception, axis=1).mean()


    print("测试集Inception相似度得分：", test_score_L2_norm_inception)
    print("测试集自编码器相似度得分：", test_score_L2_norm_ae)


if __name__ == '__main__':
    main()

