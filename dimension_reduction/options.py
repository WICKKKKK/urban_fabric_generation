#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/19
"""


import os
import torch

class Opt():
    def __init__(self):

        self.device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

        self.data_root = "../data/city_fabric/"
        self.train_subfolder = "train/"
        self.test_subfolder = "test/"

        self.model_path = "../models/ae/"
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.preload = False  # 设置是否进行预加载模型
        self.model_name = "model_latest.jpp"  # 设置预加载模型的名称
        self.preload_model_path = os.path.join(self.model_path, self.model_name)   # 生成预加载模型的路径

        self.batch_size = 32        # 数据批次数
        self.image_size = 256      # 输入图片的大小

        self.en_mode = "ae"        # 特征提取所使用的模型, 请选择 ae | vae | inception | avae 其中一种

        self.z_dim =2048           # 编码器隐空间的向量维度
        self.layers = [3, 64, 128, 256, 512, 512, 512]
        # self.layers = [3, 64, 64, 128, 128, 256]
        self.inter_dim = 500
        self.lamb_kld = 0.05          # kld loss的权重
        # self.lamb_kld = self.z_dim / (self.image_size * self.image_size * 3)

        self.epochs = 200              # 总epochs数量
        self.start_epoch = 1           # 起始epoch
        self.step = 0            # 起始step
        self.lr = 0.0002            # 学习率

        self.model_save_freq = 5       # 每 n 个epoch保存一次模型
        self.model_latest_save_freq = 1  # 每 n 个epoch保存一次模型(覆盖)
        self.print_freq = 100          # 每 n 个step 打印一次信息 (会受到批次的影响而产生不同)

        self.viz_mode = "umap"        #可视化所使用的降维模型, 请选择 umap | tsne 其中一种

        self.canvas_size = 25600