#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/3/30

数据集管理, 预处理

"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from skimage import io
import matplotlib.pyplot as plt


# 数据集管理
class MyDataset(Dataset):
    def __init__(self, root, filename, transform=None, device=torch.device("cpu")):
        """
        自定义数据集初始化
        :param root: 数据文件根目录
        :param filename: 数据文件名
        :param transform: 预处理方法
        """
        super(MyDataset, self).__init__()

        self.path = os.path.join(root, filename)        ## 城市图片数据路径
        self.device = device

        self.imgs, self.image_size = self.img_cut(self.path)

        if transform is not None:
            self.transform = transform                    ## 预处理变换
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # (0, 1) -> (-1, 1)
            ])


    def __len__(self):
        """
        :return: 得到数据集大小
        """
        return len(self.imgs)

    def denormalize(self, x_hat):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1).to(self.device)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1).to(self.device)

        x = x_hat * std + mean

        return x

    def __getitem__(self, item):
        """
        通过索引来获取图片, 以便于迭代
        :param item: 索引
        :return: 索引对应的图片
        """
        image = self.imgs[item]

        image = self.transform(image).to(self.device)

        return image

    def img_cut(self, image_path, tar_size=256, step=128):
        """
        对图像进行切分
        image_path -- 载入图片的路径
        tar_size -- 切分后的图片尺寸
        step -- 步长
        """
        image = io.imread(image_path)
        image = image[:, :, :3]

        cols = ((image.shape[0] - tar_size) // step) + 1
        rows = ((image.shape[1] - tar_size) // step) + 1
        height = (cols-1)*step + tar_size
        width = (rows-1)*step + tar_size
        image_size = (height, width, 3)
        imgs = []
        index = 0
        for i in range(cols):
            for j in range(rows):
                img = image[i * step:(i * step + tar_size), j * step:(j * step + tar_size)]
                imgs.append(img)
                index += 1
        print("切分了 %d 张图片" % (index))
        return imgs, image_size



def main():

    root = "../data/city_xiongan/"        # 根目录
    filename = "rongxi.png"   # 训练数据集目录
    dataset = MyDataset(root=root, filename=filename)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    plt.imshow(dataset.denormalize(next(iter(loader))).cpu().numpy().squeeze().transpose(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    main()