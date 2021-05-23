#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/3/30

数据集管理, 预处理

"""

import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# 数据集管理
class MyDataset(Dataset):
    def __init__(self, root, subfolder, transform=None, device=torch.device("cpu")):
        """
        自定义数据集初始化
        :param root: 数据文件根目录
        :param subfolder: 数据文件子目录
        :param transform: 预处理方法
        """
        super(MyDataset, self).__init__()

        self.path = os.path.join(root, subfolder)        ## 图片数据集路径
        self.image_list = [x for x in sorted(os.listdir(self.path))]    ## 图片文件名
        self.device = device

        if transform:
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
        return len(self.image_list)

    def denormalize(self, x_hat):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1).to(self.device)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1).to(self.device)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

    def __getitem__(self, item):
        """
        通过索引来获取图片, 以便于迭代
        :param item: 索引
        :return: 索引对应的图片
        """
        image_path = os.path.join(self.path, self.image_list[item])
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:,:,[2,1,0]] # BGR => RGB

        image = self.transform(image).to(self.device)

        return image


def main():
    root = "../data/city_fabric/"
    subfolder = "test/"

    dataset = MyDataset(root=root, subfolder=subfolder)
    print(dataset[0].shape)



if __name__ == '__main__':
    main()