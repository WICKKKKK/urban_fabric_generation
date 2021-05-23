#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/5
"""


import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from skimage import io
from torchvision.utils import save_image


class MyDataset(Dataset):
    def __init__(self, opt, mode="train"):
        super(MyDataset, self).__init__()

        if mode == "train":
            self.path = os.path.join(opt.data_root, opt.train_subfolder)
            self.image_list = [x for x in sorted(os.listdir(self.path))]
        elif mode == "test":
            self.path = os.path.join(opt.data_root, opt.test_subfolder)
            self.image_list = [x for x in sorted(os.listdir(self.path))]
        elif mode == "all":
            self.path = os.path.join(opt.data_root, opt.all_subfolder)
            self.image_list = [x for x in sorted(os.listdir(self.path))]
        else:
            raise ValueError("|mode| is invalid")

        self.device = opt.device
        self.image_size = opt.image_size


    def __len__(self):
        return len(self.image_list)

    def get_transform(self, normalize=True):
        transform_list = []
        transform_list += [transforms.ToTensor()]
        if normalize:
            transform_list += [transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                    std=(0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

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
        得到四种类型的data:
        image:         dataset中的real image(经过normalize的tensor, [-1,1])
        image_path:    该对图片的存储路径
        label:         经过处理后的label tensor(one-hot或者是经过normalize后的三通道 tensor, [-1, 1])
        label_image:   标签所对应的图像, 仍然是tensor形式, 但是没有经过normalize, 可以直接桶torchvision.utils.save_image来保存
        """
        image_path = os.path.join(self.path, self.image_list[item])
        image = io.imread(image_path)
        input_dict = {}

        real_image = image[:, self.image_size:, :]
        label_image = image[:, :self.image_size, :]

        transform_image = self.get_transform(normalize=True)

        real_image_tensor = transform_image(real_image).to(self.device)
        input_dict["image"] = real_image_tensor.to(self.device)
        input_dict["image_path"] = image_path

        label_tensor = transform_image(label_image).to(self.device)
        input_dict["label"] = label_tensor.to(self.device)
        input_dict["label_image"] = label_tensor.to(self.device)

        return input_dict

def main():
    from options import Opt
    opt = Opt()
    db = MyDataset(opt, mode="test")
    from torch.utils.data import DataLoader
    loader = DataLoader(db, batch_size=8, shuffle=False)
    data = next(iter(loader))

    print(data["label"].shape)





if __name__ == '__main__':
    main()