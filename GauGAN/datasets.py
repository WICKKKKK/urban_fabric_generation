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
import matplotlib.pyplot as plt
from skimage.segmentation import random_walker
import numpy as np
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
        elif mode == "style":
            self.path = os.path.join(opt.data_root, opt.style_subfolder)
            self.image_list = [x for x in sorted(os.listdir(self.path))]
        else:
            raise ValueError("|mode| is invalid")

        self.device = opt.device
        self.image_size = opt.image_size
        self.label_nc = opt.label_nc
        self.one_hot = opt.one_hot

        if self.one_hot:
            self.labels = opt.labels
            self.labels_index = opt.labels_index


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

    def denoise(self, image):
        """
        ???????????????????????????, ?????????????????????????????????????????????label?????????
        ????????????????????????[image_size, image_size]???????????????, ????????????1-10???(??????10???)??????
        """
        markers = np.zeros([self.image_size, self.image_size], dtype=np.uint8)
        interval = 30
        for key, value in self.labels.items():
            markers[(image[:, :, 0] >= (value[0] - interval)) & (image[:, :, 0] <= (value[0] + interval))
                    & (image[:, :, 1] >= (value[1] - interval)) & (image[:, :, 1] <= (value[1] + interval))
                    & (image[:, :, 2] >= (value[2] - interval)) & (image[:, :, 2] <= (value[2] + interval))] = \
                self.labels_index[key]
        cleaned_markers = random_walker(image, markers, beta=130, mode="bf", multichannel=True)
        out_kind = sorted(set(cleaned_markers.ravel()))
        kind = sorted(set(markers.ravel()))
        if 0 in kind:
            kind.remove(0)
        new_markers = cleaned_markers.copy()
        for i, value in enumerate(kind):
            new_markers[cleaned_markers[:, :]==out_kind[i]] = value
        # for i in range(1, len(out_kind) + 1):
        #     new_markers[cleaned_markers[:, :] == i] = kind[i - 1]
        new_markers = new_markers - 1             # ??????random walker??????label???????????????0, ?????????????????????????????????
        return np.array(new_markers, dtype=np.uint8)

    def marker2img(self, markers):
        img = np.zeros([markers.shape[0], markers.shape[1], 3], dtype=np.uint8)
        for key, value in self.labels_index.items():
            img[markers[:, :] == (value-1)] = self.labels[key]
        return img

    def label_preprocess(self, label_tensor):
        """
        ??????????????????????????????, ????????????pix2pix???????????????(paired)??????, gauGAN????????????????????????label image?????????one-hot??????,
        ???????????????????????????label, ??????label image????????????????????????, ?????????????????????????????????(label?????????)????????????label????????????,
        ??????????????????????????????, ?????????label???????????????256, ?????????????????????????????????, ???????????????label??????????????????????????????.
        ?????????????????????????????????one-hot??????, ????????????SPADE???. ??????????????????????????????????????????????????????????????????.
        ????????????????????????one-hot??????, ??????????????????????????????????????????????????????(label image ??? real image)???????????????,
        ???????????????????????????????????????(?????????????????????).
        data: ??????????????????????????????
        source code????????????instance map, ???????????????

        ???????????????label_tensor?????????one-hot????????????, ???[5] => [0,0,0,0,0,1,0...] (????????????label_nc)
        """
        label_tensor = label_tensor.long()
        _, h, w = label_tensor.size()
        nc = self.label_nc
        tensor = torch.FloatTensor(nc, h, w).zero_()
        semantic_tensor = tensor.scatter_(dim=0, index=label_tensor, value=1.0)
        return semantic_tensor

    def __getitem__(self, item):
        """
        ?????????????????????data:
        image:         dataset??????real image(??????normalize???tensor, [-1,1])
        image_path:    ???????????????????????????
        label:         ??????????????????label tensor(one-hot???????????????normalize??????????????? tensor, [-1, 1])
        label_image:   ????????????????????????, ?????????tensor??????, ??????????????????normalize, ???????????????torchvision.utils.save_image?????????
        """
        image_path = os.path.join(self.path, self.image_list[item])
        image = io.imread(image_path)
        input_dict = {}

        real_image = image[:, self.image_size:, :]
        label_image = image[:, :self.image_size, :]

        transform_image = self.get_transform(normalize=True)

        if self.one_hot:
            # real_image = self.denoise(real_image)
            # cleaned_real_image = self.marker2img(real_image)
            cleaned_real_image_tensor = transform_image(real_image).to(self.device)

            input_dict["image"] = cleaned_real_image_tensor.to(self.device)
            input_dict["image_path"] = image_path

            label_image = self.denoise(label_image)

            cleaned_label_image = self.marker2img(label_image)
            cleaned_label_image_tensor = transform_image(cleaned_label_image).to(self.device)

            label_image = label_image.reshape((1, label_image.shape[0], label_image.shape[1]))
            # ??????????????????label image????????????transform?????????, ????????????normalize, ???????????????[nc, h, w]?????????intTensor??????
            label_tensor = torch.from_numpy(label_image)
            semantic_tensor = self.label_preprocess(label_tensor).to(self.device)

            input_dict["label"] = semantic_tensor.to(self.device)
            input_dict["label_image"] = cleaned_label_image_tensor.to(self.device)
        else:
            real_image_tensor = transform_image(real_image).to(self.device)
            input_dict["image"] = real_image_tensor.to(self.device)
            input_dict["image_path"] = image_path

            label_tensor = transform_image(label_image).to(self.device)
            input_dict["label"] = label_tensor.to(self.device)
            input_dict["label_image"] = label_tensor.to(self.device)

        return input_dict

def main():
    from options import BaseOptions
    opt = BaseOptions()
    opt.initialize()
    db = MyDataset(opt, mode="test")
    from torch.utils.data import DataLoader
    loader = DataLoader(db, batch_size=8, shuffle=False)
    data = next(iter(loader))

    print(data["label"].shape)




if __name__ == '__main__':
    main()