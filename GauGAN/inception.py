#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/20
"""

import torch
from torch import nn
import torchvision
import os
from torchsummary import summary
from google_inception_v3 import inception_v3

class Inception(nn.Module):
    def __init__(self, model_path="./"):
        super(Inception, self).__init__()
        inception = inception_v3(model_path=model_path, pretrained=False, progress=True, init_weights=True)
        if os.path.isfile(model_path):
            inception.load_state_dict(torch.load(model_path))
            print("load pretrained Inception_v3 model parameters successfully!")
        elif os.path.isdir(model_path):
            url = "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"
            model_name = url.split("/")[-1]
            new_model_path = os.path.join(model_path, model_name)
            if os.path.exists(new_model_path):
                inception.load_state_dict(torch.load(new_model_path))
                print("load pretrained Inception_v3 model parameters successfully!")
            else:
                # 如果本地没有训练好的vgg19, 则自动将模型下载到 model_path 目录下
                print("no pretrained Inception_v3 model in '%s', starting download model from %s" % (model_path, url))
                state_dict = torch.utils.model_zoo.load_url(url, model_dir=model_path, map_location=None, progress=True)
                # state_dict = torch.hub.load_state_dict_from_url(url, model_dir=model_path, map_location=None, progress=True)
                inception.load_state_dict(state_dict)
                print("finish downloading and load pretrained Inception_v3 model parameters successfully!")
        else:
            print('Unexpected model_path: {}'.format(model_path))

        inception_features = list(inception.named_children())[:-1]

        self.model = nn.Sequential()
        for x, module in inception_features:
            if x == "AuxLogits" or x == "dropout":
                continue
            self.model.add_module(str(x), module)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)



def main():
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    model = Inception("../models/").to(device)
    summary(model, input_size=(3, 256, 256), batch_size=-1)

if __name__ == '__main__':
    main()