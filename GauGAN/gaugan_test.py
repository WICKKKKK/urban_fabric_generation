#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/10

推荐在Jupyter notebook中逐块运行
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os, glob
import matplotlib.pyplot as plt

from options import TestOptions
from datasets import MyDataset
from gaugan_model import GauganModel
from tqdm import tqdm


if torch.cuda.is_available():
    print(" -- 使用GPU进行训练 -- ")

opt = TestOptions()
opt.initialize()
opt.is_train = False


# 加载测试数据
train_dataset = MyDataset(opt, mode="train")
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
test_dataset = MyDataset(opt, mode="test")
test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
test_size = len(test_loader)

model = GauganModel(opt)

output_path = os.path.join(opt.test_img_infer_path, "style_by_truth/")
if not os.path.exists(output_path):
    os.mkdir(output_path)


# 根据ground truth图来进行样式生成
opt.infer_byz = False
for i, data_i in tqdm(enumerate(test_loader)):
    with torch.no_grad():
        g = model(data_i, mode="inference")
    g = test_dataset.denormalize(g.view(-1, opt.in_nc, opt.image_size, opt.image_size))
    r = test_dataset.denormalize(data_i["image"])
    label = test_dataset.denormalize(data_i["label_image"])
    img_all = torch.cat([label, r, g], dim=3)
    save_image(img_all, output_path + '%d.png' %(i), nrow=1, pad_value=255)
    print("image: %d / %d" % (i, test_size))
print("Done!")


# 根据不同z向量来进行生成得到多个方案
image_vector = []
for i, data_i in tqdm(enumerate(train_loader)):
    with torch.no_grad():
      mu, logvar = model(data_i, mode="encode_only")
    image_vector.append(mu)

image_vector = torch.cat(image_vector, dim=0)
print(image_vector.shape)
vector_max = torch.max(image_vector, dim=0).values
vector_min = torch.min(image_vector, dim=0).values
print(vector_max.shape)
print(vector_min.shape)

num = 10
opt.infer_byz = True
output_path = os.path.join(opt.test_img_infer_path, "multimode_500/")
if not os.path.exists(output_path):
    os.mkdir(output_path)
for i, data_i in enumerate(test_loader):
    img_list = []
    for j in range(num):
        seed=j
        torch.manual_seed(seed)
        np.random.seed(seed)
        z = (torch.randn(data_i["label"].size(0), opt.z_dim, dtype=torch.float32, device=opt.device)*(vector_max-vector_min) + vector_min) * 500.0
        with torch.no_grad():
            g = model(data_i, mode="inference", z=z)
        g = test_dataset.denormalize(g.view(-1, opt.in_nc, opt.image_size, opt.image_size))
        r = test_dataset.denormalize(data_i["image"])
        label = test_dataset.denormalize(data_i["label_image"])
        img_list.append(torch.cat([label, r, g], dim=3))
    img_all = torch.cat(img_list, dim=3)
    save_image(img_all, output_path + 'test_%d.png' %(i), nrow=1, pad_value=255)
    print("image: %d / %d" % (i, test_size))
print("Done!")


# 聚类得到n类所指代的最中心的n张图的path, 然后手动将这n张图放入style文件夹中
zs = []
image_zs = []
paths = []
encode_num = 10   ## 为了更好描述编码得到的形状，在加入mu的基础上，从分布中采样n次，以此得到更好的聚类效果
for i, data_i in tqdm(enumerate(train_loader)):
    with torch.no_grad():
        mu, logvar = model(data_i, mode="encode_only")
    for i in range(encode_num):
        zs.append(model.reparameterize(mu, logvar).cpu().numpy())
    zs.append(mu.cpu().numpy())
    image_zs.append(mu.cpu().numpy())
    paths.append(data_i["image_path"])
zs = np.concatenate(zs, axis=0)
print(zs.shape)

from sklearn.cluster import KMeans
center_num = 10
kmeans = KMeans(n_clusters=center_num, init="k-means++", max_iter=300, n_init=10, random_state=0)

kmeans.fit(zs)
print(kmeans.cluster_centers_.shape)


for i in range(center_num):
    center = kmeans.cluster_centers_[i]
    nearest_path = None
    min_d = 10000000000000
    for j, vectors in enumerate(image_zs):
        for k, z in enumerate(vectors):
          # print(z.shape)
          distance = np.linalg.norm(z-center)
          if distance < min_d:
              min_d = distance
              nearest_path = paths[j][k]
    print(nearest_path)



## 根据不同样式图来进行生成
opt.infer_byz = True
style_dataset = MyDataset(opt, mode="style")
style_loader = DataLoader(style_dataset, batch_size=1, shuffle=False)
output_path = os.path.join(opt.test_img_infer_path, "multistyle_5000/")
if not os.path.exists(output_path):
    os.mkdir(output_path)
for i, data_i in enumerate(test_loader):
    img_all_list = []
    for j, data_j in enumerate(style_loader):
        style_image = data_j["image"]
        with torch.no_grad():
            # g = model(data_i, mode="inference", style_image=style_image)
            z, _, _ = model.encode_z(style_image)
            g = model(data_i, mode="inference", z=z*5000.)
        g = test_dataset.denormalize(g.view(-1, opt.in_nc, opt.image_size, opt.image_size))
        r = test_dataset.denormalize(data_i["image"])
        label = test_dataset.denormalize(data_i["label_image"])
        style = test_dataset.denormalize(style_image)
        style = style.expand(data_i["image"].shape[0], 3, 256, 256)
        img_this_style = torch.cat([label, r, g, style], dim=3)
        img_all_list.append(img_this_style)
    img_all = torch.cat(img_all_list, dim=3)

    save_image(img_all, output_path + 'test_%d.png' % (i), nrow=1, pad_value=255)
    print("image: %d / %d" % (i, test_size))
print("Done!")