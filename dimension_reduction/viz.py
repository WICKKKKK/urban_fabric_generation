#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/20
"""


import torch
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP
from skimage import io
from options import Opt
from ae import AE
from vae import VAE
from inception import Inception
from avae import AVAE
from datasets import MyDataset
from torch.utils.data import DataLoader
import cv2
import os


def main():
    opt = Opt()
    all_dataset = MyDataset(opt, mode="all")
    all_loader = DataLoader(all_dataset, batch_size=opt.batch_size, shuffle=True)

    if opt.en_mode == "ae":
        model = AE(opt).to(opt.device)
    elif opt.en_mode == "vae":
        model = VAE(opt).to(opt.device)
    elif opt.en_mode == "inception":
        model = Inception(model_path="../models/").to(opt.device)
    elif opt.en_mode == "avae":
        model = AVAE(opt).to(opt.device)
    else:
        raise NotImplementedError("|en_mode|: %s is incorrect" % opt.en_mode)

    if opt.en_mode in ["ae", "vae"]:
        checkpoint = torch.load(opt.preload_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("load trained model successfully!")
    elif opt.en_mode == "avae":
        checkpoint = torch.load(opt.preload_model_path)
        model.load_state_dict(checkpoint["model_ae_state_dict"])
        print("load trained model successfully!")

    img_vector = []
    img_all = []
    img_index = []

    for i, data_i in enumerate(all_loader):
        x = all_dataset.denormalize(data_i["image"])
        index = data_i["image_path"]
        if opt.en_mode in ["ae", "avae"]:
            with torch.no_grad():
                x_hat, z = model(x)
            z = z.detach().view(x.size(0), -1)
            # x = x.detach().view(x.size(0), opt.image_size, opt.image_size, -1)
        elif opt.en_mode == "vae":
            with torch.no_grad():
                x_hat, z, mu, logvar, kld = model(x)
            z = mu.detach().view(x.size(0), -1)
            # x = x.detach().view(x.size(0), opt.image_size, opt.image_size, -1)
        elif opt.en_mode == "inception":
            with torch.no_grad():
                z = model(x)
            z = z.detach().view(x.size(0), -1)
        img_vector.append(z.cpu().numpy())
        img_all.append(x.cpu().numpy().transpose(0,2,3,1))
        img_index += index


    img_vector = np.concatenate(img_vector, axis=0)
    img_all = np.concatenate(img_all, axis=0)
    img_index = [index.split("/")[-1].split(".")[0] for index in img_index]

    if opt.viz_mode == "tsne":
        tsne = TSNE(n_components=2, random_state=0)
        two_d_vectors = tsne.fit_transform(img_vector)
    elif opt.viz_mode == "umap":
        umap = UMAP(n_neighbors=25, min_dist=0.00001, metric='correlation')
        two_d_vectors = umap.fit_transform(img_vector)
    else:
        raise NotImplementedError("|viz_mode|: %s is incorrect" % (opt.viz_mode))

    puzzles = np.ones((opt.canvas_size, opt.canvas_size, 3))
    xmin = np.min(two_d_vectors[:, 0])
    xmax = np.max(two_d_vectors[:, 0])
    ymin = np.min(two_d_vectors[:, 1])
    ymax = np.max(two_d_vectors[:, 1])

    for i, vector in enumerate(two_d_vectors):
        x, y = two_d_vectors[i, :]
        x = int((x - xmin) / (xmax - xmin) * (opt.canvas_size - opt.image_size) + opt.image_size//2)
        y = int((y - ymin) / (ymax - ymin) * (opt.canvas_size - opt.image_size) + opt.image_size//2)
        puzzles[y-opt.image_size//2: y+opt.image_size//2, x-opt.image_size//2: x+opt.image_size//2, :] = img_all[i]
        cv2.putText(puzzles, img_index[i], (x, y+opt.image_size//2+30), cv2.FONT_ITALIC, 1, (0, 0, 0), 2, cv2.LINE_AA)
    io.imsave(os.path.join(opt.model_path, '降维可视化_%s.png' %(opt.viz_mode)), puzzles)


if __name__ == '__main__':
    main()




