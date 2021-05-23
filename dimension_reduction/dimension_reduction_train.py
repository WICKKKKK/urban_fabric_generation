#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/19
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from options import Opt
from datasets import MyDataset
from torch.utils.data import DataLoader
from ae import AE
from vae import VAE
from avae import AVAE, Discriminatr_z

def main():

    if torch.cuda.is_available():
        print(" -- 使用GPU进行训练 -- ")

    opt = Opt()
    train_dataset = MyDataset(opt, mode="train")
    test_dataset = MyDataset(opt, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

    if opt.en_mode == "ae":
        model = AE(opt).to(opt.device)
        mse = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.en_mode == "vae":
        model = VAE(opt).to(opt.device)
        mse = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.en_mode == "inception":
        raise NotImplementedError("|en_mode|: %s do not need train" % opt.en_mode)
    elif opt.en_mode == "avae":
        model_ae = AVAE(opt).to(opt.device)
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()
        optimizer_ae = optim.Adam(model_ae.parameters(), lr=opt.lr)
        model_dz = Discriminatr_z(opt).to(opt.device)
        optimizer_dz = optim.Adam(model_dz.parameters(), lr=opt.lr)
    else:
        raise NotImplementedError("|en_mode|: %s is incorrect" % opt.en_mode)



    step = 0
    start_epoch = 1

    Loss = {}

    if opt.en_mode == "ae":
        Loss["train"] = []
        Loss["test"] = []
    elif opt.en_mode == "vae":
        Loss["train"] = []
        Loss["test"] = []
        Loss["kld"] = []
    elif opt.en_mode == "avae":
        Loss["train"] = []
        Loss["test"] = []
        Loss["d_z"] = []
        Loss["encoder"] = []

    if opt.preload:
        if opt.en_mode == "ae" or opt.en_mode == "vae":
            print("loading model from " + opt.model_name)
            checkpoint = torch.load(opt.preload_model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict((checkpoint["optimizer_state_dict"]))
            start_epoch = checkpoint["epoch"]
            step = checkpoint["step"]
            Loss = checkpoint["Loss"]
            print("preload successfully!")
        elif opt.en_mode == "avae":
            print("loading model from " + opt.model_name)
            checkpoint = torch.load(opt.preload_model_path)
            model_ae.load_state_dict(checkpoint["model_ae_state_dict"])
            optimizer_ae.load_state_dict((checkpoint["optimizer_ae_state_dict"]))
            model_dz.load_state_dict(checkpoint["model_dz_state_dict"])
            optimizer_dz.load_state_dict((checkpoint["optimizer_dz_state_dict"]))
            start_epoch = checkpoint["epoch"]
            step = checkpoint["step"]
            Loss = checkpoint["Loss"]
            print("preload successfully!")

    torch.save({
        "epoch": start_epoch + 1,
        "step": step,
        "Loss": Loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(opt.model_path, "model_latest.jpp"))

    for epoch in range(start_epoch, opt.epochs):
        losses, l, batch = {}, {}, 0
        if opt.en_mode == "ae":
            losses["train"] = []
            losses["test"] = []
            l["train"] = 0.0
            l["test"] = 0.0
        elif opt.en_mode == "vae":
            losses["train"] = []
            losses["kld"] = []
            losses["test"] = []
            l["train"] = 0.0
            l["test"] = 0.0
            l["kld"] = 0.0
        elif opt.en_mode == "avae":
            losses["train"] = []
            losses["test"] = []
            l["train"] = 0.0
            l["test"] = 0.0
            losses["d_z"] = []
            l["d_z"] = []
            losses["encoder"] = []
            l["encoder"] = []

        for i, data_i in enumerate(train_loader):
            batch += 1
            x = data_i["image"]
            step += x.size(0)
            if opt.en_mode == "ae":
                x_hat, z = model(x)
                loss = mse(x_hat, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses["train"].append(loss.item())
                l["train"] = np.array(losses["train"]).mean()
            elif opt.en_mode == "vae":
                x_hat, z, mu, logvar, kld = model(x)
                loss = mse(x_hat, x) + kld * opt.lamb_kld

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses["train"].append(loss.item()-kld.item()*opt.lamb_kld)
                losses["kld"].append(kld.item())
                l["train"] = np.array(losses["train"]).mean()
                l["kld"] = np.array(losses["kld"]).mean()
            elif opt.en_mode == "avae":
                x_hat, z = model_ae(x)
                # 先训练 Discriminator_z
                for param in model_dz.parameters():
                    param.requires_grad = True
                model_dz.zero_grad()  # 梯度初始化为0
                z_real_dist = torch.randn(x.size(0), opt.z_dim, device=opt.device) * 5.0
                d_real = model_dz(z_real_dist)
                d_fake = model_dz(z.detach())
                dz_loss_real = bce(d_real, torch.ones(d_real.size()).to(opt.device))
                dz_loss_fake = bce(d_fake, torch.zeros(d_fake.size()).to(opt.device))
                dz_loss = (dz_loss_fake + dz_loss_real) * 0.5
                dz_loss.backward()
                optimizer_dz.step()
                losses["d_z"].append(dz_loss.item())
                l["d_z"] = np.array(losses["d_z"]).mean()

                # 再训练 encoder
                x_hat, z = model_ae(x)
                for param in model_dz.parameters():
                    param.requires_grad = False
                model_ae.zero_grad()  # 梯度初始化为0
                d_fake = model_dz(z)
                e_loss = bce(d_fake, torch.ones(d_fake.size()).to(opt.device))
                e_loss.backward()
                optimizer_ae.step()
                losses["encoder"].append(e_loss.item())
                l["encoder"] = np.array(losses["encoder"]).mean()

                # 然后训练 decoder
                x_hat, z = model_ae(x)
                loss = mse(x_hat, x)
                optimizer_ae.zero_grad()
                loss.backward()
                optimizer_ae.step()

                losses["train"].append(loss.item())
                l["train"] = np.array(losses["train"]).mean()

            if step%opt.print_freq < opt.batch_size:
                for i, data_i in enumerate(test_loader):
                    x = data_i["image"]
                    with torch.no_grad():
                        if opt.en_mode == "ae" or opt.en_mode == "vae":
                            x_hat = model(x)[0]
                        elif opt.en_mode == "avae":
                            x_hat = model_ae(x)[0]
                    loss = mse(x_hat.detach(), x)
                    losses["test"].append(loss.item())
                l["test"] = np.array(losses["test"]).mean()

                informations = ("[%d / %d]: batch#%d / steps#%d:\n " % (epoch, opt.epochs, batch, step))
                for key, value in l.items():
                    informations += (" loss_%s= %.3f" % (key, value))
                print(informations)

        for key, value in l.items():
            Loss[key].append(value)

        if (epoch) % opt.model_latest_save_freq == 0:
            print("saving model: " + "model_latest.jpp")
            if opt.en_mode == "ae" or opt.en_mode == "vae":
                torch.save({
                    "epoch": epoch+1,
                    "step": step,
                    "Loss": Loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(opt.model_path, "model_latest.jpp"))
            elif opt.en_mode == "avae":
                torch.save({
                    "epoch": epoch + 1,
                    "step": step,
                    "Loss": Loss,
                    "model_ae_state_dict": model_ae.state_dict(),
                    "optimizer_ae_state_dict": optimizer_ae.state_dict(),
                    "model_dz_state_dict": model_dz.state_dict(),
                    "optimizer_dz_state_dict": optimizer_dz.state_dict(),
                }, os.path.join(opt.model_path, "model_latest.jpp"))

        if (epoch) % opt.model_save_freq == 0:
            print("saving model: " + "model_epoch_%d.jpp" %(epoch))
            if opt.en_mode == "ae" or opt.en_mode == "vae":
                torch.save({
                    "epoch": epoch+1,
                    "step": step,
                    "Loss": Loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(opt.model_path, "model_epoch_%d.jpp" %(epoch)))
            elif opt.en_mode == "avae":
                torch.save({
                    "epoch": epoch + 1,
                    "step": step,
                    "Loss": Loss,
                    "model_ae_state_dict": model_ae.state_dict(),
                    "optimizer_ae_state_dict": optimizer_ae.state_dict(),
                    "model_dz_state_dict": model_dz.state_dict(),
                    "optimizer_dz_state_dict": optimizer_dz.state_dict(),
                }, os.path.join(opt.model_path, "model_epoch_%d.jpp" %(epoch)))

    print("Done!")
    print("saving model: " + "model_latest.jpp")
    if opt.en_mode == "ae" or opt.en_mode == "vae":
        torch.save({
            "epoch": opt.epochs+1,
            "step": step,
            "Loss": Loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, os.path.join(opt.model_path, "model_latest.jpp"))
    elif opt.en_mode == "avae":
        torch.save({
            "epoch": opt.epochs+1,
            "step": step,
            "Loss": Loss,
            "model_ae_state_dict": model_ae.state_dict(),
            "optimizer_ae_state_dict": optimizer_ae.state_dict(),
            "model_dz_state_dict": model_dz.state_dict(),
            "optimizer_dz_state_dict": optimizer_dz.state_dict(),
        }, os.path.join(opt.model_path, "model_latest.jpp"))

    # 画出loss图
    # G的loss包含L1, 相比D loss过大, 因此除以100用来缩放
    Epochs = range(1, opt.epochs)
    for key, value in Loss.items():
        plt.plot(Epochs, value, label=key)
    plt.legend()
    plt.savefig(os.path.join(opt.model_path, "loss.png"), dpi=300)
    plt.show()



if __name__ == '__main__':
    main()