#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/5

放整合的 gaugan 模型, 以及训练的时候会用到的一些 optimizer 模块
"""

import torch
from torch import nn
import os
from torchvision.utils import save_image

from network_utils import GANLoss
from SPADE_generator import SPADEGenerator
from pix2pixHD_generator import Pix2pixHDGenerator
from discriminator import MultiscaleDiscriminator, NLayerDiscriminator
from encoder import ConvEncoder
from network_utils import VGGLoss, KLDLoss


class GauganModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if opt.use_gpu \
            else torch.FloatTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # 设置损失函数
        if opt.is_train:
            self.criterionGAN = GANLoss(opt.gan_loss_mode, tensor=self.FloatTensor)
            self.criterionFeat = torch.nn.L1Loss()
            if opt.vgg_loss:
                self.criterionVGG = VGGLoss(opt)
            if opt.use_vae:
                self.KLDLoss = KLDLoss()

    def forward(self, data, mode, z=None, style_image=None):
        """
        使用 mode 来区分开不同计算 loss function
        """
        input_semantics, real_image = data["label"], data["image"]

        if mode == "generator":
            g_loss, generated = self.compute_generator_loss(input_semantics, real_image)
            return g_loss, generated
        elif mode == "discriminator":
            d_loss = self.compute_discriminator_loss(input_semantics, real_image)
            return d_loss
        elif mode == "encode_only":
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == "inference":
            with torch.no_grad():
                if self.opt.infer_byz:
                    fake_image = self.generate_fake_byz(input_semantics, z)
                else:
                    if style_image is None:
                        fake_image, _ = self.generate_fake_byimg(input_semantics, real_image)
                    else:
                        fake_image, _ = self.generate_fake_byimg(input_semantics, style_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")


    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.is_train:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.TTUR:
            G_lr, D_lr = opt.lr/2, opt.lr*2
        else:
            G_lr, D_lr = opt.lr, opt.lr

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        if opt.continue_train:
            print("loading optimizers from " + opt.model_name)
            checkpoint = torch.load(opt.preload_model_path)
            optimizer_G.load_state_dict(checkpoint["optimizerG_state_dict"])
            optimizer_D.load_state_dict(checkpoint["optimizerD_state_dict"])

        return optimizer_G, optimizer_D


    def initialize_networks(self, opt):
        if opt.netG == "spade":
            netG = SPADEGenerator(opt).to(opt.device)
        elif opt.netG == "pix2pixHD":
            netG = Pix2pixHDGenerator(opt).to(opt.device)
        else:
            raise NotImplementedError('generator method [%s] is not implemented' % opt.netG)
        netG.print_network()
        netG.init_weights(opt.init_type, opt.init_gain)

        if opt.is_train:
            if opt.netD == "multiscale":
                netD = MultiscaleDiscriminator(opt).to(opt.device)
            elif opt.netD == "patch":
                netD = NLayerDiscriminator(opt).to(opt.device)
            else:
                raise NotImplementedError('discriminator method [%s] is not implemented' % opt.netD)
            netD.print_network()
            netD.init_weights(opt.init_type, opt.init_gain)
        else:
            netD = None

        if opt.use_vae:
            netE = ConvEncoder(opt).to(opt.device)
            netE.print_network()
            netE.init_weights(opt.init_type, opt.init_gain)
        else:
            netE = None

        if not opt.is_train or opt.continue_train:
            print("loading network parameters from " + opt.model_name)
            checkpoint = torch.load(opt.preload_model_path)
            netG.load_state_dict(checkpoint["netG_state_dict"])
            if opt.is_train:
                netD.load_state_dict(checkpoint["netD_state_dict"])
            if opt.use_vae:
                netE.load_state_dict(checkpoint["netE_state_dict"])

        return netG, netD, netE


    def compute_generator_loss(self, input_semantics, real_image):
        """
        计算单次地 G_losses(包含GAN loss, KLD loss, GAN_feat loss, VGG loss)
        """
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake_byimg(input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses["KLD"] = KLD_loss

        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)

        G_losses["GAN"] = self.criterionGAN(pred_fake, target_is_real=True, for_discriminator=False)

        if self.opt.ganFeat_loss:
           num_D = len(pred_fake)
           GAN_Feat_loss = self.FloatTensor(1).fill_(0)
           for i in range(num_D):
               # 中间层不包含最后的预测层, 因此减一
               num_intermediate_outputs = len(pred_fake[i]) - 1
               for j in range(num_intermediate_outputs):
                   unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                   GAN_Feat_loss += unweighted_loss * self.opt.lamb_feat / num_D
           G_losses["GAN_Feat"] = GAN_Feat_loss
        else:
            G_losses["L1"] = self.criterionFeat(fake_image, real_image) * self.opt.lamb_L1

        if self.opt.vgg_loss:
            G_losses["VGG"] = self.criterionVGG(fake_image, real_image) * self.opt.lamb_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        """
        计算单次的 D_losses(包含, D_fake loss, D_real loss)
        """
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake_byimg(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_(requires_grad=True)      ## 将fake_image从计算图中分离出来, 然后设置需要更新梯度
                                                               ## (因为需要加入到判别器计算loss过程中, 用来计算D_fake loss)

        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)

        D_losses["D_fake"] = self.criterionGAN(pred_fake, target_is_real=False, for_discriminator=True)
        D_losses["D_real"] = self.criterionGAN(pred_real, target_is_real=True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake_byimg(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lamb_kld

        fake_image = self.netG(input_semantics, z=z)

        return fake_image, KLD_loss

    def generate_fake_byz(self, input_semantics, z):

        if z is None:
            z = torch.randn(input_semantics.size(0), self.opt.z_dim,
                            dtype=torch.float32, device=self.opt.device)

        fake_image = self.netG(input_semantics, z=z)

        return fake_image

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # 在batch Normalization中, fake图像和real图像推荐放在同一批次中以免normalize之后信息出现偏差
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def divide_pred(self, pred):
        """
        将fake和real的image预测结果从同一个批次中分开
        由于预测结果可能包含multiscale GAN(以list形式), 因此需要检测list type
        """
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0)//2] for tensor in p])
                real.append([tensor[tensor.size(0)//2:] for tensor in p])
        else:
            fake = pred[:pred.size(0)//2]
            real = pred[pred.size(0)//2:]

        return fake, real                 # pred无论是否为list, 也无论批次大小, 返回的fake和real都会是list

    def reparameterize(self, mu, logvar):
        """
        从 mu 和 logvar所代表的高斯分布下随机采样一个z向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu



class GauganTrainer():
    """
    创建Gaugan模型和optimizers, 并且用他们来更新权重, 并报告损失
    """
    def __init__(self, opt):
        self.opt = opt
        self.gaugan_model = GauganModel(opt)

        self.generated = None

        if opt.is_train:
            self.optimizer_G, self.optimizer_D = self.gaugan_model.create_optimizers(opt)
            self.old_lr = opt.lr


    def run_generate_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.gaugan_model(data, mode="generator")
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.generated = generated
        self.g_losses = g_losses

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.gaugan_model(data, mode="discriminator")
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        for key, value in self.g_losses.items():
            self.g_losses[key] = value.data.item()
        for key, value in self.d_losses.items():
            self.d_losses[key] = value.data.item()
        return {**self.d_losses, **self.g_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        if epoch > self.opt.keep_lr_epochs:
            lrd = self.opt.lr / (self.opt.epochs - self.opt.keep_lr_epochs)
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.TTUR:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2
            else:
                new_lr_G = new_lr
                new_lr_D = new_lr

            for param_group in self.optimizer_D.param_groups:
                param_group["lr"] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
        else:
            if self.opt.TTUR:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2
            else:
                new_lr_G, new_lr_D = new_lr, new_lr
            print('learning rate of G: %f' % (new_lr_G), 'learning rate of D: %f' % (new_lr_D))


    def save_latest(self, epoch, step, losses):
        print("saving model: " + "model_latest_epoch.jpp")
        torch.save({
            "epoch": epoch+1,
            "step": step,
            "losses": losses,
            "netG_state_dict": self.gaugan_model.netG.state_dict(),
            "netD_state_dict": self.gaugan_model.netD.state_dict(),
            "netE_state_dict": self.gaugan_model.netE.state_dict(),
            "optimizerG_state_dict": self.optimizer_G.state_dict(),
            "optimizerD_state_dict": self.optimizer_D.state_dict(),
        }, os.path.join(self.opt.model_path, "model_latest.jpp"))

    def save_epoch(self, epoch, step, losses):
        print("saving model: " + "model_epoch_%d.jpp" % epoch)
        torch.save({
            "epoch": epoch+1,
            "step": step,
            "losses": losses,
            "netG_state_dict": self.gaugan_model.netG.state_dict(),
            "netD_state_dict": self.gaugan_model.netD.state_dict(),
            "netE_state_dict": self.gaugan_model.netE.state_dict(),
            "optimizerG_state_dict": self.optimizer_G.state_dict(),
            "optimizerD_state_dict": self.optimizer_D.state_dict(),
        }, os.path.join(self.opt.model_path, "model_epoch_%d.jpp" % epoch))

    def save_imgs(self, data, epoch, path):
        g = self.gaugan_model(data, mode="inference")
        g = self.denormalize(g.view(-1, self.opt.in_nc, self.opt.image_size, self.opt.image_size))
        r = self.denormalize(data["image"])
        label = self.denormalize(data["label_image"])
        img_all = torch.cat([label, r, g], dim=3)
        save_image(img_all, path + 'epoch_%d.png' %(epoch), nrow=1, pad_value=255)

    def denormalize(self, x_hat):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1).to(self.opt.device)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1).to(self.opt.device)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean
        return x

    def print_info(self, epoch, batch, step, losses, opt):
        # losses = self.get_latest_losses()
        informations = ("[%d / %d]: batch#%d / steps#%d:\n " %(epoch, opt.epochs, batch, step))
        for key, value in losses.items():
            informations += (" loss_%s= %.3f" %(key, value))
        print(informations)
