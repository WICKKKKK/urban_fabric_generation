#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/5
"""

import torch
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self):
        # 设置cpu or gpu
        self.use_gpu = False
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda: 0")

        # 设置路径
        self.data_root = "../data/city_fabric/"  # 数据集根目录
        self.train_subfolder = "train/"       # 训练数据集目录
        self.test_subfolder = "test/"        # 测试数据集目录

        self.model_path = "../models/GauGAN_G4D1_TL/"      # 模型保存目录

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 预加载模型设置
        self.model_name = "model_latest.jpp"                                       # 设置预加载模型的名称
        self.preload_model_path = os.path.join(self.model_path, self.model_name)   # 生成预加载模型的路径


        self.batch_size = 1        # 数据批次数
        self.image_size = 256      # 输入图片的大小

        # 定义结构参数
        self.in_nc = 3            # 输入图片通道数
        self.out_nc = 3           # 输出图片通道数
        self.label_nc = 3         # 但不准备将label数据整理成 one-hot 格式时, 就按照默认的RGB三通道来输入到SPADE中
        self.fnc_g = 64           # 生成器第一层卷积通道数

        # 设置标签类别颜色值, 会自动根据所给定的颜色值, 来对label进行image segmentation, 将图像上的颜色归类到所给出的颜色值中
        # 颜色值可以根据指定需求来给出, 不限数量, 算法会自动将数据整理成 one-hot 形式来进行输入(在channel维度上合并)
        self.one_hot = True
        if self.one_hot:
            self.labels = {}
            self.labels["道路"] = [255, 255, 255]
            self.labels["R"] = [255, 255, 100]
            self.labels["A"] = [255, 100, 200]
            self.labels["B"] = [255, 0, 0]
            self.labels["M"] = [200, 150, 100]
            self.labels["W"] = [180, 90, 255]
            self.labels["S"] = [180, 180, 180]
            self.labels["U"] = [0, 150, 200]
            self.labels["G"] = [0, 205, 0]
            self.labels["E"] = [0, 255, 255]
            self.labels["其他"] = [0, 0, 0]

            self.label_nc = len(self.labels)

            self.labels_index = {}
            starting_index = 1
            for key in self.labels.keys():
                self.labels_index[key] = starting_index
                starting_index += 1

        self.netG = "spade"                # Generator的类型, 请选择 spade | pix2pixHD 其中一种
        self.padding_mode = "reflect"      # padding模式, 请选择 zero | reflect | replicate 其中一种
                                           # 默认是reflect, 因为对于肌理图来说, 边缘镜像地扩展会是一种相对合理的方式
        self.norm_G = "batch"              # Normalization的类型, 请选择 batch | instance | none 其中一种
        self.apply_spectral = True         # 是否在卷积层后添加 spectral Normalization
        self.kernel_size = 3               # SPADE 中卷积层的 kernel size
        if self.netG == "spade":
            self.num_up_layers = "most"   # 选择 upsampling 层的数量模式, 请选择 normal | more | most 其中一种
                                            # normal是5层, more是6层, most是7层
                                            # 一般推荐使用 "most" 即可, 能够使得生成器一开始对z向量的全连接层参数量减少许多, 模型量减少一些, 训练起来能够更快一些
        elif self.netG == "pix2pixHD":
            self.resnet_n_downsample = 4    # pix2pixHD中下采样层的数量
            self.resnet_n_blocks = 9        # 使用残差块的数量
            self.resnet_kernel_size = 3     # 残差块中卷积层的 kernel size
            self.resnet_initial_kernel_size = 7   # 第一层的卷积层所使用的 kernel size
            self.norm_G = "instance"              # 仅控制pix2pixHD生成器中Normalization层的类型, 请选择 instance | batch | none 其中一种
        else:
            raise NotImplementedError('generator method [%s] is not implemented' % self.netG)

        self.use_vae = True                # 是否使用 vae, 如果是则训练 encoder
        self.norm_E = "instance"           # encoder 中的 Normalization 层类型, 请选择 batch | instance | none 其中一种
        self.fnc_e = 64                    # encoder 第一层卷积的输出通道数
        self.z_dim = 256                   # latent z vector 维度
        self.infer_byz = False             # inference选择通过手动输入z来生成, 还是通过给定一张风格图像来生成

        # 权重初始化参数
        self.init_type = "xavier"      # 网络权重初始化的方式, 请选择 normal | xavier | xavier_uniform | kaiming | orthogonal | none 其中一种
        self.init_gain = 0.02          # 对于 normal, xavier and orthogonal方法的scale系数

        self.initialized = True
        self.is_train = False




class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self):
        super().initialize()

        self.is_train = True

        self.continue_train = False                                                     # 设置是否进行预加载模型

        self.train_img_save_path = os.path.join(self.model_path, "generated_train/")    # 训练集生成数据保存目录
        self.test_img_save_path = os.path.join(self.model_path, "generated_test/")      # 测试集生成数据保存目录
        if not os.path.exists(self.train_img_save_path):
            os.mkdir(self.train_img_save_path)
        if not os.path.exists(self.test_img_save_path):
            os.mkdir(self.test_img_save_path)

        # GAN 损失函数的类型 以及 判别器的类型
        self.gan_loss_mode = "hinge"         # 选择使用哪种类型的 gan 损失函数, 请选择 hinge | original | lsgan 其中一种
        self.netD = "multiscale"             # 选择那种类型的判别器, 请选择 patch | multiscale 其中一种
        self.norm_D = "instance"             # Normalization的类型, 请选择 batch | instance | none 其中一种
        self.n_layers_D = 4                  # 判别器卷积层的数目
        if self.netD == "multiscale":
            self.num_D = 2                   # 选择几种尺度的判别器
        elif self.netD == "patch":
            pass
        else:
            raise NotImplementedError('discriminator method [%s] is not implemented' % self.netD)
        self.fnc_d = 64                      # 判别器第一层卷积通道数

        # 损失函数相关参数
        self.TTUR = True          # 判别器和生成器的学习率变成 4:1
        self.beta1 = 0.0          # momentum term of Adam
        self.beta2 = 0.9
        if not self.TTUR:
            self.beta1 = 0.5
            self.beta2 = 0.999
        self.lamb_feat = 10.0           # feature matching loss 的权重
        self.lamb_vgg = 10.0            # vgg loss 的权重
        self.lamb_kld = 0.05            # KL Divergence loss 的权重
        self.lamb_L1 = 10.0            # 如果不计算 feature matching loss, 则计算real image与fake image的L1 loss, 与pix2pix一样的系数
        self.ganFeat_loss = True         # 是否计算 feature matching loss
        self.vgg_loss = True             # 是否计算vgg loss

        self.vgg_model_path = "../models/"       # 设置预训练的vgg网络模型存放位置(文件或文件夹都可以)

                                         # 如果该path上找不到该模型则会自动下载到该位置上


        # 学习率相关参数
        self.lr = 0.0002               # 学习率
        self.lr_policy = "linear"      # 学习率变化规则, 请选择 linear | step | plateau | cosine 其中一种
        self.optimizer = "adam"
        self.G_per_steps = 1           # 每次生成器和判别器迭代循环中, 生成器训练的次数
        # self.D_steps_per_G = 1         # 判别器更新的步数与生成器更新的步数的比率, 原文中使用TTUR和训练判别器多次等方法让生成器训练更加稳定, 但我感觉在肌理生成项目中, 使用这些技术会使生成器性能下降.

        # 训练相关参数
        self.epochs = 200              # 总epochs数量
        self.keep_lr_epochs = 100      # 保持初始学习率的epochs数量

        # 定义保存输出显示参数
        self.train_generated_freq = 1  # 每 n 个epoch保存一次训练时的生成效果
        self.test_generated_freq = 1   # 每 n 个epoch保存一次测试集的测试效果
        self.test_pick = 8             # 在测试集中随机选择 n 张图片在训练过程中进行inference
        self.model_save_epoch_freq = 5       # 每 n 个epoch保存一次模型(另存为)
        self.model_save_latest_freq = 1      # 每 n 个epoch保存一次最新模型(覆盖)
        self.print_freq = 100          # 每 n 个step 打印一次信息 (会受到批次的影响而产生不同)
        # self.display_freq = 500        # 每 n 个step 在visdom上显示图片
        # self.display_loss_freq = 100   # 每 n 个step 在visdom上显示损失变化


class TestOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self):
        super().initialize()

        self.is_train = False
        self.style_subfolder = "style/"                                             # 测试使用的样式数据集目录

        self.test_img_infer_path = os.path.join(self.model_path, "test_result/")  # 测试集inference保存目录
        if not os.path.exists(self.test_img_infer_path):
            os.mkdir(self.test_img_infer_path)




def main():
    opt = TrainOptions()
    opt.initialize()
    print(opt.preload_model_path)


if __name__ == '__main__':
    main()