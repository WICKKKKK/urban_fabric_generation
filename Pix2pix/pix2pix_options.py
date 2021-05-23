#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
author: WICKKK
date: 2021/4/1
"""


import os
import torch

class Opt():
    def __init__(self):

        # 设置cpu or gpu
        self.device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

        # 设置路径
        self.data_root = "../data/city_fabric/"  # 数据集根目录
        self.train_subfolder = "train/"       # 训练数据集目录
        self.test_subfolder = "test/"         # 测试数据集目录

        self.model_path = "../models/Pix2pix_multiD3_L1_G4D1_TL/"      # 模型保存目录

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.train_img_save_path = os.path.join(self.model_path, "generated_train/")    # 训练集生成数据保存目录
        self.test_img_save_path = os.path.join(self.model_path, "generated_test/")      # 测试集生成数据保存目录

        self.test_img_infer_path = os.path.join(self.model_path, "test_result/")        # 测试集inference保存目录

        # 预加载模型设置
        self.preload = False                                                       # 设置是否进行预加载模型
        self.model_name = "model_latest.jpp"                                       # 设置预加载模型的名称
        self.preload_model_path = os.path.join(self.model_path, self.model_name)   # 生成预加载模型的路径
        self.infer_model_name = "model_latest.jpp"                                    # 测试模型的名称
        self.infer_model_path = os.path.join(self.model_path, self.infer_model_name)  # 测试模型的路径

        self.batch_size = 1        # 数据批次数
        self.image_size = 256      # 输入图片的大小

        # 定义结构参数
        self.in_ch = 3            # 输入图片通道数
        self.out_ch = 3           # 输出图片通道数
        self.fch_g = 64           # 生成器第一层卷积通道数
        self.fch_d = 64           # 判别器第一层卷积通道数

        self.G_type = "ResNet"               # Generator的类型, 请选择 UNet | ResNet 其中一种
        self.padding_mode = "reflect"      # padding模式, 请选择 zero | reflect | replicate 其中一种
                                           # 默认是reflect, 因为对于肌理图来说, 边缘镜像地扩展会是一种相对合理的方式
        self.n_block = 24                   # 如果选用ResNet, 需要指定一下残差块的数量, source code中有 6/9 两种选项, 可以根据实际来调节
        self.norm_type = "batch"           # Normalization的类型, 请选择 batch | instance | none 其中一种
        self.use_dropout = True            # 选择是否打开生成器中的Dropout层, UNet中是upsampling的前三层默认打开
        if self.G_type == "ResNet":
            self.use_dropout = False       # ResNet中默认是没有使用Dropout层
            self.norm_type = "instance"    # CycleGAN中默认的Normalization type是instance norm,
                                           # 原作者认为ResNet在经过第一层卷积层(7x7 kernel)之后会将颜色信息给编码下来, 因此使用了instance norm
                                           # 而在pix2pix中, 使用instance norm在卷积层后会损失颜色信息, 因此使用batch norm
                                           # 但原作者又提到, 为了在pix2pix中实现instance norm, 采用了batch size = 1的设定, 就很奇怪...
                                           # (确实之前的实验中, CycleGAN对于颜色信息缺失会比较严重, 而pix2pix则相对较好)
        self.num_D = 2                     # 选择多尺度判别器的大小, 如果为1, 那么就是原始的30x30patchGAN
        self.gan_feat_loss = False         # 选择是否导出判别器的中间层feature来计算feature matching loss

        # 权重初始化参数
        self.init_type = "normal"      # 网络权重初始化的方式, 请选择 normal | xavier | kaiming | orthogonal 其中一种
        self.init_gain = 0.02          # 对于 normal, xavier and orthogonal方法的scale系数

        # 学习率相关参数
        self.lr_G = 0.0002        # 生成器学习率
        self.lr_D = 0.0002        # 判别器学习率
        self.lr_policy = "linear"      # 学习率变化规则, 请选择 linear | step | plateau | cosine 其中一种
        self.G_per_steps = 1      # 每次生成器和判别器迭代循环中, 生成器训练的次数

        # 损失函数相关参数
        self.beta1 = 0.5          # momentum term of Adam (一般用0.9)
        if self.gan_feat_loss:
          self.lamb = 10.0         # 如果设置gan_feat_loss时, 默认权重为10.0(pix2pixHD原论文)
        else:
          self.lamb = 100.0        # 不计算gan_feat_loss时, 则计算real image与fake image的L1Loss, 默认权重为100.0(pix2pix论文)

        # 训练相关参数
        self.epochs = 200              # 总epochs数量
        self.keep_lr_epochs = 100      # 保持初始学习率的epochs数量
        self.start_epoch = 1           # 起始epoch
        self.step = 0            # 起始step

        # 定义保存输出显示参数
        self.train_generated_freq = 1  # 每 n 个epoch保存一次训练时的生成效果
        self.test_generated_freq = 1   # 每 n 个epoch保存一次测试集的测试效果
        self.test_pick = 8             # 在测试集中随机选择 n 张图片在训练过程中进行inference
        self.model_save_freq = 5       # 每 n 个epoch保存一次模型
        self.model_latest_save_freq = 1  # 每 n 个epoch保存一次模型(覆盖)
        self.print_freq = 100          # 每 n 个step 打印一次信息 (会受到批次的影响而产生不同)
        self.display_freq = 500        # 每 n 个step 在visdom上显示图片
        self.display_loss_freq = 100   # 每 n 个step 在visdom上显示损失变化