# urban_fabric_generation
Code and models for urban fabric generation using Pix2pix and GauGAN

该项目探究深度学习相关技术在城市肌理生成上的应用可能性，使用[Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)和[GauGAN](https://github.com/NVlabs/SPADE)两种图像转换模型对肌理成对数据进行学习，并得到了不错的效果。在项目中，我尝试了一种基于自编码器的特征提取方法，来比较不同模型生成效果上的差异，以此调制出更适合于肌理生成的模型超参数和模型架构，最后将模型成功应用于步进式条件生成和城市尺度肌理生成。

This project researches on applications of deep learning technologies in urban texture generation, uses [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [GauGAN](https://github.com/NVlabs/SPADE) models to learn urban fabric paired data, and gets good results. In the project, I tried a feature extraction method based on Autoencoder to compare the differences in the generation effect of different models, so as to modulate the model hyperparameters and model architecture which are more suitable for urban fabric generation. Finally, pretained models can ben applied to step-by-step conditional generation and city-scale fabric generation.

## 步进式生成/Step-by-step generation
![0](https://user-images.githubusercontent.com/35953653/119255814-450da780-bbf0-11eb-9fa0-7af90315faa6.gif)

## 城市尺度肌理生成/City-scale fabric generation
![容西肌理_2](https://user-images.githubusercontent.com/35953653/119256071-63c06e00-bbf1-11eb-9afa-ba99f693ee9b.png)
