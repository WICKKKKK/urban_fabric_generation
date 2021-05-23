# urban_fabric_generation
Code and models for urban fabric generation using Pix2pix and GauGAN

该项目研究深度学习相关技术在城市肌理生成上的应用可能性，使用[Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)和[GauGAN](https://github.com/NVlabs/SPADE)两种图像转换模型对肌理成对数据进行学习，并得到不错的结果。在项目中，我尝试了一种基于自编码器的特征提取方法，来比较不同模型生成效果上的差异，以此调制出更适合于肌理生成的模型超参数和模型架构，最后将模型应用于步进式条件生成和城市尺度肌理生成。

This project researches on applications of deep learning technologies in urban texture generation, uses [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [GauGAN](https://github.com/NVlabs/SPADE) models to learn urban fabric paired data, and gets good results. In the project, I tried a feature extraction method based on Autoencoder to compare the differences in the generation effect of different models, so as to modulate the model hyperparameters and model architecture which are more suitable for urban fabric generation. Finally, pretained models can ben applied to step-by-step conditional generation and city-scale fabric generation.

## 步进式生成/Step-by-step generation
![0](https://user-images.githubusercontent.com/35953653/119255814-450da780-bbf0-11eb-9fa0-7af90315faa6.gif)

## 城市尺度肌理生成/City-scale fabric generation
![容西肌理_2](https://user-images.githubusercontent.com/35953653/119256071-63c06e00-bbf1-11eb-9afa-ba99f693ee9b.png)

## 使用/Usage
该项目使用Pytorch 1.2（更新的版本应该也可以），运行项目代码需要在你的Python环境中安装相应的库，可通过以下命令实现：

`pip install -r requirements.txt`

文件中已经放入城市肌理数据集和雄安西区容西片区土地利用规划简化图，因此配制好环境即可直接运行代码进行模型训练。默认参数下代码可在满足4G显存条件的电脑上运行（前提是已安装CUDA和相应的cudnn），如果配置允许，可适当调大Batch size或者调整网络架构（架构调整可通过options参数界面来实现），以达到更好的模型训练效果。项目所用模型在16G Tesla V100显卡上训练200 epochs得到，Batch size为12。 由于模型较大，因此文件中不包含训练完成的模型，如有需要可通过微信联系我：18482185419

Dataset of urban fabric and the simplified image of land use planning in Rongxi Xiongan have been put into this project, which can be trained directly by running the code. Under the default parameters, code can be run on a computer with 4G GPU (provided that CUDA and the corresponding cudnn are installed). if the configuration allows, you can appropriately adjust the Batch size or adjust the network architecture (architecture adjustment can be achieved through options module) to achieve better model training results. The model used in the project was trained on a 16G Tesla V100 GPU for 200 epochs, Batch size is 12. Due to the pre-training model is too large, the final trained model is not included in the project. If you need it, you can contact me through Wechat: 18482185419
