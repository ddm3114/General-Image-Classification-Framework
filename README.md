# Genreral Image Classification Framework

目前框架内集成了：

## 计算机视觉：期中作业任务1：
### 在[CUB-200-2011]( https://data.caltech.edu/records/65de6-vp158)数据集上从零开始训练新的输出层，并对其余参数使用较小的学习率进行微调。
  使用框架resnet18

## 计算机视觉：期末作业任务2：
### 在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型.
  resnet34和swin transfomer
  
## 神经网络与深度学习：Project2：
### Train a Network on CIFAR-10
  自定义网络（两个baseblock作为主干）

---
# 使用说明
## 框架是自己写的！！！！！框架是自己写的！！！！！框架是自己写的！！！！！框架是自己写的！！！！！
## 自定义模型去model.py中的MyModel中更改。调模型可以根据model.py中的resnet34或swin transfomer更改。
## 调的新模型（不包括自定义模型）需要更改的部分：
- model.py
- get_model.py
- config.json
## 使用新的数据集需要更改的部分
- dataset.py
- get_dataloader.py
- config.json
## 使用新的优化器需要更改的部分
- get_optim.py
## 优化器中目前只支持的优化参数：
- 自定义网络的全部参数或者线性分类头（允许分别训练）
- 预训练网络中的全部参数和线性分类头（允许分别训练）
## 使用框架内存在的模型可以只修改config.json
## 训练使用train.py，推理部分需要自行修改inference.py
---
# *希望以后的培养方案少一些图像分类*
对框架有问题可联系：学邮:21307140101@m.fudan.edu.cn            私人邮箱:latzbnzl@gmail.com
