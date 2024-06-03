
import torch
from torchvision.io import read_image
from torchvision.models import resnet18,ResNet18_Weights
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision import transforms
from utlis import transform
from utlis import read_image,show_image
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100,CIFAR10
from torch.utils.data import DataLoader

class Augmentation_Dataset:
    def __init__(self,dataset,transform=None):
        self.transform = transform
        self.dataset = dataset
        self.length = int(dataset.__len__()//2)
        
        if dataset.classes:
            self.num_classes = len(dataset.classes)
        elif dataset.num_classes:
            self.num_classes = dataset.num_classes
        else:
            raise ValueError('Number of classes not found')
        self.cutmix = T.CutMix(num_classes = self.num_classes)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        img,label = self.augmentation(idx)
        
        
        
        return img, label
    
    def augmentation(self,idx):
        imgs = []
        labels = []

        image, label = self.dataset.__getitem__(2*idx)
        imgs.append(image)
        labels.append(label)
        image, label = self.dataset.__getitem__(2*idx+1)
        imgs.append(image)
        labels.append(label)


        # to_tensor = ToTensor()
       
        labels = torch.tensor(labels)
        imgs = torch.stack(imgs)
        img, label = self.cutmix(imgs, labels)
        return img[0],label[0]
                  

# 加载CIFAR-100数据集
class CIFAR100_Dataset:
    def __init__(self,transform = None):
        train_dataset = CIFAR100(root='./dataset', train=True, download=True, transform =transform)
        print(type(train_dataset))
        test_dataset = CIFAR100(root='./dataset', train=False, download=True, transform=transform)
        num_classes = len(train_dataset.classes)
        print('CIFAR100 loaded')
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"类别数: {num_classes}")
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes
    
   
class CIFAR10_Dataset:
    def __init__(self,transform = None):
        train_dataset = CIFAR10(root='./dataset', train=True, download=True, transform =transform)
        print(type(train_dataset))
        test_dataset = CIFAR10(root='./dataset', train=False, download=True, transform=transform)
        num_classes = len(train_dataset.classes)
        print('CIFAR10 loaded')
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"类别数: {num_classes}")
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes





if __name__ == '__main__':
    dataset = CIFAR10_Dataset(transform=transform)
    train_dataset,test_dataset = dataset.train_dataset,dataset.test_dataset
    num_classes = dataset.num_classes
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


