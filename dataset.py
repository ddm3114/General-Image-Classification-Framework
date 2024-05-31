
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
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

class Augmentation_Dataset:
    def __init__(self,dataset,transform=None):
        self.transform = transform
        self.dataset = dataset

        if dataset.classes:
            self.num_classes = len(dataset.classes)
        elif dataset.num_classes:
            self.num_classes = dataset.num_classes
        else:
            raise ValueError('Number of classes not found')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image, label = self.dataset.__getitem__(2*idx)
        
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def augmentation(self):
        pass



# 加载CIFAR-100数据集
train_dataset = CIFAR100(root='./dataset', train=True, download=True, transform = transform)
print(type(train_dataset))
test_dataset = CIFAR100(root='./dataset', train=False, download=True, transform=transform)
num_classes = len(train_dataset.classes)
print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"类别数: {num_classes}")
# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# root_path = 'dataset/CUB_200_2011/images/'
# with open(r'dataset\CUB_200_2011\images.txt', 'r') as f:
#     images = f.readlines()
# images = [x.strip().split(' ')[1] for x in images]
# images = [root_path+x for x in images]

# with open(r'dataset\CUB_200_2011\image_class_labels.txt', 'r') as f:
#     labels = f.readlines() 
# labels = [x.strip().split(' ')[1] for x in labels]
# labels = [int(x)-1 for x in labels]
# with open(r'dataset\CUB_200_2011\train_test_split.txt', 'r') as f:
#     train_test = f.readlines()
# train_test = [x.strip().split(' ')[1] for x in train_test]

# train_images = [images[i] for i in range(len(images)) if train_test[i] == '1']
# train_labels = [labels[i] for i in range(len(images)) if train_test[i] == '1']

# test_images = [images[i] for i in range(len(images)) if train_test[i] == '0']
# test_labels = [labels[i] for i in range(len(images)) if train_test[i] == '0']
# train_dataset = list(zip(train_images, train_labels))
# test_dataset = list(zip(test_images, test_labels))

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)



if __name__ == '__main__':
    cutmix = T.CutMix(num_classes = num_classes)
    for i, (img, label) in enumerate(train_dataloader):
        print(img.shape,label.shape)
        show_image(img[0],'result/original_0.png')
        show_image(img[1],'result/original_1.png')
        img, label = cutmix(img, label)
        print(img.shape,label.shape)
        img_pth = f'result/image_{i}.png'
        show_image(img[0],img_pth)
        print("label:",label[0])
        break