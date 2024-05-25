
import torch
from torchvision.io import read_image
from torchvision.models import resnet18,ResNet18_Weights
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


root_path = 'dataset/CUB_200_2011/images/'
with open(r'dataset\CUB_200_2011\images.txt', 'r') as f:
    images = f.readlines()
images = [x.strip().split(' ')[1] for x in images]
images = [root_path+x for x in images]

with open(r'dataset\CUB_200_2011\image_class_labels.txt', 'r') as f:
    labels = f.readlines() 
labels = [x.strip().split(' ')[1] for x in labels]
labels = [int(x)-1 for x in labels]
with open(r'dataset\CUB_200_2011\train_test_split.txt', 'r') as f:
    train_test = f.readlines()
train_test = [x.strip().split(' ')[1] for x in train_test]

train_images = [images[i] for i in range(len(images)) if train_test[i] == '1']
train_labels = [labels[i] for i in range(len(images)) if train_test[i] == '1']

test_images = [images[i] for i in range(len(images)) if train_test[i] == '0']
test_labels = [labels[i] for i in range(len(images)) if train_test[i] == '0']
train_dataset = list(zip(train_images, train_labels))
test_dataset = list(zip(test_images, test_labels))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

