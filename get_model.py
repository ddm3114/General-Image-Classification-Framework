import torch
from torchvision.models import resnet18,ResNet18_Weights
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
if torch.cuda.is_available():
    device = torch.device('cuda')
from utlis import read_sample,transform,save_dict
from model import ResNet18,ResNet50,Swin_T,baseModel
from dataset import CIFAR100_Dataset,Augmentation_Dataset
import json
import torch.optim.lr_scheduler as lr_scheduler
from get_optim import get_optim
from get_dataloader import get_dataloader

def get_model(model,pretrained = False,train_backbone = False,num_classes = 200):

    if model == 'ResNet18':
        model = ResNet18(pretrained=pretrained,train_backbone=train_backbone,num_classes= num_classes).to(device)
        print('ResNet18 model loaded')
    elif model == 'ResNet50':
        model = ResNet50(pretrained=pretrained,train_backbone=train_backbone,num_classes=num_classes).to(device)
        print('ResNet50 model loaded')
    elif model == 'Swin_T':
        model = Swin_T(pretrained=pretrained,train_backbone=train_backbone,num_classes=num_classes).to(device)
        print('Swin_T model loaded')

    elif model == 'baseModel':
        model = baseModel(num_classes=num_classes,train_backbone=train_backbone)
    else:
        raise ValueError('model not supported')
    
    return model