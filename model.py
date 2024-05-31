import torch
from torchvision.io import read_image
from torchvision.models import resnet18,ResNet18_Weights
from torchvision.models import resnet50,ResNet50_Weights
from torchvision.models import swin_t,Swin_T_Weights
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

class ResNet18(torch.nn.Module):
    def __init__(self,pretrained = True,train_backbone = False,num_classes = 200):
        super(ResNet18, self).__init__()
        if train_backbone:
            self.resnet18 = resnet18(pretrained=pretrained)
        else:
            self.resnet18 = resnet18(pretrained=pretrained)
            for param in self.resnet18.parameters():
                param.requires_grad = False
                
        self.classifier = torch.nn.Linear(1000, num_classes)
    def forward(self, x):
        x = self.resnet18(x)
        x = self.classifier(x)
        return x

class ResNet50(torch.nn.Module):
    def __init__(self,pretrained = True,train_backbone = False,num_classes = 200):
        super(ResNet50, self).__init__()
        if pretrained:
            self.resnet50 = resnet50(weights = 'DEFAULT')
        else:
            self.resnet50 = resnet50()

        if train_backbone:
            for param in self.resnet50.parameters():
                param.requires_grad = True
        else:
            for param in self.resnet50.parameters():
                param.requires_grad = False

        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = torch.nn.Linear(num_ftrs, num_classes)        
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet50(x)
        return x

class Swin_T(torch.nn.Module):
    def __init__(self,pretrained = True,train_backbone = False,num_classes = 200):
        super(Swin_T, self).__init__()
        if pretrained:
            self.swin_t = swin_t(weights = Swin_T_Weights.DEFAULT)
        else:
            self.swin_t = swin_t()
        if train_backbone:
            for param in self.swin_t.parameters():
                param.requires_grad = True
        else:

            for param in self.swin_t.parameters():
                param.requires_grad = False

        num_ftrs = self.swin_t.head.in_features
        self.swin_t.head = torch.nn.Linear(num_ftrs, num_classes)     

        for param in self.swin_t.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.swin_t(x)
        return x
    
if __name__ == "__main__":
    model = Swin_T(pretrained= False)
    print("swin_t: ", sum(p.numel() for p in model.parameters()))
    model = ResNet50(pretrained= False)
    print("resnet50: ", sum(p.numel() for p in model.parameters()))