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

class ResNet18(torch.nn.Module):
    def __init__(self,pretrained = True,train_backbone = False):
        super(ResNet18, self).__init__()
        if train_backbone:
            self.resnet18 = resnet18(pretrained=pretrained)
        else:
            self.resnet18 = resnet18(pretrained=pretrained)
            for param in self.resnet18.parameters():
                param.requires_grad = False
                
        self.classifier = torch.nn.Linear(1000, 200)
    def forward(self, x):
        x = self.resnet18(x)
        x = self.classifier(x)
        return x

