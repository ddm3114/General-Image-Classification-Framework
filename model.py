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
from transformers import ViTModel
from torch import nn
import torch.nn.init as init

import torch.nn.functional as F
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


class ClassifierHead(nn.Module):
    def __init__(self, in_features,hidden_dim = 1024, num_classes=1024):
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.residual = nn.Linear(in_features, num_classes)
        self._initialize_weights()

    def forward(self, x):
        identity = self.residual(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        x += identity  # Residual connection
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化 (Kaiming 初始化)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier 初始化 (Glorot 初始化)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

## 通用于调取外部模型，只需要将mymodel的名字传入即可
class baseModel(torch.nn.Module):
    def __init__(self,model_name,pretrained = True,train_backbone = False,hidden_dim = 1024,num_classes = 200):
        super(baseModel, self).__init__()
        self.model = self.create_model(model_name,pretrained)
        if train_backbone:
            for param in self.model.parameters():
                param.requires_grad = True
        else:

            for param in self.model.parameters():
                param.requires_grad = False

        if hasattr(self.model, 'head'):
            num_ftrs = self.model.head.in_features
            self.model.head = ClassifierHead(num_ftrs,hidden_dim,num_classes)     
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'fc'):
            num_ftrs = self.model.fc.in_features
            self.model.fc = ClassifierHead(num_ftrs,hidden_dim,num_classes)     
            for param in self.model.fc.parameters():
                param.requires_grad = True

        elif hasattr(self.model, 'classifier'):
            num_ftrs = self.model.fc.in_features
            self.model.fc = ClassifierHead(num_ftrs,hidden_dim,num_classes)     
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x
    
    def create_model(self,model_name,pretrained):
        if model_name == 'ResNet18':
            model = resnet18(pretrained=pretrained)
            print('ResNet18 model loaded')
        elif model_name == 'ResNet50':
            if pretrained == True:
                model = resnet50(weights = 'DEFAULT')
            else:
                model = resnet50()
            print('ResNet50 model loaded')

        elif model_name == 'Swin_T':
            if pretrained == True:
                model = swin_t(weights = Swin_T_Weights.DEFAULT)
            else:
                model = swin_t()
            print('Swin_T model loaded')

        elif model_name == 'MyModel':
            model = MyModel(pretrained=pretrained)
            print('Your Custom_Model loaded')
        else:
            raise ValueError('model not supported')
        
        return model



# class ResNet50(torch.nn.Module):
#     def __init__(self,pretrained = True,train_backbone = False,hidden_dim=4096,num_classes = 200):
#         super(ResNet50, self).__init__()
#         if pretrained:
#             self.model = resnet50(weights = 'DEFAULT')
#         else:
#             self.model = resnet50()

#         if train_backbone:
#             for param in self.model.parameters():
#                 param.requires_grad = True
#         else:
#             for param in self.model.parameters():
#                 param.requires_grad = False

#         num_ftrs = self.model.fc.in_features
#         self.model.fc = ClassifierHead(num_ftrs,hidden_dim, num_classes)
#         for param in self.model.fc.parameters():
#             param.requires_grad = True

#     def forward(self, x):
#         x = self.model(x)
#         return x

# class Swin_T(torch.nn.Module):
#     def __init__(self,pretrained = True,train_backbone = False,hidden_dim = 4096,num_classes = 200):
#         super(Swin_T, self).__init__()
#         if pretrained:
#             self.model = swin_t(weights = Swin_T_Weights.DEFAULT)
#         else:
#             self.model = swin_t()
#         if train_backbone:
#             for param in self.model.parameters():
#                 param.requires_grad = True
#         else:

#             for param in self.model.parameters():
#                 param.requires_grad = False

#         num_ftrs = self.model.head.in_features
#         self.model.head =  ClassifierHead(num_ftrs,hidden_dim,num_classes)     

#         for param in self.model.head.parameters():
#             param.requires_grad = True

#     def forward(self, x):
#         x = self.model(x)
#         return x
    
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out
    
class MyModel(nn.Module):
    def __init__(self, pretrained = True,num_classes=1000):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = BasicBlock(64,64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = BasicBlock(128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self._initialize_weights()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)

        x = self.maxpool(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化 (Kaiming 初始化)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier 初始化 (Glorot 初始化)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class ClassifierHead(nn.Module):
    def __init__(self, in_features, hidden_dim = 1024, num_classes=1024):
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.residual = nn.Linear(in_features, num_classes)
        self._initialize_weights()

    def forward(self, x):
        identity = self.residual(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        x += identity  # Residual connection
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化 (Kaiming 初始化)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier 初始化 (Glorot 初始化)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

## 通用于调取外部模型，只需要将mymodel的名字传入即可
    
if __name__ == "__main__":
    
    model = baseModel(model_name='MyModel',pretrained=False,train_backbone=False,hidden_dim=1024,num_classes=200)   
    print(model)
    
    print("swin_t: ", sum(p.numel() for p in model.parameters()))
    # model = ResNet50(pretrained= False)
    # print("resnet50: ", sum(p.numel() for p in model.parameters()))
    # model = baseModel(num_classes=10)
    # print(model)
    for name, parms in model.named_parameters():
            if parms.requires_grad:
                print(f"{name}: {parms.data}")