import torch
from torchvision.models import resnet18,ResNet18_Weights
import os
from PIL import Image
if torch.cuda.is_available():
    device = torch.device('cuda')

from dataset import train_dataloader,test_dataloader
import json

# image = ['dataset/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg']
# model = ResNet18().to(device)
# model.load_state_dict(torch.load('model.pth'))
# with open('classes.txt', 'r') as f:
#     classes = f.readlines()
# classes = [x.strip() for x in classes]

# with torch.no_grad():
#     model.eval()
#     img = Image.open(image[0]).convert('RGB')
#     img = transform(img)
#     img = img.unsqueeze(0)
#     img = img.to(device)
#     output = model(img)
#     print(output.argmax(dim=1).item())
#     print(output.max().item())
#     print(output)
#     print(output.argmax(dim=1))
#     del model
#     del img