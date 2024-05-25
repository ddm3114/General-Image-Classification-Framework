
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

def read_image(sample,transform =None):
    imgs =[]
    labels = []
    for i in range(len(sample[0])):
        img = sample[0][i]
        label = sample[1][i]
    
        img = Image.open(img).convert('RGB')
        if transform:
            img = transform(img)
        else:
            img = ToTensor()(img)
        if img.shape[0] != 3:
            continue
        if label >=200:
            continue
        
        imgs.append(img)
        labels.append(label)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)

    imgs = imgs.to(device)
    labels = labels.to(device)
    return imgs, labels

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Resize((224, 224)),  # 将图像的大小调整为 224x224
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用常用的均值和标准差进行标准化
])

import matplotlib.pyplot as plt

def plot(train_loss, test_loss, accuracy_list, image_dir=None):
    # 创建目标目录（如果不存在）
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)

    # 绘制并保存训练和测试损失图
    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.legend()
    plt.title('Train and Test Loss')
    loss_plot_path = os.path.join(image_dir, 'loss_plot.png') if image_dir else 'loss_plot.png'
    plt.savefig(loss_plot_path)

    # 绘制并保存准确率图
    plt.figure()
    plt.plot(accuracy_list, label='accuracy')
    plt.legend()
    plt.title('Accuracy')
    accuracy_plot_path = os.path.join(image_dir, 'accuracy_plot.png') if image_dir else 'accuracy_plot.png'
    plt.savefig(accuracy_plot_path)



def save_dict(model,config):
    os.makedirs(config['save_dir'],exist_ok=True)
    save_path = os.path.join(config['save_dir'],'model.pth')
    torch.save(model.state_dict(),save_path)
    print(f'model saved in {config["save_dir"]}')

def load_dict(model,config):
    model.load_state_dict(torch.load(config['save_dir']))
    print(f'model loaded from {config["save_dir"]}')
    return model

