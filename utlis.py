
import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import json
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def read_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = ToTensor()(img)
    return img

def show_image(img,save_path):
    img.squeeze_(0)
    print('the shape of the image is :',img.shape)
    to_pil = ToPILImage()
    img_pil = to_pil(img)
    plt.imshow(img_pil)
    img_pil.save(save_path)
    print(f"Image saved to {save_path}")

def read_sample(sample,transform =None):
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

    return imgs, labels

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Resize((64,64), interpolation=transforms.InterpolationMode.BILINEAR)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用常用的均值和标准差进行标准化
])



def save_dict(model,config):
    os.makedirs(config['save_dir'],exist_ok=True)
    save_path = os.path.join(config['save_dir'],'model.pth')
    torch.save(model.state_dict(),save_path)

    config_path = os.path.join(config['save_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f'Model and config saved to {config["save_dir"]}')

def load_dict(model,config):
    model.load_state_dict(torch.load(config['save_dir']))
    
    print(f'model loaded from {config["save_dir"]}')
    return model

