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
from utlis import read_image,transform,save_dict,plot
from model import ResNet18
from dataset import train_dataloader,test_dataloader
import json

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def train(train_dataloader,test_dataloader,hyperparameters,save_model=False):
    args = hyperparameters
    epochs = args['epochs']
    model = args['model']
    pretrained = args['pretrained']
    print('pretrained:',pretrained)
    lr_head = args['lr_head']
    weight_decay = args['weight_decay']
    train_backbone = args['train_backbone']

    if train_backbone:
        lr_backbone = args['lr_backbone']
        

    if model == 'ResNet18':
        model = ResNet18(pretrained=pretrained,train_backbone=train_backbone).to(device)
        print('ResNet18 model loaded')
        print('lr_head:',lr_head)
        print('weight_decay:',weight_decay)
        if train_backbone:
            print('train_backbone:',train_backbone)
            print('lr_backbone:',args['lr_backbone'])
        print('epochs:',epochs)
    else:
        raise ValueError('model not supported')
    
    if train_backbone and lr_backbone:

        optimizer = torch.optim.Adam([{'params': model.resnet18.parameters(), 'lr': lr_backbone},
                                    {'params': model.classifier.parameters(), 'lr': lr_head}],
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr_head, weight_decay=weight_decay)

    criterion = torch.nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    accuracy_list = []

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = []
        for iter,sample in enumerate(train_dataloader):
            imgs,labels = read_image(sample,transform=transform)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())

            
            
            if iter % 20 == 0:
                print(f'epoch:{epoch},iter:{iter},loss: {loss.item()}')
                current_memory = torch.cuda.memory_allocated()
                print(f"当前显存占用：{current_memory} bytes")
            del imgs
            del labels
        print(f'epoch:{epoch},train loss: {sum(train_loss_epoch)/len(train_loss_epoch)}')
        train_loss.append(sum(train_loss_epoch)/len(train_loss_epoch))

        
        test_loss_epoch = []
        accuracy_epoch = []
        model.eval()
        for sample in test_dataloader:
            img,label = read_image(sample,transform)
            output = model(img)
            accuracy = (output.argmax(1) == label).sum().item()/len(label)
            loss = criterion(output, label)
            test_loss_epoch.append(loss.item())
            del img
            del label
            accuracy_epoch.append(accuracy)
        
        test_loss.append(sum(test_loss_epoch)/len(test_loss_epoch))
        accuracy_list.append(sum(accuracy_epoch)/len(accuracy_epoch))
        print(f'test loss: {sum(test_loss_epoch)/len(test_loss_epoch)}')
        print(f'accuracy: {sum(accuracy_epoch)/len(accuracy_epoch)}')
    if save_model:
        save_dict(model,hyperparameters)
    del model
    return train_loss,test_loss,accuracy_list

if __name__ == '__main__':
    # with open('config.json', 'r') as f:
    #     configs = json.load(f)
    # for config in configs:
    #     train_loss,test_loss,accuracy_list = train(train_dataloader,test_dataloader,config,save_model=True)
    #     plot(train_loss,test_loss,accuracy_list,config['image_dir'])

    # with open('config_pretrained.json', 'r') as f:
    #     configs = json.load(f)
    # for config in configs:
    #     train_loss,test_loss,accuracy_list = train(train_dataloader,test_dataloader,config,save_model=True)
    #     plot(train_loss,test_loss,accuracy_list,config['image_dir'])
    with open('config_pretrained.json', 'r') as f:
        configs = json.load(f)
    for config in configs:
        if config['id'] == 'best_parameters':
            train_loss,test_loss,accuracy_list = train(train_dataloader,test_dataloader,config,save_model=True)
            plot(train_loss,test_loss,accuracy_list,config['image_dir'])