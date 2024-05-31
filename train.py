import torch
from torchvision.io import read_image
from torchvision.models import resnet18,ResNet18_Weights
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.io import read_sample
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
if torch.cuda.is_available():
    device = torch.device('cuda')
from utlis import read_image,transform,save_dict,plot
from model import ResNet18,ResNet50,Swin_T
from dataset import train_dataloader,test_dataloader
import json

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def train(train_dataloader,test_dataloader,hyperparameters,num_classes = 200,save_model=False):
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
        model = ResNet18(pretrained=pretrained,train_backbone=train_backbone,num_classes= num_classes).to(device)
        print('ResNet18 model loaded')
    elif model == 'ResNet50':
        model = ResNet50(pretrained=pretrained,train_backbone=train_backbone,num_classes=num_classes).to(device)
        print('ResNet50 model loaded')
    elif model == 'Swin_T':
        model = Swin_T(pretrained=pretrained,train_backbone=train_backbone,num_classes=num_classes).to(device)
        print('Swin_T model loaded')
    else:
        raise ValueError('model not supported')
    
    if model.fc.parameters():
        if train_backbone and lr_backbone:

            optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': lr_backbone},
                                        {'params': model.fc.parameters(), 'lr': lr_head}],
                                        weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr_head, weight_decay=weight_decay)
    else:
        if train_backbone and lr_backbone:

            optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': lr_backbone},
                                        {'params': model.head.parameters(), 'lr': lr_head}],
                                        weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.head.parameters(), lr=lr_head, weight_decay=weight_decay)

    criterion = torch.nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    accuracy_list = []
    writer = SummaryWriter()

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = []
        train_accuracy_epoch = []
        train_accuracy_list = []
        for iter,sample in enumerate(train_dataloader):
            imgs,labels = read_image(sample,transform=transform)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())
            accuracy = (output.argmax(1) == labels).sum().item()/len(labels)
            train_accuracy_epoch.append(accuracy)
    
            if iter % 20 == 0:
                print(f'epoch:{epoch},iter:{iter},loss: {loss.item()}')
                current_memory = torch.cuda.memory_allocated()
                print(f"当前显存占用：{current_memory} bytes")
            del imgs
            del labels
        print(f'epoch:{epoch},train loss: {sum(train_loss_epoch)/len(train_loss_epoch)}')
        epoch_loss = sum(train_loss_epoch)/len(train_loss_epoch)
        epoch_accuracy = sum(train_accuracy_epoch)/len(train_accuracy_epoch)

        train_loss.append(loss)
        train_accuracy_list.append(epoch_accuracy)

        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_accuracy, epoch)
        
        test_loss_epoch = []
        test_accuracy_epoch = []
        model.eval()
        for sample in test_dataloader:
            img,label = read_sample(sample,transform)
            output = model(img)
            accuracy = (output.argmax(1) == label).sum().item()/len(label)
            loss = criterion(output, label)
            test_loss_epoch.append(loss.item())
            test_accuracy_epoch.append(accuracy)
            del img
            del label
            

        epoch_loss = sum(test_loss_epoch)/len(test_loss_epoch)
        epoch_accuracy = sum(test_accuracy_epoch)/len(test_accuracy_epoch)

        test_loss.append(epoch_loss)
        accuracy_list.append(epoch_accuracy)

        writer.add_scalar('Test Loss', epoch_loss, epoch)
        writer.add_scalar('Test Accuracy', epoch_accuracy, epoch)
        
        print(f'test loss: {sum(test_loss_epoch)/len(test_loss_epoch)}')
        print(f'accuracy: {epoch_accuracy}')
    if save_model:
        save_dict(model,hyperparameters)
    del model

    writer.close()

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
    with open('config.json', 'r') as f:
        configs = json.load(f)
    for config in configs:
        train_loss,test_loss,accuracy_list = train(train_dataloader,test_dataloader,config,num_classes=100,save_model=True)