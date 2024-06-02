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
from model import ResNet18,ResNet50,Swin_T
from dataset import CIFAR100_Dataset,Augmentation_Dataset
import json
import torch.optim.lr_scheduler as lr_scheduler
from get_optim import get_optim
from get_dataloader import get_dataloader
from get_model import get_model

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def train(hyperparameters,num_classes,augment = False,save_model=False):

    args = hyperparameters
    id = args['id']
    dataset = args['dataset']
    augment  = args['augment']
    epochs = args['epochs']
    model = args['model']
    optim = args['optimizer']
    pretrained = args['pretrained']
    print('pretrained:',pretrained)
    lr_head = args['lr_head']
    weight_decay = args['weight_decay']
    step_size = args['step_size']
    gamma = args['gamma']
    train_backbone = args['train_backbone']
    batch_size =args['batch_size']
    lr_backbone = args['lr_backbone'] if 'lr_backbone' in args else None
        
    writer = SummaryWriter(log_dir=f'runs/{id}')
    #tensorboard --logdir=runs

    model = get_model(model= model,pretrained=pretrained,train_backbone=train_backbone,num_classes=num_classes)
    
    if augment:
        train_dataloader,test_dataloader,augment_dataloader = get_dataloader(dataset = dataset,batch_size=batch_size,augment = augment)
    else:
        train_dataloader,test_dataloader = get_dataloader(dataset = dataset,batch_size=batch_size,augment = augment)

    optimizer = get_optim(model,lr_head,lr_backbone,weight_decay,train_backbone,optim=optim)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    criterion = torch.nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    accuracy_list = []
    
    for epoch in range(epochs):
        model.train()
        train_loss_epoch = []
        train_accuracy_epoch = []
        train_accuracy_list = []
        train_iter = 0
        print('Training train data')
        for iter,sample in enumerate(train_dataloader):
            train_iter= iter

            if isinstance(sample[0], str):
                img,label = read_sample(sample,transform)
            elif isinstance(sample[0],torch.Tensor):
                img,label = sample
            else:
                raise TypeError(f"'{type(model).__name__}' object has wrong type either str or torch.Tensor'")
        
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())
            accuracy = (output.argmax(1) == label).sum().item()/len(label)
            train_accuracy_epoch.append(accuracy)

            if iter % 100 == 0:
                print(f'[Training]epoch:{epoch},iter:{iter},loss: {loss.item()}')
            del img
            del label
        
        if augment:
            lam = 0.5
            print('Training augmenting data')
            for iter,sample in enumerate(augment_dataloader):
                img,label = sample

                img = img.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                output = model(img)
                loss = lam*criterion(output, label)

                loss.backward()
                optimizer.step()
                train_loss_epoch.append(loss.item()/lam)
                accuracy = (output.argmax(1) == label.argmax(1)).sum().item()/label.shape[0]
                train_accuracy_epoch.append(accuracy)
                del img
                del label
                if iter % 100 == 0:
                    print(f'[Training auguemted data]epoch:{epoch},iter:{train_iter+iter},loss: {loss.item()/lam}') 

        print(f'[Train]epoch:{epoch},train loss: {sum(train_loss_epoch)/len(train_loss_epoch)}')
        epoch_loss = sum(train_loss_epoch)/len(train_loss_epoch)
        epoch_accuracy = sum(train_accuracy_epoch)/len(train_accuracy_epoch)

        train_loss.append(loss)
        train_accuracy_list.append(epoch_accuracy)
        scheduler.step()
        if epoch % step_size == 0:
            print(f'[Train]lr changed to {scheduler.get_last_lr()}')
        
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_accuracy, epoch)

        
        print('\n')
        print('Testing')
        test_loss_epoch = []
        test_accuracy_epoch = []
        model.eval()
        for sample in test_dataloader:
            if isinstance(sample[0], str):
                img,label = read_sample(sample,transform)
            elif isinstance(sample[0],torch.Tensor):
                img,label = sample
            else:
                raise TypeError(f"'{type(model).__name__}' object has wrong type either str or torch.Tensor'")
            img = img.to(device)
            label = label.to(device)

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
        
        print(f'[Test]test loss: {sum(test_loss_epoch)/len(test_loss_epoch)}')
        print(f'[Test]accuracy: {epoch_accuracy}')
        print('-'*50)

    if save_model:
        save_dict(model,hyperparameters)
    del model

    writer.close()
    print("you can use '$ tensorboard --logdir=runs' to manage your model")
    return train_loss,test_loss,accuracy_list

if __name__ == '__main__':
    
    with open('config.json', 'r') as f:
        configs = json.load(f)

    for config in configs:
        train_loss,test_loss,accuracy_list = train(config,num_classes=100,augment = True,save_model=True)