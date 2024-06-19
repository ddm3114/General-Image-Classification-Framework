import torch
from torchvision.io import read_image

from torch.utils.data import DataLoader

from torchvision import transforms
if torch.cuda.is_available():
    device = torch.device('cuda')
from utlis import read_sample,transform,save_dict
from dataset import CIFAR100_Dataset,Augmentation_Dataset,CIFAR10_Dataset

def get_dataloader(dataset = 'CIFAR100',batch_size = 32,augment = False,transform = transform):
    if dataset == 'CIFAR100':
        dataset = CIFAR100_Dataset(transform=transform)
        train_dataset,test_dataset = dataset.train_dataset,dataset.test_dataset
        num_classes = dataset.num_classes
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        if augment:
            augmented_dataset = Augmentation_Dataset(train_dataset,transform=transform)
            augmented_dataloader = DataLoader(augmented_dataset,batch_size=batch_size,shuffle=True)
            return train_dataloader,test_dataloader,augmented_dataloader
        
    elif dataset == 'CIFAR10':
        dataset = CIFAR10_Dataset(transform=transform)
        train_dataset,test_dataset = dataset.train_dataset,dataset.test_dataset
        num_classes = dataset.num_classes
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        if augment:
            augmented_dataset = Augmentation_Dataset(train_dataset,transform=transform)
            augmented_dataloader = DataLoader(augmented_dataset,batch_size=batch_size,shuffle=True)
            return train_dataloader,test_dataloader,augmented_dataloader
        
    else:
        raise NameError(f"Dataset:{dataset} is not supported now")
    
    return train_dataloader,test_dataloader



if __name__ == '__main__':
    train,test = get_dataloader(dataset='1')