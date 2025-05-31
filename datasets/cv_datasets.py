import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

def load_cifar10(batch_size=64, val_split=0.1, data_root='./data', num_workers=2):
    '''Loads the CIFAR-10 dataset and returns train, validation, and test DataLoaders.'''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    if val_split > 0:
        val_size = int(len(full_trainset) * val_split)
        train_size = len(full_trainset) - val_size
        trainset, valset = random_split(full_trainset, [train_size, val_size])
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        trainset = full_trainset
        valloader = None # Or an empty DataLoader if preferred for consistency

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader
