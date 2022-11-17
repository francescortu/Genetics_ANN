from tkinter import N
import torch
import torchvision
import torchvision.transforms as transforms
import os


transform = transforms.Compose([
                    transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)), ])

def cifar10(batch_size=4, test = False):

    # We download the train and the test dataset in the given root and applying the given transforms
    trainset = get_dataset(cifar10)
    if test:
         testset = torchvision.datasets.CIFAR10(root='./data', train=False,  download=True, transform=transform)
    else:
        trainset, testset = torch.utils.data.random_split(trainset, [40000, 10000])


    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,  shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,  shuffle=False, num_workers=2) 

    # input_size 
    # input_size =  trainloader.dataset.data.shape[1]
    input_size = 32

    # number of classes
    # n_classes = len(trainloader.dataset.classes)
    n_classes = 10

    #input_channels
    input_channels = 3

    return trainloader, testloader, input_size, n_classes, input_channels


def MNIST(batch_size=4, test = False):

    # We download the train and the test dataset in the given root and applying the given transforms
    trainset = get_dataset(MNIST)
    

    if test:
        testset = torchvision.datasets.MNIST(root='./data', train=False,  download=True, transform=transform)
    else:
        trainset, testset = torch.utils.data.random_split(trainset, [50000, 10000],  generator=torch.Generator().manual_seed(42) )
  

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,  shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,  shuffle=False, num_workers=2) 
    # input_size
    # input_size =  trainloader.dataset.data.shape[1]
    input_size = 28

    # number of classes
    # n_classes = len(trainloader.dataset.classes)
    n_classes = 10

    #input_channels
    input_channels = 1


    return trainloader, testloader, input_size, n_classes, input_channels


def get_dataset(dataset):

    if dataset == cifar10:
        return torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    elif dataset == MNIST:
        return torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported")