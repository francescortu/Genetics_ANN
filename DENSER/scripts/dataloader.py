from tkinter import N
import torch
import torchvision
import torchvision.transforms as transforms
import os


def cifar10(batch_size=4):
    transform = transforms.Compose([
                        transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)), ])

    # We download the train and the test dataset in the given root and applying the given transforms
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,  download=True, transform=transform)

    batch_size = batch_size

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,  shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,  shuffle=False, num_workers=2) 

    # input_size = 28*28
    input_size =  trainloader.dataset.data.shape[1]

    # number of classes
    n_classes = len(trainloader.dataset.classes)


    return trainloader, testloader, input_size, n_classes


def MNIST(batch_size=4):
    transform = transforms.Compose([
                        transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)), ])

    # We download the train and the test dataset in the given root and applying the given transforms
    trainset = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,  download=True, transform=transform)

    batch_size = batch_size

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,  shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,  shuffle=False, num_workers=2) 

    # input_size = 28*28
    input_size =  trainloader.dataset.data.shape[1]

    # number of classes
    n_classes = len(trainloader.dataset.classes)


    return trainloader, testloader, input_size, n_classes