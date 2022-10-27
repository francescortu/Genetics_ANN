import torch
import torchvision
import torchvision.transforms as transforms
import os




def cifar10():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

def MINST():
        transform = transforms.Compose([
                   transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)), ])

        # We download the train and the test dataset in the given root and applying the given transforms
        trainset = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False,  download=True, transform=transform)

        batch_size=4

        # dataloaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,  shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,  shuffle=False, num_workers=2) 

        return trainloader, testloader