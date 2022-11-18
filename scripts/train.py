from time import sleep
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim

from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from scripts.dataloader import get_dataset
import numpy as np

DEBUG = 0

def train(model, dataset_fun, trainloader, batch_size = 4, epochs = 1, all = False):
    '''
    model: the model to train
    trainloader: the dataloader for the training data
    batch_size: the batch size used to construct the trainloader
    epochs: the number of epochs to train the model
    inspect: the number of items to be used for training before printing the loss
    '''
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # the device type is automatically chosen

    model.to(device)
    
    if not all:
        dataset = get_dataset(dataset_fun)
        # construct a smaller dataloader for the training
        inspected = int(len(trainloader.dataset) / 100)
        init = np.random.randint(0, len(trainloader.dataset) - inspected)
        dataset.data = dataset.data[init:init+inspected]
        dataset.targets = dataset.targets[init:init+inspected]
        trainloader = DataLoader(
                        dataset,
                        batch_sampler= BatchSampler(RandomSampler(dataset), batch_size, False),
                        num_workers=2,
                        pin_memory=True
                    )

    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for _ in range(epochs):  # loop over the dataset multiple times

        #dataloader_iterator = iter(trainloader) # instantiate an iterator which loops through the trainloader, this is needed only if we do not wnat to go throught all the trainset

        # for i in tqdm(range(iterations), desc=f"training epoch:{epoch}"):
        for data in trainloader: #range(iterations):
            try:
                inputs, labels = data #next(dataloader_iterator)
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # calculate outputs by running images through the network
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            except StopIteration:
                print("StopIteration, not enough data")
            
    return model

'''
This is a simple function to check the model is properly built and correctly working  
'''
def test_model(model, trainloader):
    '''
    model: the model to train
    trainloader: the dataloader for the training data
    '''
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # the device type is automatically chosen
    model.to(device)
    dataloader_iterator = iter(trainloader) # instantiate an iterator which loops through the trainloader, this is needed only if we do not wnat to go throught all the trainset
    
    try:
        inputs, labels = next(dataloader_iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # run inputs through the network to see if everything works
        model(inputs)
        
        

    except Exception as e:
        print("This network will be discarded as some measures are incorrect. In particular:\n", e)
        return False
            
    return True

    
def eval(model, testloader):
    '''
    model: the model to evaluate
    testloader: the dataloader for the test data
    '''
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # the device type is automatically chosen
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        # for data in tqdm(testloader, desc="evaluating"):
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model.forward(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct // total
    print(f'Accuracy of the network on the 10000 test images: {accuracy} %')
    return accuracy


