# train the network using SGD but without using all the training data at once
from time import sleep
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim

def train(model, trainloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu") ):
    model.to(device)
    inspected = 50000
    iterations = int(inspected / 4)
    

    dataloader_iterator = iter(trainloader)

    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in tqdm(range(iterations), desc="training"):
        try:
            inputs, labels = next(dataloader_iterator)
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
def eval(model, testloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu") ):
  
  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in tqdm(testloader, desc="evaluating"):
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