import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelisationNet.model import Net
from commandLine import parse_arguments

def initdata(isOwnData=False):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if isOwnData else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    if isOwnData:
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
        
        classes = tuple(str(i) for i in range(10))
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    
    return trainloader, testloader, classes

def imshow(img, classes, labels, isGrey = False):
    img = img / 2 + 0.5
    npimg = img.numpy()
    print(' '.join(classes[labels[j]] for j in range(4)))
    if isGrey:
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def loopDataSet(trainloader, optimizer, net, criterion, device):
    net.to(device)
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch+1}, {i+1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

def accuracy(testloader, net, device):
    net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

def main():
    args = parse_arguments()

    isOwnData = args.c
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader, classes = initdata(isOwnData)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images), classes, labels, isGrey=isOwnData)

    net = Net(len(classes), isOwnData).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loopDataSet(trainloader, optimizer, net, criterion, device)

    accuracy(testloader, net, device)

    if args.s:
        outputFile = "src/model/model-" + "image" if not args.c else "chiffre" + ".pth"
        torch.save(net.state_dict(), outputFile)
        print(f"Modèle sauvegardé sous '{outputFile}'")


if __name__ == "__main__":
    main()