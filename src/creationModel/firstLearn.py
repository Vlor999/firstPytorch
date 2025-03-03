import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelisationNet.model import Net
from commandLine import parse_arguments

def affiche(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

class AugmentedMNIST(Dataset):
    def __init__(self, transform, train=True):
        self.dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    
    def __len__(self):
        return len(self.dataset) * 2
    
    def __getitem__(self, index):
        image, label = self.dataset[index % len(self.dataset)]
        
        if index >= len(self.dataset):
            image = 1 - image  
        
        return image, label

def initdata(isOwnData=False):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10), 
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.Normalize((0.5,), (0.5,)) if isOwnData else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    if isOwnData:
        trainset = AugmentedMNIST(transform, train=True)
        testset = AugmentedMNIST(transform, train=False)

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

def imshow(img, classes, labels, isGrey=False):
    img = img / 2 + 0.5 
    npimg = img.numpy()
    
    print(' '.join(classes[labels[j]] for j in range(max(4, len(labels)))))
    
    if isGrey:
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    plt.axis('off')
    plt.show()

def loopDataSet(trainloader, optimizer, net, criterion, device, scheduler):
    for epoch in range(10):  
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

            if i % 1000 == 999:
                print(f'[{epoch+1}, {i+1}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0

        scheduler.step()

        print(f'Learning rate après epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}')  
    print("Entrainement fini")


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
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    loopDataSet(trainloader, optimizer, net, criterion, device, scheduler)

    accuracy(testloader, net, device)

    if args.s:
        outputFile = f"src/model/model-{'chiffre' if args.c else 'image'}.pth"
        torch.save(net.state_dict(), outputFile)
        print(f"Modèle sauvegardé sous '{outputFile}'")

if __name__ == "__main__":
    main()
