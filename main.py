#! /usr/bin/python3

import torch

def launchSeed(isRand=True):
    if isRand:
        x = torch.rand(1) * 1000
    else:
        x = 1234
    torch.manual_seed(x)

def main():
    launchSeed(False)
    x = torch.rand(5, 3)
    tensorZeros = torch.zeros((5, 3), dtype=torch.int64)
    print(tensorZeros)
    return x

if __name__ == "__main__":
    tensor = main()
    print(tensor)


