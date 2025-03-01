#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F

def launchSeed(isRand=True):
    if isRand:
        x = torch.rand(1) * 1000
    else:
        x = 1234
    torch.manual_seed(x)

def creationTensor():
    x = torch.randn(1, 10)
    prevH = torch.randn(1, 20)
    WH = torch.randn(20, 20)
    WX = torch.randn(20, 10)
    return x, prevH, WH, WX

def ADE():
    x, prevH, WH, WX = creationTensor()
    i2h = torch.mm(WX, x.t())
    h2h = torch.mm(WH, prevH.t())
    nextH = i2h + h2h
    nextH = nextH.tanh()
    loss = nextH.sum()
    return loss

def main():
    launchSeed(False)
    loss = ADE()
    print(loss)

if __name__ == "__main__":
    tensor = main()


