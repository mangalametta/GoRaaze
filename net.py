from tkinter import NE
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module): #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
        kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
        kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,use_1x1conv=True, strides=1))
    else:
        blk.append(Residual(num_channels, num_channels))
    return blk

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(4, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256), nn.ReLU())
        b2 = nn.Sequential(*resnet_block(256, 256, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(256, 256, 2))
        b4 = nn.Sequential(*resnet_block(256, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 256, 2))
        b6 = nn.Sequential(*resnet_block(256, 256, 2))
        b7 = nn.Sequential(*resnet_block(256, 256, 2))
        b8 = nn.Sequential(*resnet_block(256, 256, 2))
        b9 = nn.Sequential(*resnet_block(256, 256, 2))
        b10 = nn.Sequential(*resnet_block(256, 256, 2))
        self.final = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, stride=1),nn.BatchNorm2d(1), nn.ReLU())
        self.flatting = nn.Sequential(nn.Conv2d(256, 4, kernel_size=1, stride=1),nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten())
        self.net = nn.Sequential(b1, b2, b3, b4, b5, b6, b7, b8, b9 ,b10)
        self.fc1 = nn.Linear(1445,1000)
        self.fc2 = nn.Linear(1000,1)

    def forward(self, x, c):
        # first conv
        
        x = self.net(x)
        y = self.final(x)
        # fully connected layers
        w = self.flatting(x)
        w = torch.cat((w,c),dim=1)
        w = F.relu(self.fc1(w))
        w = self.fc2(w)
        return y, w


if __name__ == "__main__":
    X = torch.rand(size=(1, 4, 19,19))
    C = torch.rand(size=(1, 1))
    net = Network()
    o = net.forward(X,C)
    print(o[1].shape)