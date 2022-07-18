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
            blk.append(Residual(input_channels, num_channels,use_1x1conv=True, strides=2))
    else:
        blk.append(Residual(num_channels, num_channels))
    return blk

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=3),nn.BatchNorm2d(64), nn.ReLU())
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(b1, b2, b3, b4, b5,nn.Flatten())
        self.fc1 = nn.Linear(4609, 1000)
        self.fc2 = nn.Linear(1000, 362)

    def forward(self, x):
        # only the input needs to be reshape here
        c = x[...,-1]
        x = torch.reshape(x[...,:-1],x.shape[:-1]+(19,19))
        # first conv
        x = self.net(x)
        # fully connected layers
        x = torch.cat((x, c), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    X = torch.rand(size=(512, 1, 362))
    net = Network()
    print(net.forward(X).shape)