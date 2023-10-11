import torch.nn as nn
from torch.nn import functional as F

def conv_block(in_channels: int, 
               out_channels: int,
                pool: bool = False,
                pool_size: int = 2):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.ReLU()
    ]
    if pool:  
        layers.append(nn.MaxPool2d(kernel_size=pool_size))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, 
                 in_channels: int = 1, 
                 num_classes: int = 10):
        super(ResNet9, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True, pool_size=2)
        self.conv3 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv4 = conv_block(128, 256, pool=True)
        self.conv5 = conv_block(256, 512, pool=True, pool_size=2)
        self.conv6 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.pool = nn.MaxPool2d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.conv3(out)
        out = out + residual
        
        out = self.conv4(out)
        out = self.conv5(out)
        residual = self.conv6(out)
        out = out + residual
        
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out