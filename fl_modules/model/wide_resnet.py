import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        self.droprate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, 
                 num_of_layers: int, 
                 in_planes: int, 
                 out_planes: int, 
                 block: nn.Module, 
                 stride: int, 
                 drop_rate = 0.0):
        super(NetworkBlock, self).__init__()
        self.layers = self._make_layer(block, in_planes, out_planes, num_of_layers, stride, drop_rate)
    
    def _make_layer(self, block: nn.Module, in_planes: int, out_planes: int, num_of_layers: int, stride: int, drop_rate: float):
        layers = []
        num_of_layers = int(num_of_layers)
        for i in range(num_of_layers):
            if i == 0:
                layers.append(block(in_planes, out_planes, stride, drop_rate))
            else:
                layers.append(block(out_planes, out_planes, 1, drop_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class WideResNet(nn.Module):
    def __init__(self, 
                 depth: int = 28, 
                 widen_factor = 2,
                 num_classes: int = 10, 
                 drop_rate: float = 0.0,
                 block = BasicBlock):
        """
        Args:
            depth: depth of the network. e.g. 28 for WRN-28-10
            widen_factor: widen factor. Default is 1, e.g. 10 for WRN-28-10
            num_classes: number of classes
            drop_rate: dropout rate. Default is 0
        """
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6 # number of layers in each network block
        
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, 
                               n_channels[0], 
                               kernel_size=3, 
                               stride=1,
                               padding=1, 
                               bias=False)
       
        self.block1 = self._make_layer(block, n_channels[0], n_channels[1], n, 1, drop_rate)
        self.block2 = self._make_layer(block, n_channels[1], n_channels[2], n, 2, drop_rate)
        self.block3 = self._make_layer(block, n_channels[2], n_channels[3], n, 2, drop_rate)
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[3], num_classes)
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def _make_layer(self, block: nn.Module, in_planes: int, out_planes: int, num_of_layers: int, stride: int, drop_rate: float):
        layers = []
        num_of_layers = int(num_of_layers)
        for i in range(num_of_layers):
            if i == 0:
                layers.append(block(in_planes, out_planes, stride, drop_rate))
            else:
                layers.append(block(out_planes, out_planes, 1, drop_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        
        out = self.gap(out)
        out = nn.Flatten()(out)
        return F.softmax(self.fc(out), dim=1)