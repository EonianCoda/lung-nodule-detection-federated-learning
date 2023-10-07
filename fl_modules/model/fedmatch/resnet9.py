import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels: int, 
               out_channels: int):
    
    
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.ReLU()
    ]
    # if pool:  
    #     layers.append(nn.MaxPool2d(kernel_size=pool_size))
    return nn.Sequential(*layers)

# def conv1x1(in_planes: int, out_planes: int, strides = 1, groups = 1, dilation = 1, bias=  False):
#     return nn.Conv3d(in_planes, out_planes, kernel_size = 1, stride = strides, padding = 0, groups = groups, bias = bias, dilation = dilation)

# def conv3x3(in_planes: int, out_planes: int, strides = 1, groups = 1, dilation = 1, bias=  False):
#     return nn.Conv3d(in_planes, out_planes, kernel_size = 3, stride = strides, padding = 1, groups = groups, bias = bias, dilation = dilation)

class ResNet9(nn.Module):
    def __init__(self, 
                 in_channels: int = 1, 
                 num_classes: int = 10):
        super(ResNet9, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.psi_factor = psi_factor


        # self.stem = nn.Sequential(
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True, pool_size=2)
        self.conv3 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv4 = conv_block(128, 256, pool=True)
        self.conv5 = conv_block(256, 512, pool=True, pool_size=2)
        self.conv6 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.pool = nn.MaxPool2d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes, bias=False)

    # def _make_layer(self, in_planes: int, out_planes: int, blocks: int, stride: int = 1):
    #     layers = [ResBlock3D(in_planes, out_planes, stride, normalization='batch', bias=False, attn_block=self.attn_block)]
    #     for i in range(1, blocks):
    #         layers.append(ResBlock3D(out_planes, out_planes, normalization='batch', bias=False, attn_block=self.attn_block))

    #     return nn.Sequential(*layers)

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
        return out

# Define the model
input_shape = (3, 224, 224)  # Assuming RGB images of size 224x224
num_classes = 10  # Adjust accordingly
psi_factor = 0.1  # Adjust accordingly

model = ResNet9(input_shape, num_classes, psi_factor)

# Adjust weights
with torch.no_grad():
    for param in model.parameters():
        param.mul_(1 + psi_factor)
