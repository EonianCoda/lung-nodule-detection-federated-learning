import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes: int, out_planes: int, strides = 1, groups = 1, dilation = 1, bias=  False):
    return nn.Conv3d(in_planes, out_planes, kernel_size = 1, stride = strides, padding = 0, groups = groups, bias = bias, dilation = dilation)

class EcaLayer(nn.Module):
    """Constructs a ECA module.

    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size:int = 3, shortcut = False):
        super(EcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding = 'same', bias=False) 
        self.sigmoid = nn.Sigmoid()

        self.shortcut = shortcut
    def forward(self, x):
        y = self.avg_pool(x) # (N, C, 1, 1, 1)
        y = torch.flatten(y, 1).unsqueeze(-1) # (N, C, 1)
        y = y.transpose(-1, -2) # (N, 1, C)
        y = self.conv(y) # (N, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1) # (N, C, 1, 1, 1)
        
        y = self.sigmoid(y)
        if self.shortcut:
            return x * y + x
        else:
            return x * y

class ConvBlock3D(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size = 3, stride = 1, padding = 'same', activation = True, normalization = 'instance', bias = True):
        super(ConvBlock3D, self).__init__()
        if stride != 1 and padding == 'same':
            padding = (kernel_size - 1) // 2
        self.conv3d = nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, bias = bias)
        self.activation = nn.ReLU() if activation else None
        self.normalization = normalization

        if self.normalization == 'instance':
            self.norm_layer = nn.InstanceNorm3d(out_planes)
        elif self.normalization == 'instance_affine':
            self.norm_layer = nn.InstanceNorm3d(out_planes, affine = True)
        elif self.normalization == 'batch':
            self.norm_layer = nn.BatchNorm3d(out_planes)
        elif 'group' in self.normalization:
            num_groups = int(self.normalization.split('_')[-1])
            self.norm_layer = nn.GroupNorm(num_groups, out_planes)
        else:
            raise ValueError('Invalid normalization type {}'.format(self.normalization))
        
    def forward(self, x):
        x = self.conv3d(x)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        if self.activation:
            x = self.activation(x)

        return x

class ResBlock3D(nn.Module):
    def __init__(self, 
                 in_planes: int, 
                 out_planes: int, 
                 strides = 1, 
                 normalization = 'instance',
                 bias = True, 
                 attn_block = None):
        """
        Args:
            normlization: 'instance' or 'batch'
        """
        super(ResBlock3D, self).__init__()
        self.conv1 = ConvBlock3D(in_planes, out_planes, stride = strides, normalization = normalization, bias = bias)
        self.conv2 = ConvBlock3D(out_planes, out_planes, normalization = normalization, bias = bias)
        self.strides = strides

        if attn_block is not None:
            self.attn_block = attn_block()
        else:
            self.attn_block = None
        if strides != 1 or in_planes != out_planes:
            self.identity = conv1x1(in_planes, out_planes, strides = strides, bias=True)
        else:
            self.identity = None

    def forward(self, x):
        identity = x

        conv = self.conv1(x)
        conv = self.conv2(conv)

        # Attention block
        if self.attn_block is not None:
            conv = self.attn_block(conv)
            
        # Shortcut connection
        if self.identity is not None:
            identity = self.identity(identity)
            conv = conv + identity
            
        return conv