import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes: int, out_planes: int, strides = 1, groups = 1, dilation = 1, bias=  False):
    return nn.Conv3d(in_planes, out_planes, kernel_size = 1, stride = strides, padding = 0, groups = groups, bias = bias, dilation = dilation)

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
    
class AttentionGate(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, pool_size = 2):
        super(AttentionGate, self).__init__()
        self.theta_x = conv1x1(out_planes, out_planes, bias = True)
        self.phi_g = conv1x1(in_planes, out_planes, bias = True)
        self.psi_f = conv1x1(out_planes, 1, bias = True)
        self.pool_size = pool_size
                
    def forward(self, x, g):
        theta_x = self.theta_x(x)
        theta_x = F.max_pool3d(theta_x, kernel_size = self.pool_size, stride = 2, padding = (1 if self.pool_size == 3 else 0))

        phi_g = self.phi_g(g)
        f = F.relu(theta_x + phi_g)

        psi_f = self.psi_f(f)
        rate = torch.sigmoid(psi_f)
        rate = F.interpolate(rate, scale_factor=(2, 2, 2), mode='trilinear')

        att_x = x * rate
        return att_x