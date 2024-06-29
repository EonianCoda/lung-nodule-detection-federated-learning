import torch.nn as nn

class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu', mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1,
                 norm_type='none', act_type='ReLU'):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,
                              padding=kernel_size // 2 + dilation - 1, dilation=dilation, bias=False)
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = act_layer(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

def act_layer(act='ReLU'):
    if act == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act == 'LeakyReLU':
        return nn.LeakyReLU(inplace=True, negative_slope=0.1)
    elif act == 'ELU':
        return nn.ELU(inplace=True)
    elif act == 'PReLU':
        return nn.PReLU(inplace=True)
    elif act == 'RRelu':
        return nn.RReLU(inplace=True)
    elif act == 'Silu':
        return nn.SiLU(inplace=True)
    else:
        return Identity()

def norm_layer3d(norm_type, num_features):
    if norm_type == 'batchnorm':
        return nn.BatchNorm3d(num_features=num_features, momentum=0.05)
    elif norm_type == 'instancenorm':
        return nn.InstanceNorm3d(num_features=num_features, affine=True)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(num_groups=num_features // 8, num_channels=num_features)
    else:
        return Identity()