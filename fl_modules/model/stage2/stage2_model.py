import torch
import torch.nn as nn

from .modules import ConvBlock3D, ResBlock3D, EcaLayer

class Stage2Model(nn.Module):
    def __init__(self, base_planes = 16, attn_block = EcaLayer, num_blocks=[2, 2, 2, 2]) -> None:
        super(Stage2Model, self).__init__()

        self.attn_block = attn_block
        self.conv1 = ConvBlock3D(1, base_planes, normalization='batch', bias=False)
        self.conv2 = self._make_layer(base_planes, base_planes * 2, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(base_planes * 2, base_planes * 4, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(base_planes * 4, base_planes * 8, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(base_planes * 8, base_planes * 16, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base_planes * 16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self, in_planes: int, out_planes: int, blocks: int, stride: int = 1):
        layers = [ResBlock3D(in_planes, out_planes, stride, normalization='batch', bias=False, attn_block=self.attn_block)]
        for i in range(1, blocks):
            layers.append(ResBlock3D(out_planes, out_planes, normalization='batch', bias=False, attn_block=self.attn_block))

        return nn.Sequential(*layers)
    
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x