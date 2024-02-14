import torch
import torch.nn as nn
from .modules import CBAM, ConvBlock

class GFE(nn.Module):
    def __init__(self, in_channels, n_filters=[16, 32, 48, 64]):
        super(GFE, self).__init__()
        
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=n_filters[0], kernel_size=3, stride=1)
        self.conv2 = ConvBlock(in_channels=n_filters[0] + 2, out_channels=n_filters[1], kernel_size=3, stride=1)
        self.conv3 = ConvBlock(in_channels=n_filters[1] + 3, out_channels=n_filters[2], kernel_size=3, stride=1)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.last_conv = ConvBlock(in_channels=n_filters[2], out_channels=n_filters[3], kernel_size=3, stride=1)
        self.cbam = CBAM(in_channels=n_filters[3])

    def forward(self, x1, x2, x3):
        feat1 = self.conv1(x1)
        feat2 = self.conv2(torch.cat([feat1, x2, x1], dim=1))
        feat3 = self.conv3(torch.cat([feat2, x3, x2, x1], dim=1))
        
        feat_pool = self.pool(feat3)
        feat_pool = self.last_conv(feat_pool)
        
        feat_refine = self.cbam(feat_pool)

        return feat_refine

class Stage2Model(nn.Module):
    def __init__(self, in_channels = 1, n_filters=[16, 32, 48, 64, 128]):
        super(Stage2Model, self).__init__()
        self.gfe1 = GFE(in_channels, n_filters[:4])
        self.gfe2 = GFE(in_channels, n_filters[:4])
        
        self.conv = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2),
                                  ConvBlock(in_channels=n_filters[3], out_channels=n_filters[4], kernel_size=3, stride=1),
                                  CBAM(in_channels=n_filters[4]))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), 
                                        nn.Flatten(), 
                                        nn.Linear(n_filters[4], 1), 
                                        nn.Sigmoid())
        
        self._weight_init()
        
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1, x2, x3 = x[0], x[1], x[2]
        zoom_in_stream = self.gfe1(x1, x2, x3)
        zoom_out_stream = self.gfe2(x3, x2, x1)
        feat_combine = self.conv(zoom_in_stream + zoom_out_stream) 
        output = self.classifier(feat_combine)

        return output