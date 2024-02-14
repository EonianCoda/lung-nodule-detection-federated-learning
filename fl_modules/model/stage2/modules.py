import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,
                              padding=kernel_size // 2 + dilation - 1, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()
        self.mlp_layer1 = nn.Sequential(nn.Linear(in_channels, in_channels // ratio), 
                                        nn.ReLU())
        
        self.mlp_layer2 = nn.Sequential(nn.Linear(in_channels // ratio, in_channels),
                                        nn.ReLU())
                                        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_feat = self.flatten(self.avg_pool(x))
        avg_feat = self.mlp_layer1(avg_feat)
        avg_feat = self.mlp_layer2(avg_feat) 
        
        max_feat = self.flatten(self.max_pool(x))
        max_feat = self.mlp_layer1(max_feat)
        max_feat = self.mlp_layer2(max_feat)
        
        feat_combined = avg_feat + max_feat
        feat_combined = self.sigmoid(feat_combined)
        feat_combined = feat_combined.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return feat_combined * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feature = torch.mean(x, dim=1, keepdim=True)
        max_feature, _ = torch.max(x, dim=1, keepdim=True)

        feature_combined = torch.cat([avg_feature, max_feature], dim=1)
        feature_combined = self.conv(feature_combined)
        feature_combined = self.sigmoid(feature_combined)

        return x * feature_combined

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        channel_feat_refined = self.channel_attention(x)
        final_feat_refined = self.spatial_attention(channel_feat_refined)

        return final_feat_refined