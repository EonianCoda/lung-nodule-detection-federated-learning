import torch
import torch.nn as nn
from torch.nn import functional as F
from .modules import ConvBlock3D, ResBlock3D, AttentionGate

class DecoderBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, normalization = 'instance'):
        super(DecoderBlock, self).__init__()

        self.attention_gate = AttentionGate(in_planes, out_planes, pool_size = 3)
        self.upsample = ConvBlock3D(in_planes, out_planes, normalization=normalization)
        self.merge = nn.Sequential(
            ConvBlock3D(out_planes * 2, out_planes, normalization=normalization),
            ConvBlock3D(out_planes, out_planes, normalization=normalization),
        )

    def forward(self, x, g):
        att = self.attention_gate(x, g)
        upsampled = self.upsample(F.interpolate(g, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False))
        merged = torch.cat((att, upsampled), dim=1)
        output = self.merge(merged)
        return output

class Stage1Model(nn.Module):
    def __init__(self, base_planes = 8, normalization = 'instance'):
        super(Stage1Model, self).__init__()
        
        self.encoder1 = self._make_encoder_block(1, base_planes)
        self.encoder2 = self._make_encoder_block(base_planes, base_planes * 2)
        self.encoder3 = self._make_encoder_block(base_planes * 2, base_planes * 4)
        
        self.bridge = nn.Sequential(
                        ConvBlock3D(base_planes * 4, base_planes * 8, normalization=normalization),
                        ConvBlock3D(base_planes * 8, base_planes * 8, normalization=normalization),)
        
        self.decoder3 = self._make_decoder_block(base_planes * 8, base_planes * 4)
        self.decoder2 = self._make_decoder_block(base_planes * 4, base_planes * 2)
        self.decoder1 = self._make_decoder_block(base_planes * 2, base_planes)

        self.output256 = nn.Conv3d(base_planes * 2, 1, kernel_size=1)
        self.output512 = nn.Conv3d(base_planes, 1, kernel_size=1)

        self.maxpool = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
        self.sigmoid = nn.Sigmoid()
        
        self._weight_init()
        
    def _make_encoder_block(self, in_planes: int, out_planes: int, normalization = 'instance'):
        return ResBlock3D(in_planes, out_planes, normalization=normalization)

    def _make_decoder_block(self, in_planes: int, out_planes: int, normalization = 'instance'):
        return DecoderBlock(in_planes, out_planes, normalization=normalization)

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        
        # Bridge
        enc4 = self.bridge(self.maxpool(enc3))
        
        # Decoder
        dec3 = self.decoder3(enc3, enc4)
        dec2 = self.decoder2(enc2, dec3)
        dec1 = self.decoder1(enc1, dec2)

        # Output
        output256 = self.sigmoid(self.output256(dec2))
        output512 = self.sigmoid(self.output512(dec1))

        return output512, output256