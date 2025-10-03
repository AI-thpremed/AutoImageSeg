# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from thop import profile, clever_format

class DecoderBlock(nn.Module):
    """
    Decoder Block for LinkNet
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class LinkNet(nn.Module):
    """
    LinkNet model for image segmentation
    """
    def __init__(self,  in_ch=3, out_ch=1, backbone='resnet18', pretrained=False):
        super(LinkNet, self).__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone. Choose 'resnet18', 'resnet34', or 'resnet50'.")

        self.encoder1 = nn.Sequential(*list(self.backbone.children())[:4])
        self.encoder2 = self.backbone.layer1
        self.encoder3 = self.backbone.layer2
        self.encoder4 = self.backbone.layer3
        self.encoder5 = self.backbone.layer4

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Decoder
        d4 = self.decoder4(e5)
        d4 = F.interpolate(d4, size=e4.size()[2:], mode='bilinear', align_corners=True)
        d4 = d4 + e4

        d3 = self.decoder3(d4)
        d3 = F.interpolate(d3, size=e3.size()[2:], mode='bilinear', align_corners=True)
        d3 = d3 + e3

        d2 = self.decoder2(d3)
        d2 = F.interpolate(d2, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d2 = d2 + e2

        d1 = self.decoder1(d2)
        d1 = F.interpolate(d1, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d1 = d1 + e1

        # Final Convolution
        out = self.final_conv(d1)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out

# Example usage
if __name__ == "__main__":
    model = LinkNet( in_ch=3, out_ch=1, backbone='resnet18', pretrained=False)

    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)

    print("Output shape:", output.shape)
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
