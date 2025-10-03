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

class FCN(nn.Module):
    """
    Fully Convolutional Network (FCN) for image segmentation.
    """
    def __init__(self,  in_ch=3, out_ch=1, backbone='resnet50', pretrained=False):
        super(FCN, self).__init__()
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone. Choose 'resnet50' or 'resnet101'.")

        # Remove the fully connected layer and the last two layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # FCN head
        self.conv6 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout2d(p=0.1, inplace=False)

        self.conv7 = nn.Conv2d(512, out_ch, kernel_size=1, bias=False)

        # Score pooling layers
        self.score_pool4 = nn.Conv2d(1024, out_ch, kernel_size=1, bias=False)
        self.score_pool3 = nn.Conv2d(512, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        input_size = x.size()[2:]

        # Backbone feature extraction
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # bn1
        x = self.backbone[2](x)  # relu
        x = self.backbone[3](x)  # maxpool

        x = self.backbone[4](x)  # layer1
        x = self.backbone[5](x)  # layer2
        pool3 = x
        x = self.backbone[6](x)  # layer3
        pool4 = x
        x = self.backbone[7](x)  # layer4

        # FCN head
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.conv7(x)

        # Score pooling layers
        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        # Upsampling and score fusion
        x = F.interpolate(x, size=pool4.size()[2:], mode='bilinear', align_corners=True)
        x = x + score_pool4
        x = F.interpolate(x, size=pool3.size()[2:], mode='bilinear', align_corners=True)
        x = x + score_pool3
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x

# Example usage
if __name__ == "__main__":
    model = FCN( in_ch=3, out_ch=1, backbone='resnet50', pretrained=False)

    input_tensor = torch.randn(1, 3, 256, 256)

    output = model(input_tensor)

    print("Output shape:", output.shape)
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
