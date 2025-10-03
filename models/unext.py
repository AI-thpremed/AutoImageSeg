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

"""
UNeXt: MLP-based lightweight U-Net (single-file, PyTorch)
Author  : your_name
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile, clever_format

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )


class MLPBlock(nn.Module):
    """MLP-Mixer block for UNext"""
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        # x: (B,C,H,W) -> (B,H*W,C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B,HW,C)
        x = x + self.mlp1(self.norm1(x))
        x = x + self.mlp2(self.norm2(x))
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class UNext(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, base_dim=32):
        super().__init__()
        # Encoder
        self.enc = nn.ModuleList([
            ConvBNReLU(in_ch, base_dim),
            ConvBNReLU(base_dim, base_dim * 2),
            ConvBNReLU(base_dim * 2, base_dim * 4),
            ConvBNReLU(base_dim * 4, base_dim * 8),
        ])
        # Bottleneck MLP
        self.bottleneck = MLPBlock(base_dim * 8)
        # Decoder
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(base_dim * 8, base_dim * 4, 2, 2),
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, 2),
            nn.ConvTranspose2d(base_dim * 2, base_dim, 2, 2),
            nn.ConvTranspose2d(base_dim, out_ch, 2, 2),
        ])

    def forward(self, x):
        feats = []
        h, w = x.shape[-2:]
        # Encoder
        for layer in self.enc:
            x = F.max_pool2d(layer(x), 2)
            feats.append(x)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder
        for i, up in enumerate(self.up):
            x = up(x)
            if i < 3:
                x = x + F.interpolate(feats[-i - 2], size=x.shape[-2:], mode='nearest')
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)



if __name__ == "__main__":
    model = UNext(in_ch=3, out_ch=1)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print("Output shape:", output.shape)
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
