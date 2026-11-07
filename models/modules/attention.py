import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    Captures local cross-channel interaction without dimensionality reduction using an adaptive 1D convolution.
    """
    def __init__(self, channel: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 != 0 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module (FFM) with Efficient Channel Attention.
    Aligns and fuses spatial detail features from LTD with semantic context from GFE.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.eca = ECALayer(out_c)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, high_res: torch.Tensor, low_res: torch.Tensor) -> torch.Tensor:
        low_res_up = F.interpolate(low_res, size=high_res.shape[2:], mode='bilinear', align_corners=True)
        cat_feat = torch.cat([high_res, low_res_up], dim=1)
        out = self.conv2(self.conv1(cat_feat))
        out = self.eca(out)
        return out * self.attn(self.pool(out))
