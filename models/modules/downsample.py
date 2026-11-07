import torch
import torch.nn as nn

class LearningToDownsample(nn.Module):
    """
    Learning to Downsample (LTD) module.
    Rapidly reduces spatial dimensions by 8x using a 3-stage hierarchy of standard convolution 
    and depthwise separable convolutions, preserving high-frequency boundary features.
    """
    def __init__(self, in_c: int = 3, dw1: int = 32, dw2: int = 48, out_c: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, dw1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dw1),
            nn.ReLU(inplace=True)
        )
        self.ds1 = nn.Sequential(
            nn.Conv2d(dw1, dw1, kernel_size=3, stride=2, padding=1, groups=dw1, bias=False),
            nn.Conv2d(dw1, dw2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dw2),
            nn.ReLU(inplace=True)
        )
        self.ds2 = nn.Sequential(
            nn.Conv2d(dw2, dw2, kernel_size=3, stride=2, padding=1, groups=dw2, bias=False),
            nn.Conv2d(dw2, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.ds1(x)
        x = self.ds2(x)
        return x
