import torch
import torch.nn as nn

class NonBottleneck1D(nn.Module):
    """
    Factorized 1D Non-Bottleneck Residual Block.
    Decomposes standard 2D 3x3 convolution into sequential (3x1) and (1x3) 1D convolutions,
    substantially reducing parameter count and computational complexity while maintaining receptive field.
    """
    def __init__(self, channels: int, drop_prob: float = 0.0, dilation: int = 1):
        super().__init__()
        self.conv3x1 = nn.Conv2d(channels, channels, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3 = nn.Conv2d(channels, channels, (1, 3), stride=1, 
                                 padding=(0, dilation) if dilation > 1 else (0, 1), 
                                 bias=True, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv3x1(x)
        out = self.relu(self.bn1(out))
        out = self.conv1x3(out)
        out = self.bn2(out)
        out = self.dropout(self.relu(out))
        return out + x
