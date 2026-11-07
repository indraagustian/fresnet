import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsamplerBlock(nn.Module):
    """
    Transposed convolution upsampling block with batch normalization and ReLU activation.
    """
    def __init__(self, ninput: int, noutput: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))

class DecoderNBB(nn.Module):
    """
    Cascaded 1D Non-Bottleneck block for decoder feature refinement with residual skip connection.
    """
    def __init__(self, chann: int, dropprob: float = 0.0):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-3)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-3)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv3x1_1(x)
        out = F.relu(out)
        out = self.conv1x3_1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv3x1_2(out)
        out = F.relu(out)
        out = self.conv1x3_2(out)
        out = self.bn2(out)
        if self.dropout.p > 0:
            out = self.dropout(out)
        return F.relu(out + x)

class ERFNetStyleDecoder(nn.Module):
    """
    Hierarchical factorized decoder reconstructing full spatial resolution across three successive stages.
    """
    def __init__(self, in_channels: int = 128, num_classes: int = 3):
        super().__init__()
        # Stage 1: 1/8 -> 1/4
        self.up1 = UpsamplerBlock(in_channels, 64)
        self.nbb1 = DecoderNBB(64, 0.0)
        self.nbb2 = DecoderNBB(64, 0.0)

        # Stage 2: 1/4 -> 1/2
        self.up2 = UpsamplerBlock(64, 16)
        self.nbb3 = DecoderNBB(16, 0.0)
        self.nbb4 = DecoderNBB(16, 0.0)

        # Stage 3: 1/2 -> 1/1 + Classification
        self.seg_head = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nbb2(self.nbb1(self.up1(x)))
        x = self.nbb4(self.nbb3(self.up2(x)))
        return self.seg_head(x)
