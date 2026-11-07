import math
from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.factorized_res import NonBottleneck1D
from .modules.downsample import LearningToDownsample
from .modules.attention import FeatureFusionModule
from .modules.decoders import ERFNetStyleDecoder

def _scale_channel(ch: int, width_mult: float, min_ch: int = 8, div: int = 8) -> int:
    """Scale channel count with rounding to multiples of div."""
    return max(min_ch, int(round(ch * width_mult / div) * div))

class GlobalFeatureExtractor(nn.Module):
    """
    Global Feature Extractor (GFE) with multi-rate dilated 1D factorized convolutions.
    """
    def __init__(self, in_c: int, block_c: int, num_blocks: int = 3, dilations: list = None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]
        self.transition = nn.Sequential(
            nn.Conv2d(in_c, block_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(block_c),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            d = dilations[i] if i < len(dilations) else 1
            self.blocks.add_module(f'fres_block_{i}', NonBottleneck1D(block_c, drop_prob=0.1, dilation=d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.transition(x))

class FRESNet(nn.Module):
    """
    Fast Residual Efficient Segmentation Network (FRES-Net).
    
    Arhitecture:
      - LearningToDownsample (LTD): Rapid 8x downsampling preserving spatial edges
      - GlobalFeatureExtractor (GFE): Multi-rate dilated factorized 1D non-bottleneck residual blocks
      - FeatureFusionModule (FFM): Cross-scale fusion with Efficient Channel Attention (ECA)
      - ERFNetStyleDecoder: Factorized transposed convolution decoder
      
    Args:
      num_classes (int): Number of target semantic segmentation classes (default: 3: Background, Crop, Weed)
      in_channels (int): Input image channels (default: 3 for RGB)
      width_mult (float): Capacity scaling width multiplier (e.g., 0.33, 0.50, 1.00, 2.00, 3.00)
    """
    def __init__(self, num_classes: int = 3, in_channels: int = 3, width_mult: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.width_mult = width_mult

        # Scaled channel dimensions
        d1 = _scale_channel(32, width_mult)
        d2 = _scale_channel(48, width_mult)
        d3 = _scale_channel(64, width_mult)
        gb = _scale_channel(96, width_mult)
        ff = _scale_channel(128, width_mult)

        self.ltd = LearningToDownsample(in_channels, d1, d2, d3)
        self.gfe = GlobalFeatureExtractor(d3, gb, num_blocks=3, dilations=[1, 2, 4])
        self.ffm = FeatureFusionModule(d3 + gb, ff)
        self.decoder = ERFNetStyleDecoder(in_channels=ff, num_classes=num_classes)

        # Feature projection layer for contrastive feature alignment when scaling width
        if ff != 128:
            self.proj_feat = nn.Conv2d(ff, 128, kernel_size=1, bias=False)
        else:
            self.proj_feat = nn.Identity()

    def forward(self, x: torch.Tensor, return_features: bool = False):
        low = self.ltd(x)
        glob = self.gfe(low)
        fused = self.ffm(low, glob)
        out = self.decoder(fused)
        
        if return_features:
            fused_feat = self.proj_feat(fused)
            features_up = F.interpolate(fused_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
            return out, features_up
        return out

def build_fresnet(variant: str = 'fresnet_w100', num_classes: int = 3, pretrained: Optional[str] = None) -> FRESNet:
    """
    Factory helper to instantiate FRES-Net variants.
    Variants: 'fresnet_w033', 'fresnet_w050', 'fresnet_w100' (default), 'fresnet_w200', 'fresnet_w300'
    """
    width_map = {
        'fresnet_w033': 0.33,
        'fresnet_w050': 0.50,
        'fresnet_w100': 1.00,
        'fresnet_w200': 2.00,
        'fresnet_w300': 3.00,
    }
    width = width_map.get(variant, 1.00)
    model = FRESNet(num_classes=num_classes, width_mult=width)
    
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained}")
    return model
