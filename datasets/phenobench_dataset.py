import os
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class PhenoBenchDataset(Dataset):
    """
    Dataset loader for PhenoBench Semantic Segmentation Benchmark.
    Resolution: 512x512
    Classes:
      0: Background / Soil
      1: Crop (Sugar beet)
      2: Weed
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        target_size: Tuple[int, int] = (512, 512),
        is_train: bool = False
    ):
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        self.is_train = is_train

        self.img_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'semantics')

        self.img_names = []
        if os.path.exists(self.img_dir):
            self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            mask_np = np.array(mask, dtype=np.int64)
            mask_tensor = torch.from_numpy(mask_np).long().unsqueeze(0)
        else:
            # Blind test set dummy mask
            mask_tensor = torch.zeros((1, image.height, image.width), dtype=torch.long)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.resize(image_tensor, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask_tensor = TF.resize(mask_tensor, self.target_size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        if self.is_train:
            if torch.rand(1).item() > 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_tensor = TF.normalize(image_tensor, mean=mean, std=std)

        return image_tensor, mask_tensor
