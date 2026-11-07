import os
import json
import random
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class CWFIDDataset(Dataset):
    """
    Dataset loader for Crop/Weed Field Image Dataset (CWFID).
    Compatible with both Protocol 1 (Standard 45/6/9) and Protocol 2 (Cluster-Stratified 45/6/9).
    
    Classes:
      0: Background / Soil
      1: Crop (Sugar beet / Carrot)
      2: Weed
    """
    def __init__(
        self,
        root_dir: str,
        target_size: Tuple[int, int] = (512, 512),
        split: str = 'test',
        protocol: int = 1,
        custom_indices: Optional[List[int]] = None,
        is_train: bool = False
    ):
        self.root_dir = root_dir
        self.target_size = target_size
        self.is_train = is_train
        
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'annotations')
        
        # Load split indices
        if custom_indices is not None:
            self.indices = custom_indices
        else:
            split_file = f"cwfid_protocol{protocol}_split.json"
            split_path = os.path.join(os.path.dirname(__file__), 'splits', split_file)
            if not os.path.exists(split_path):
                # Fallback default Protocol 1
                if split == 'train':
                    self.indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 56, 57, 58]
                elif split == 'val':
                    self.indices = [16, 18, 24, 26, 34, 36]
                else:
                    self.indices = [48, 49, 50, 52, 53, 54, 55, 59, 60]
            else:
                with open(split_path, 'r') as f:
                    data = json.load(f)
                self.indices = data[split]

        # Match files by index
        self.image_files = []
        if os.path.exists(self.img_dir):
            all_files = sorted(os.listdir(self.img_dir))
            for idx in self.indices:
                prefix = f"{idx:03d}_"
                matched = False
                for f in all_files:
                    if f.startswith(prefix) and f.endswith('_image.png'):
                        self.image_files.append(f)
                        matched = True
                        break
                if not matched:
                    # Direct standard naming fallback
                    self.image_files.append(f"{idx:03d}_image.png")
        else:
            self.image_files = [f"{idx:03d}_image.png" for idx in self.indices]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace('_image.png', '_annotation.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask_rgb = np.array(Image.open(mask_path).convert('RGB'))

        # Map RGB annotation to semantic class indices
        # Red channel > 128 -> Weed (2); Green channel > 128 -> Crop (1)
        mask_mapped = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.int64)
        mask_mapped[mask_rgb[:, :, 1] > 128] = 1
        mask_mapped[mask_rgb[:, :, 0] > 128] = 2

        image_tensor = TF.to_tensor(image) # (3, H, W) in [0, 1]
        mask_tensor = torch.from_numpy(mask_mapped).long().unsqueeze(0) # (1, H, W)

        # Resize to target resolution
        image_tensor = TF.resize(image_tensor, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask_tensor = TF.resize(mask_tensor, self.target_size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        # Data augmentation for training
        if self.is_train:
            if random.random() > 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)
            if random.random() > 0.5:
                kernel_size = random.choice([3, 5])
                sigma = random.uniform(0.1, 2.0)
                image_tensor = TF.gaussian_blur(image_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        # Normalize with ImageNet statistics
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_tensor = TF.normalize(image_tensor, mean=mean, std=std)

        return image_tensor, mask_tensor
