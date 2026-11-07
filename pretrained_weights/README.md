# Pretrained Model Checkpoints for FRES-Net

This directory contains the official pretrained PyTorch weights (`.pth`) for the FRES-Net model family evaluated in our study:
> **"Efficient crop and weed segmentation using a lightweight factorized network and contrastive learning"**

All models are fully compatible with PyTorch 2.x and torchvision. Because of FRES-Net's ultra-lightweight design (0.55 M parameters), all checkpoints are compact (~2.2 MB each) and ready for immediate inference, deployment, or fine-tuning on custom agricultural datasets.

---

## Available Pretrained Weights

| Checkpoint File | Target Benchmark | Protocol / Variant | Parameters | File Size | Key Test Performance | Primary Use Case |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `fresnet_phenobench_best.pth` | **PhenoBench** | Full 512×512 ($w=1.00$) | **0.55 M** | 2.19 MB | **84.41% mIoU** (Rank 8 CodaBench), 86.9% Weed IoU | Out-of-the-box sugar beet & weed segmentation / Transfer learning |
| `fresnet_phenobench_best_weed_iou.pth` | **PhenoBench** | Full 512×512 ($w=1.00$) | **0.55 M** | 2.18 MB | **87.21% Weed IoU**, 84.15% mIoU | Weed-priority precision robotic spraying |
| `fresnet_cwfid_protocol1_best.pth` | **CWFID** | Protocol 1 ($w=1.00$) | **0.55 M** | 2.19 MB | **72.10% mIoU**, 83.21% $F_1^{cw}$ | Standard benchmark reproduction (Protocol 1) |
| `fresnet_cwfid_protocol2_best.pth` | **CWFID** | Protocol 2 ($w=1.00$) | **0.55 M** | 2.18 MB | **66.60% mIoU**, 78.47% $F_1^{cw}$ | Cluster-stratified benchmark reproduction (Protocol 2) |
| `fresnet_w033_cwfid_prot2_best.pth` | **CWFID** | Protocol 2 ($w=0.33$) | **0.18 M** | 0.76 MB | **58.60% mIoU**, 72.10% $F_1^{cw}$ | Ultra-constrained embedded edge devices |
| `fresnet_w300_cwfid_prot2_best.pth` | **CWFID** | Protocol 2 ($w=3.00$) | **3.61 M** | 14.00 MB | **71.74% mIoU**, 83.05% $F_1^{cw}$ | High-capacity upper bound |

---

## Quick Loading in Python

```python
import torch
from models.fresnet import FRESNet

# 1. Instantiate model
model = FRESNet(num_classes=3, width_mult=1.0)

# 2. Load pretrained weights
checkpoint_path = 'pretrained_weights/fresnet_phenobench_best.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 3. Extract and load state_dict
state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
model.load_state_dict(state_dict, strict=False)
model.eval()

print("FRES-Net loaded successfully!")
```

---

## Fine-Tuning on Custom Agricultural Data

To fine-tune FRES-Net on your own agricultural dataset:

```bash
python train_cwfid.py \
    --data_root /path/to/custom_dataset \
    --pretrained pretrained_weights/fresnet_phenobench_best.pth \
    --epochs 50 \
    --batch_size 8 \
    --lr 5e-4
```

---

## Google Drive Mirror

For permanent open access and additional raw training logs / TensorBoard event logs, model weights and checkpoints can also be accessed via the open Google Drive mirror:
* **Google Drive Open Mirror:** `[Insert Public Google Drive Folder Link Here]`
