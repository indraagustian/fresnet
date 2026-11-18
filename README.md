<div align="center">

# FRES-Net: Fast Residual Efficient Segmentation Network

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

**Official PyTorch Implementation and Pretrained Models for:**  
> **"Efficient crop and weed segmentation using a lightweight factorized network and contrastive learning"**  
> *Indra Agustian<sup>1,2</sup>, Igi Ardiyanto<sup>1</sup>, Ruvita Faurina<sup>2</sup>, and Sunu Wibirama<sup>1</sup>*  
> <sup>1</sup>*Universitas Gadjah Mada, Indonesia* | <sup>2</sup>*Universitas Bengkulu, Indonesia*

</div>

---

## 🌟 Key Highlights

* **Ultra-Lightweight Footprint:** Only **0.554 M parameters** and **10.88 GFLOPs** (at 512×512 resolution), making FRES-Net over **100× smaller** than conventional segmentation models (e.g., DeepLabV3+ with ResNeXt-101 at 58.9 M parameters and 346.8 GFLOPs).
* **Competitive Benchmark Accuracy:**
  * **PhenoBench Benchmark:** **84.41% mIoU** on the official blind test server, outperforming heavy baselines while utilizing <1% of their parameter budget.
  * **CWFID Benchmark:** **72.80% mIoU** (74.75% $F_1^{cw}$, PDR = 6.69) on Protocol 1 and **78.64% mIoU** (68.70% $\mathrm{mIoU}_{cw}$, PDR = 7.23) on the rigorous cluster-stratified Protocol 2.
* **Real-Time Edge Inference:**
  * **107.5 FPS** on NVIDIA RTX 4090 GPU (9.30 ms latency).
  * **63.8 FPS** on NVIDIA Jetson Orin Nano edge platform (15.67 ms latency) — well exceeding standard real-time field robotics thresholds (30 FPS).
* **Self-Contained Pretrained Weights:** Direct out-of-the-box weights (`.pth`, ~2.18 MB each) included for inference, zero-shot testing, and rapid fine-tuning on custom agricultural datasets.

---

## 🏗️ Architecture Overview

FRES-Net decouples high-resolution spatial boundary preservation from deep semantic context extraction:

```
Input RGB (512x512)
       │
       ├─────────────────────────────────────────────┐ (Spatial detail branch)
       ▼                                             │
┌──────────────────────────────┐                     │
│ Learning to Downsample (LTD) │ (8x downsampling)   │
└──────────────┬───────────────┘                     │
               ▼                                     ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐
│ Global Feature Extractor     ├──────>│ Feature Fusion Module (FFM)  │
│ (Dilated Factorized 1D NBB)  │       │ + Efficient Channel Attn     │
└──────────────────────────────┘       └──────────────┬───────────────┘
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │ ERFNet-Style Factorized      │
                                       │ Transposed Conv Decoder      │
                                       └──────────────┬───────────────┘
                                                      ▼
                                            Semantic Mask (512x512)
                                       [0: Soil, 1: Crop, 2: Weed]
```

1. **Learning to Downsample (LTD):** Rapid 8× resolution reduction utilizing depthwise separable convolutions to preserve high-frequency crop/weed leaf boundaries without computational bloat.
2. **Global Feature Extractor (GFE):** Sequential 1D factorized convolutions ($3\times 1$ and $1\times 3$) with progressive dilation rates ($d \in \{1, 2, 4\}$) providing extensive contextual receptive field.
3. **Feature Fusion Module (FFM) with ECA:** Cross-scale feature aggregation enhanced with parameter-free Efficient Channel Attention (ECA).
4. **ERFNet-Style Hierarchical Decoder:** 3-stage transposed convolution upsampler with residual non-bottleneck blocks eliminating checkerboard artifacts.

---

## 📁 Official Benchmark Outputs

Official CodaBench blind test evaluation outputs for submission FRESNet+PCL+ECA+ERFNetDec Optimized Thresholds (scoring run 1957) are archived in:
* [Benchmark_results/phenobench_codabench_submission_1957/](benchmark_results/phenobench_codabench_submission_1957/): Contains detailed_results.html, scores.txt, and per-date YAML score breakdowns generated directly by the official CodaBench evaluation server.

---

## 📦 Directory Structure

```
fresnet/
├── models/
│   ├── fresnet.py                 # Core FRES-Net model & capacity scaling variants
│   ├── losses.py                  # Focal, Dice, and Pixel Contrastive (PCL) losses
│   └── modules/
│       ├── factorized_res.py      # NonBottleneck1D factorized block
│       ├── downsample.py          # LearningToDownsample (LTD) module
│       ├── attention.py           # ECALayer and FeatureFusionModule (FFM)
│       └── decoders.py            # ERFNet-Style factorized decoder
├── datasets/
│   ├── cwfid_dataset.py           # CWFID dataset loader (Protocols 1 & 2)
│   ├── phenobench_dataset.py      # PhenoBench dataset loader
│   └── splits/
│       ├── cwfid_protocol1_split.json  # Fixed standard 45/6/9 split indices
│       └── cwfid_protocol2_split.json  # Cluster-stratified 45/6/9 split indices
├── pretrained_weights/            # Curated pretrained checkpoints (~2.18 MB each)
│   ├── fresnet_phenobench_best.pth
│   ├── fresnet_phenobench_best_weed_iou.pth
│   ├── fresnet_cwfid_protocol1_best.pth
│   ├── fresnet_cwfid_protocol2_best.pth
│   ├── fresnet_w033_cwfid_prot2_best.pth
│   ├── fresnet_w300_cwfid_prot2_best.pth
│   └── README.md
├── tools/
│   ├── profile_latency.py         # Measure FPS, latency, and FLOPs
│   └── export_onnx.py             # Export PyTorch weights to ONNX
├── train_cwfid.py                 # Training / Fine-tuning on CWFID
├── train_phenobench.py            # Training / Fine-tuning on PhenoBench
├── eval.py                        # Benchmark evaluation script
├── inference.py                   # Single image inference & color overlay
├── requirements.txt
├── LICENSE                        # MIT License
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/indraagustian/fresnet.git
cd fresnet
pip install -r requirements.txt
```

### 2. Inference on a Test Image

Run inference on an agricultural field image with automatic green (Crop) and red (Weed) overlay visualization:

```bash
python inference.py \
    --input examples/sample_field.png \
    --output output_overlay.png \
    --checkpoint pretrained_weights/fresnet_phenobench_best.pth
```

### 3. Python API Quick Usage

```python
import torch
from models.fresnet import FRESNet

# Initialize FRES-Net (0.554 M params, 10.88 GFLOPs)
model = FRESNet(num_classes=3, width_mult=1.0)

# Load pretrained weights
checkpoint = torch.load('pretrained_weights/fresnet_phenobench_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# Forward pass
dummy_input = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    predictions = model(dummy_input) # Output shape: (1, 3, 512, 512)
```

---

## 📊 Pretrained Weights & Benchmark Results

All pretrained weights are included directly in [`pretrained_weights/`](file:///G:/My%20Drive/fresnet/pretrained_weights):

| Model Checkpoint | Dataset | Protocol / Scaling | Params (M) | GFLOPs ($512^2$) | Test mIoU | Weed IoU | Crop IoU | $\mathrm{mIoU}_{cw}$ | PDR |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`fresnet_phenobench_best.pth`](file:///G:/My%20Drive/fresnet/pretrained_weights/fresnet_phenobench_best.pth) | **PhenoBench** | Full 512×512 ($w=1.00$) | **0.554** | **10.88** | **84.41%** | 86.90% | 76.50% | 81.70% | **7.76** |
| [`fresnet_phenobench_best_weed_iou.pth`](file:///G:/My%20Drive/fresnet/pretrained_weights/fresnet_phenobench_best_weed_iou.pth) | **PhenoBench** | Full 512×512 ($w=1.00$) | **0.554** | **10.88** | **84.15%** | **87.21%** | 76.32% | 81.76% | **7.73** |
| [`fresnet_cwfid_protocol1_best.pth`](file:///G:/My%20Drive/fresnet/pretrained_weights/fresnet_cwfid_protocol1_best.pth) | **CWFID** | Protocol 1 ($w=1.00$) | **0.554** | **10.88** | **72.80%** | 69.22% | 51.15% | 60.19% | **6.69** |
| [`fresnet_cwfid_protocol2_best.pth`](file:///G:/My%20Drive/fresnet/pretrained_weights/fresnet_cwfid_protocol2_best.pth) | **CWFID** | Protocol 2 ($w=1.00$) | **0.554** | **10.88** | **78.64%** | 76.93% | 60.46% | 68.70% | **7.23** |
| [`fresnet_w033_cwfid_prot2_best.pth`](file:///G:/My%20Drive/fresnet/pretrained_weights/fresnet_w033_cwfid_prot2_best.pth) | **CWFID** | Protocol 2 ($w=0.33$) | **0.141** | **2.94** | **73.18%** | 70.88% | 50.42% | 60.65% | **24.89** |
| [`fresnet_w300_cwfid_prot2_best.pth`](file:///G:/My%20Drive/fresnet/pretrained_weights/fresnet_w300_cwfid_prot2_best.pth) | **CWFID** | Protocol 2 ($w=3.00$) | **3.598** | **49.34** | **82.40%** | 79.92% | 68.69% | 74.30% | **1.67** |

*Note: Pretrained weights and training checkpoints are also mirrored on Google Drive: [Open Google Drive Link].*

---

## 🧪 Evaluation & Reproduction

To evaluate the pretrained checkpoint on the CWFID test split:

```bash
# Protocol 1 evaluation
python eval.py \
    --dataset cwfid \
    --data_root /path/to/cwfid \
    --checkpoint pretrained_weights/fresnet_cwfid_protocol1_best.pth \
    --protocol 1

# Protocol 2 evaluation
python eval.py \
    --dataset cwfid \
    --data_root /path/to/cwfid \
    --checkpoint pretrained_weights/fresnet_cwfid_protocol2_best.pth \
    --protocol 2
```

To profile hardware latency, throughput (FPS), and FLOPs on your machine:

```bash
python tools/profile_latency.py --variant fresnet_w100 --device cuda
```

---

## 🛠️ Training & Fine-Tuning

### Training on CWFID from Scratch or Fine-Tuning

```bash
python train_cwfid.py \
    --data_root /path/to/cwfid \
    --protocol 1 \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --save_dir checkpoints_cwfid
```

### Fine-Tuning on PhenoBench or Custom Agricultural Data

```bash
python train_phenobench.py \
    --data_root /path/to/phenobench \
    --pretrained pretrained_weights/fresnet_phenobench_best.pth \
    --epochs 50 \
    --batch_size 8 \
    --lr 5e-4
```

---

## ⚡ ONNX Edge Deployment

Export FRES-Net to ONNX for embedded deployment with TensorRT, ONNX-Runtime, or OpenVINO:

```bash
python tools/export_onnx.py \
    --checkpoint pretrained_weights/fresnet_phenobench_best.pth \
    --output pretrained_weights/fresnet.onnx
```

---

## 📜 Citation

If you find FRES-Net useful in your research or applications, please cite our paper:

```bibtex
@misc{agustian2026fresnet,
  author       = {Agustian, Indra and Ardiyanto, Igi and Faurina, Ruvita and Wibirama, Sunu},
  title        = {Efficient Crop and Weed Segmentation Using a Lightweight Factorized Network and Contrastive Learning},
  year         = {2026},
  howpublished = {\url{https://github.com/indraagustian/fresnet}},
  note         = {Manuscript under review; official code repository}
}
```

---

## 📄 License & Acknowledgments

* **Code License:** [MIT License](LICENSE)
* **Funding Acknowledgments:** This work was supported by the Indonesia Endowment Fund for Education (LPDP) through the Indonesian Education Scholarship scheme (Decree No. KEP.1241/LPDP/LPDP.3/2022).
