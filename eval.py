import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.fresnet import FRESNet
from datasets.cwfid_dataset import CWFIDDataset
from datasets.phenobench_dataset import PhenoBenchDataset

def compute_iou_and_f1(conf_matrix):
    """
    Computes per-class IoU and F1-score from confusion matrix.
    conf_matrix shape: (num_classes, num_classes) [ground_truth, prediction]
    """
    tp = np.diag(conf_matrix).astype(float)
    fp = (np.sum(conf_matrix, axis=0) - tp).astype(float)
    fn = (np.sum(conf_matrix, axis=1) - tp).astype(float)

    denominator_iou = tp + fp + fn
    iou = np.divide(tp, denominator_iou, out=np.zeros_like(tp), where=denominator_iou != 0)
    
    denominator_f1 = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator_f1, out=np.zeros_like(tp), where=denominator_f1 != 0)
    
    return iou, f1

def evaluate(model, dataloader, num_classes=3, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()

    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.cpu().numpy().flatten()
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()

            # Fast vectorized confusion matrix accumulation
            valid = (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
            if np.any(valid):
                conf_mat += np.bincount(
                    num_classes * targets[valid] + preds[valid],
                    minlength=num_classes**2
                ).reshape(num_classes, num_classes)

    iou, f1 = compute_iou_and_f1(conf_mat)
    class_names = ['Background', 'Crop', 'Weed']
    
    print("\n" + "=" * 55)
    print("           FRES-NET BENCHMARK EVALUATION")
    print("=" * 55)
    for c, name in enumerate(class_names):
        print(f"  Class {c} ({name:10s}): IoU = {iou[c]*100:6.2f}% | F1 = {f1[c]*100:6.2f}%")
    print("-" * 55)
    print(f"  Overall Mean IoU (mIoU)  : {np.mean(iou)*100:6.2f}%")
    print(f"  Overall Mean F1 (mF1)    : {np.mean(f1)*100:6.2f}%")
    print(f"  Crop/Weed IoU (mIoU_cw)  : {((iou[1] + iou[2]) / 2.0)*100:6.2f}%")
    print(f"  Crop/Weed F1 (F1_cw)     : {((f1[1] + f1[2]) / 2.0)*100:6.2f}%")
    print("=" * 55)

    return iou, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate FRES-Net Checkpoint")
    parser.add_argument('--dataset', type=str, default='cwfid', choices=['cwfid', 'phenobench'])
    parser.add_argument('--data_root', type=str, required=True, help="Path to dataset root folder")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to pretrained .pth checkpoint")
    parser.add_argument('--protocol', type=int, default=1, choices=[1, 2], help="CWFID protocol (1 or 2)")
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    net = FRESNet(num_classes=3, width_mult=args.width_mult)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    net.load_state_dict(sd, strict=False)
    print(f"Successfully loaded checkpoint: {args.checkpoint}")

    if args.dataset == 'cwfid':
        ds = CWFIDDataset(args.data_root, split='test', protocol=args.protocol, is_train=False)
    else:
        ds = PhenoBenchDataset(args.data_root, split='val', is_train=False)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    evaluate(net, loader, device=args.device)
