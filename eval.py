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
    conf_matrix shape: (num_classes, num_classes) [pred, target] or [target, pred]
    """
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp

    denominator_iou = tp + fp + fn
    iou = np.divide(tp, denominator_iou, out=np.zeros_like(tp, dtype=float), where=denominator_iou != 0)
    
    denominator_f1 = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator_f1, out=np.zeros_like(tp, dtype=float), where=denominator_f1 != 0)
    
    return iou, f1

def evaluate(model, dataloader, num_classes=3, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()

    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.cpu().numpy()
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for t, p in zip(targets.flatten(), preds.flatten()):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    conf_mat[t, p] += 1

    iou, f1 = compute_iou_and_f1(conf_mat)
    class_names = ['Background', 'Crop', 'Weed']
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for c, name in enumerate(class_names):
        print(f"Class {c} ({name:10s}): IoU = {iou[c]*100:6.2f}% | F1 = {f1[c]*100:6.2f}%")
    print("-" * 50)
    print(f"Mean IoU (mIoU)        : {np.mean(iou)*100:6.2f}%")
    print(f"Mean F1 (mF1)          : {np.mean(f1)*100:6.2f}%")
    print(f"Crop/Weed F1 (F1^cw)   : {((f1[1] + f1[2]) / 2.0)*100:6.2f}%")
    print("=" * 50)

    return iou, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate FRES-Net Checkpoint")
    parser.add_argument('--dataset', type=str, default='cwfid', choices=['cwfid', 'phenobench'])
    parser.add_argument('--data_root', type=str, required=True, help="Path to dataset root folder")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to pretrained .pth checkpoint")
    parser.add_argument('--protocol', type=int, default=1, choices=[1, 2], help="CWFID protocol (1 or 2)")
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    net = FRESNet(num_classes=3, width_mult=args.width_mult)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    net.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint from: {args.checkpoint}")

    if args.dataset == 'cwfid':
        ds = CWFIDDataset(args.data_root, split='test', protocol=args.protocol, is_train=False)
    else:
        ds = PhenoBenchDataset(args.data_root, split='val', is_train=False)

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    evaluate(net, loader, device=args.device)
