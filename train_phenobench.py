import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.fresnet import FRESNet
from models.losses import FRESNetCompositeLoss
from datasets.phenobench_dataset import PhenoBenchDataset
from eval import evaluate

def train_phenobench(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset & Loaders
    train_ds = PhenoBenchDataset(args.data_root, split='train', target_size=(args.img_size, args.img_size), is_train=True)
    val_ds = PhenoBenchDataset(args.data_root, split='val', target_size=(args.img_size, args.img_size), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Model
    model = FRESNet(num_classes=3, width_mult=args.width_mult).to(device)

    if args.pretrained:
        print(f"Loading pretrained weights from: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(sd, strict=False)

    # 3. Loss & Optimizer
    criterion = FRESNetCompositeLoss(dice_weight=0.45, focal_weight=0.45, cl_weight=0.10, gamma=2.0, temperature=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    best_miou = 0.0

    print("=" * 60)
    print("Starting FRES-Net Training on PhenoBench")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f} M | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits, features = model(images, return_features=True)
            loss, loss_dict = criterion(logits, masks, features)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Validation
        if epoch % 2 == 0 or epoch == args.epochs:
            iou, f1 = evaluate(model, val_loader, device=device)
            val_miou = iou.mean()
            scheduler.step(val_miou)

            print(f"[Epoch {epoch:03d}/{args.epochs:03d}] Train Loss: {epoch_loss:.4f} | Val mIoU: {val_miou*100:.2f}% (Weed IoU: {iou[2]*100:.2f}%)")

            if val_miou > best_miou:
                best_miou = val_miou
                best_path = os.path.join(args.save_dir, 'fresnet_phenobench_best.pth')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_miou': best_miou}, best_path)
                print(f"  >>> Best model saved to: {best_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train / Finetune FRES-Net on PhenoBench")
    parser.add_argument('--data_root', type=str, required=True, help="Path to PhenoBench dataset root")
    parser.add_argument('--save_dir', type=str, default='checkpoints_phenobench')
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--pretrained', type=str, default=None, help="Optional pretrained weights for fine-tuning")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    train_phenobench(args)
