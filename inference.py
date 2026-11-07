import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from models.fresnet import FRESNet

def run_inference(model, image_path: str, output_path: str, target_size=(512, 512), device='cuda', alpha=0.5):
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()

    orig_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = orig_img.size

    img_tensor = TF.to_tensor(orig_img)
    img_tensor = TF.resize(img_tensor, target_size, interpolation=TF.InterpolationMode.BILINEAR)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_tensor)
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize prediction back to original image size
    pred_pil = Image.fromarray(pred).resize((orig_w, orig_h), resample=Image.NEAREST)
    pred_arr = np.array(pred_pil)

    # Create RGB mask: 0=black, 1=green (Crop), 2=red (Weed)
    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    color_mask[pred_arr == 1] = [46, 204, 113]   # Crop: Emerald Green
    color_mask[pred_arr == 2] = [231, 76, 60]    # Weed: Coral Red

    # Blend overlay
    orig_np = np.array(orig_img)
    fg_mask = (pred_arr > 0)[..., None]
    blended = np.where(fg_mask, (1 - alpha) * orig_np + alpha * color_mask, orig_np).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(blended).save(output_path)
    print(f"Saved inference visualization to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run FRES-Net inference on an image")
    parser.add_argument('--input', type=str, required=True, help="Path to input RGB image")
    parser.add_argument('--output', type=str, default='output_overlay.png', help="Output overlay image path")
    parser.add_argument('--checkpoint', type=str, default='pretrained_weights/fresnet_phenobench_best.pth')
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    net = FRESNet(num_classes=3, width_mult=args.width_mult)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    net.load_state_dict(sd, strict=False)
    
    run_inference(net, args.input, args.output, device=args.device)
