import os
import argparse
import torch
from models.fresnet import FRESNet

def export_onnx(model, output_path: str, input_size=(1, 3, 512, 512), opset: int = 14):
    model.eval()
    dummy_input = torch.randn(*input_size)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported ONNX model to: {output_path} (Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export FRES-Net to ONNX format")
    parser.add_argument('--checkpoint', type=str, default='pretrained_weights/fresnet_phenobench_best.pth')
    parser.add_argument('--output', type=str, default='pretrained_weights/fresnet.onnx')
    parser.add_argument('--width_mult', type=float, default=1.0)
    args = parser.parse_args()

    net = FRESNet(num_classes=3, width_mult=args.width_mult)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        net.load_state_dict(sd, strict=False)
        print(f"Loaded weights from {args.checkpoint}")
        
    export_onnx(net, args.output)
