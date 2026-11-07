import time
import argparse
import torch
from models.fresnet import FRESNet

def profile_model(model, input_size=(1, 3, 512, 512), device='cuda', num_warmup=20, num_runs=100):
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print(f"Profiling on device: {device}")
    
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Latency timing
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    avg_latency = sum(timings) / len(timings)
    fps = 1000.0 / avg_latency
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 50)
    print(f"Model Parameters: {num_params:,} ({num_params / 1e6:.3f} M)")
    print(f"Average Latency : {avg_latency:.2f} ms")
    print(f"Throughput      : {fps:.1f} FPS")
    print("=" * 50)

    return num_params, avg_latency, fps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Profile FRES-Net latency and throughput")
    parser.add_argument('--variant', type=str, default='fresnet_w100', choices=['fresnet_w033', 'fresnet_w050', 'fresnet_w100', 'fresnet_w200', 'fresnet_w300'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--runs', type=int, default=100)
    args = parser.parse_args()

    width_map = {'fresnet_w033': 0.33, 'fresnet_w050': 0.50, 'fresnet_w100': 1.00, 'fresnet_w200': 2.00, 'fresnet_w300': 3.00}
    net = FRESNet(num_classes=3, width_mult=width_map[args.variant])
    profile_model(net, device=args.device, num_runs=args.runs)
