#!/usr/bin/env python
"""Comprehensive Benchmark Evaluation Script.
Evaluates model on SIR2, Reflection Removal benchmarks with PSNR/SSIM/LPIPS.
Usage: python src/benchmark.py --model checkpoints/v11/best_loss.pt
"""
import os, sys, argparse, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from models.ojm_net_v10 import OJMNetV10

# Import lpips
try:
    import lpips
    _lpips = lpips.LPIPS(net="alex", verbose=False).eval()
except:
    _lpips = None


class SIR2Dataset(Dataset):
    """SIR2 benchmark dataset loader."""
    def __init__(self, root, size=256):
        self.root = root
        self.size = size
        self.pairs = []
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((size, size))
        if not os.path.exists(root):
            print(f"  WARNING: SIR2 not found: {root}")
            return
        # SIR2 structure: {scene}/{img} with input/gt pairs
        for scene in sorted(os.listdir(root)):
            scene_dir = os.path.join(root, scene)
            if not os.path.isdir(scene_dir):
                continue
            input_dir = os.path.join(scene_dir, "input")
            gt_dir = os.path.join(scene_dir, "gt")
            if not (os.path.exists(input_dir) and os.path.exists(gt_dir)):
                continue
            for f in sorted(os.listdir(input_dir)):
                if f.lower().endswith((".jpg", ".png", ".bmp")):
                    inp = os.path.join(input_dir, f)
                    gtf = os.path.join(gt_dir, f)
                    if os.path.exists(gtf):
                        self.pairs.append((inp, gtf))
        print(f"  SIR2 [{root}]: {len(self.pairs)} pairs")

    def __len__(self):
        return max(1, len(self.pairs))

    def __getitem__(self, idx):
        if not self.pairs:
            d = torch.zeros(3, self.size, self.size)
            return d.clone(), d.clone()
        inp, gt = self.pairs[idx % len(self.pairs)]
        I = self.to_tensor(self.resize(Image.open(inp).convert("RGB")))
        T = self.to_tensor(self.resize(Image.open(gt).convert("RGB")))
        return I, T


@torch.no_grad()
def evaluate(model, loader, device, name=""):
    model.eval()
    total_psnr = 0.0; total_ssim = 0.0; total_lpips = 0.0; count = 0
    for I, Tg in loader:
        I = I.to(device)
        Tg = Tg.to(device)
        Tp, _, _ = model(I)
        mse = F.mse_loss(Tp.clamp(0,1), Tg.clamp(0,1))
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        total_psnr += psnr.item()
        # SSIM
        k1,k2=0.01,0.03; c1,c2=(k1*1)**2,(k2*1)**2
        mp=F.avg_pool2d(Tp,3,1,1); mt=F.avg_pool2d(Tg,3,1,1)
        sp=(F.avg_pool2d(Tp**2,3,1,1)-mp**2).clamp(1e-8)
        st=(F.avg_pool2d(Tg**2,3,1,1)-mt**2).clamp(1e-8)
        spt=F.avg_pool2d(Tp*Tg,3,1,1)-mp*mt
        num=(2*mp*mt+c1)*(2*spt+c2); den=(mp**2+mt**2+c1)*(sp+st+c2)
        ssim=(num.clamp(0)/den.clamp(1e-8)).mean().item()
        total_ssim += ssim
        # LPIPS
        if _lpips is not None:
            total_lpips += _lpips(Tp.clamp(0,1), Tg.clamp(0,1)).mean().item()
        count += 1
    if count == 0: return {}
    avg_psnr = total_psnr/count; avg_ssim = total_ssim/count
    avg_lpips = total_lpips/count if _lpips else None
    result = {"psnr": round(avg_psnr, 2), "ssim": round(avg_ssim, 4)}
    if avg_lpips: result["lpips"] = round(avg_lpips, 4)
    print(f"  {name}: PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f}" +
          (f" LPIPS={avg_lpips:.4f}" if avg_lpips else ""))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Checkpoint path")
    p.add_argument("--sir2", type=str, default="datasets/SIR2", help="SIR2 root")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark device: {device}")

    model = OJMNetV10(f=48, steps=6, sd=0.0).to(device)
    ck = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"Model loaded from {args.model}")

    results = {}
    for name, root in [("SIR2", args.sir2)]:
        if not os.path.exists(root):
            print(f"  Skipping {name}: {root} not found")
            continue
        ds = SIR2Dataset(root, size=args.size)
        dl = DataLoader(ds, args.batch, shuffle=False)
        results[name] = evaluate(model, dl, device, name=name)

    print("="*50)
    print("BENCHMARK RESULTS")
    print(json.dumps(results, indent=2))
    print("="*50)


if __name__ == "__main__":
    main()