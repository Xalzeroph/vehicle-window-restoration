"""ExpoMamba 低光增强推理（Triton selective_scan 替代 + CUDA graph 加速）。

用法:
    python3 inference_expomamba.py --input <图或目录> --output <目录>
        [--ckpt ../pretrained/expomamba/best_psnr.pth]
        [--norm none|imagenet] [--gpu 0]
"""
import argparse, glob, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.expomamba_2d import ExpoMamba

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def norm_img(t):
    return (t - MEAN.to(t.device)) / STD.to(t.device)


def denorm_img(t):
    return (t * STD.to(t.device) + MEAN.to(t.device)).clamp(0, 1)


class GraphRunner:
    def __init__(self, model, device, use_graph=True):
        self.model = model
        self.device = device
        self.use_graph = use_graph
        self.graphs = {}  # (H, W) -> (graph, x_static, y_static)
        if not use_graph:
            return

    def run(self, x):
        if not self.use_graph:
            with torch.no_grad():
                return self.model(x)
        key = tuple(x.shape[-2:])
        ent = self.graphs.get(key)
        if ent is None:
            ent = self._capture(x)
            self.graphs[key] = ent
        g, xs, ys = ent
        xs.copy_(x)
        g.replay()
        return ys.clone()

    def _capture(self, x):
        torch.cuda.empty_cache()
        xs = x.clone()
        with torch.no_grad():
            for _ in range(3):
                self.model(xs)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model(xs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            ys = self.model(xs)
        torch.cuda.synchronize()
        return g, xs, ys


def pad16(im):
    h, w = im.shape[:2]
    hp = (h + 15) // 16 * 16
    wp = (w + 15) // 16 * 16
    if hp == h and wp == w:
        return im, (h, w)
    out = np.pad(im, ((0, hp - h), (0, wp - w), (0, 0)), mode="edge")
    return out, (h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="expomamba_out")
    ap.add_argument("--ckpt", default="../pretrained/expomamba/best_psnr.pth")
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_side", type=int, default=1280,
                    help="长边超过则等比缩放（0=不缩放）；低光增强按惯例不建议喂超大图")
    ap.add_argument("--no_graph", action="store_true",
                    help="禁用 CUDA graph（输入尺寸混杂时多个 graph 常驻显存会 OOM）")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*")))
    else:
        files = sorted(glob.glob(args.input))
    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    if not files:
        print("no images found"); return
    os.makedirs(args.output, exist_ok=True)

    dev = f"cuda:{args.gpu}"
    model = ExpoMamba(n_channels=3, bilinear=True, base_channels=48, deep_supervision=True)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().to(dev)
    runner = GraphRunner(model, dev, use_graph=not args.no_graph)
    norm = (lambda t: t) if args.norm == "none" else norm_img
    denorm = (lambda t: t.clamp(0, 1)) if args.norm == "none" else denorm_img

    t0 = time.time()
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        if args.max_side and max(im.size) > args.max_side:
            s = args.max_side / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        a = np.asarray(im).astype(np.float32) / 255.0
        ap_, (h, w) = pad16(a)
        x = torch.from_numpy(ap_.transpose(2, 0, 1)[None]).to(dev)
        x = norm(x)
        with torch.no_grad():
            y = runner.run(x)
        y = denorm(y)[0].cpu().numpy().transpose(1, 2, 0)
        out_im = Image.fromarray((y[:h, :w] * 255).astype(np.uint8))
        out_im.save(os.path.join(args.output, os.path.basename(f)))
        if (i + 1) % 8 == 0 or i == len(files) - 1:
            print(f"[{i+1}/{len(files)}] {(time.time()-t0)/(i+1)*1000:.0f} ms/frame")
    print(f"done -> {args.output}")


if __name__ == "__main__":
    main()
