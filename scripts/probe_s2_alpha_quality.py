#!/usr/bin/env python3
"""S2 出口头的 α̃ 质量诊断（训练分布内）：α̃ / g 与 GT α 的一致性。

回答："出口头真的在估 α 吗，还是学到了一个近常数的工作点？"
指标：逐场景 corr(预测场, GT α 场)、MAE(场景均值 vs GT 场景均值)；
     基线 = 常数预测器（全局平均 α）。
用法: python3 scripts/probe_s2_alpha_quality.py --exit runs/ckpt/exit_E_s20260915/ckpt_final.pth --n 48
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, RDNET)
sys.path.insert(0, HERE)
os.chdir(RDNET)
from finetune_rdnet_exit import build_plain, ExitHead  # noqa: E402


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exit", required=True)
    ap.add_argument("--triplets", default="data/train_sets/train_triplets_rdt_0915")
    ap.add_argument("--n", type=int, default=48, help="每档取样场景数")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    ex = args.exit if os.path.isabs(args.exit) else os.path.join(PROJ, args.exit)
    trip = args.triplets if os.path.isabs(args.triplets) else os.path.join(PROJ, args.triplets)

    dev = torch.device(f"cuda:{args.gpu}")
    net, net_c = build_plain(os.path.join(PROJ, "runs/ckpt/rdt_plain/ckpt_final.pth"))
    net.to(dev); net_c.to(dev)
    hk = torch.load(ex, map_location="cpu")
    head = ExitHead(hk["args"]["arm"]).to(dev).eval()
    head.load_state_dict(hk["head"], strict=True)
    arm = hk["args"]["arm"]

    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(trip, "T")))[::max(1, 4000 // args.n)][:args.n]
    levels = [1.0, 0.75, 0.6, 0.45, 0.3]
    stats = {lv: [] for lv in levels}
    with torch.no_grad():
        for lv in levels:
            tag = f"l{round(lv*100):03d}"
            for sid in ids:
                I = load(os.path.join(trip, "I", f"{sid}_{tag}.png"))[None]
                A = np.load(os.path.join(trip, "alpha", f"{sid}_{tag}.npy")).astype(np.float32)
                if A.ndim == 2:
                    A = A[None]
                x = torch.from_numpy(I).to(dev)
                code = net_c(x)
                with torch.cuda.amp.autocast():
                    _, cols = net(x, code, prompt=True)
                o = cols[-1].float()
                R_hat, T_head = o[:, :3].clamp(0, 1), o[:, 3:].clamp(0, 1)
                beta, main, _ = head(x.clamp(0, 1), R_hat, T_head)
                pred = main[0, 0].cpu().numpy()          # α̃ (E) 或 g (G)
                a = A[0]
                pred_r, a_r = pred.ravel(), a.ravel()
                corr = float(np.corrcoef(pred_r, a_r)[0, 1])
                if arm == "G":                            # g ≈ 1/α 时一致性按 1/g 比
                    corr = float(np.corrcoef(pred_r, 1.0 / np.maximum(a_r, 0.05))[0, 1])
                stats[lv].append((corr, float(pred.mean()), float(a.mean()), float(beta.mean())))

    print(f"arm={arm}  n/档={len(ids)}")
    print(f"{'level':>6} {'corr(预测,GT)':>14} {'预测均值':>9} {'GT α 均值':>10} {'β 均值':>8}")
    for lv in levels:
        c = np.array(stats[lv])
        print(f"{lv:6.2f} {c[:,0].mean():14.3f} {c[:,1].mean():9.3f} {c[:,2].mean():10.3f} {c[:,3].mean():8.3f}")
    allc = np.concatenate([np.array(stats[lv]) for lv in levels])
    print(f"合计: corr {allc[:,0].mean():.3f} | 预测均值 {allc[:,1].mean():.3f} vs GT {allc[:,2].mean():.3f} "
          f"| 场景均值 MAE {np.abs(allc[:,1]-allc[:,2]).mean():.3f}")


if __name__ == "__main__":
    main()
