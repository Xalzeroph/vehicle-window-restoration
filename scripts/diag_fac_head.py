#!/usr/bin/env python3
"""S1 判决辅助诊断: 分解头 naked 锚定失败归因 (m̂ vs ĉ).

分区:
  film     = α_GT < 0.95            (可观测膜区, 掩膜监督目标)
  naked-near = α_GT ≥ 0.95 且距膜区 ≤4px   (疑"被 clip 的膜", 物理不可分)
  naked-far  = α_GT ≥ 0.95 且距膜区 >4px   (真裸区, 锚定应发生处)

判读: 若 naked-far 的 m̂ 也高 → 掩膜头真的在裸区误报 (锚定失败);
      若只有 naked-near 高 → 失败集中在物理不可分区, 锚定结构本身成立。
另报逐图 ĉ / 预测档 / p_max。
"""
import argparse, glob, os, sys
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "RDNet"))
os.chdir(os.path.join(PROJ, "RDNet"))

from models.arch.rdnet_atap import FullNet_NLP_ATap_Fac  # noqa: E402

LEVELS = [1.0, 0.6, 0.3]


def load(f):
    return np.asarray(Image.open(f).convert("RGB"), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["taps", "raw"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ts", default=os.path.join(PROJ, "data/test_sets/synth_rrw_v21_0913"))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    ckpt = args.ckpt or os.path.join(PROJ, "runs/ckpt", f"fac_{args.encoder}", "ckpt_final.pth")
    device = torch.device(f"cuda:{args.gpu}")

    net = FullNet_NLP_ATap_Fac(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
                               num_subnet=4, loss_col=4, inject=True,
                               raw_encoder=(args.encoder == "raw"))
    net.load_state_dict(torch.load(ckpt, map_location="cpu")["icnn"], strict=False)
    net.eval().to(device)

    stems = sorted(os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(args.ts, "gt", "*.png")))
    print(f"=== 诊断 [fac-{args.encoder}] {ckpt} ===")
    print(f"{'档':>5s} {'区域':>11s} {'像素%':>7s} {'α_GT':>6s} {'α̂':>6s} {'m̂':>6s}")
    for lv in LEVELS:
        acc = {k: dict(n=0, a=0.0, ah=0.0, m=0.0) for k in ("film", "naked-near", "naked-far")}
        tot = 0
        chats, lv_pred, lv_true = [], [], []
        for stem in stems:
            i = load(os.path.join(args.ts, "levels", f"alpha_{lv:g}", stem + ".png"))
            x = torch.from_numpy(i.transpose(2, 0, 1) / 255.0)[None].to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                a, m, g, p = net.forward_alpha_only(x)
            ah = a.float()[0, 0].cpu().numpy()
            mh = m.float()[0, 0].cpu().numpy().clip(0, 1)
            ph = p.float()[0].cpu().numpy()
            gt = np.load(os.path.join(args.ts, "alpha", f"{stem}_a{lv:g}.npy")).astype(np.float32)

            film = gt < 0.95
            naked = ~film
            d = distance_transform_edt(~film)
            near = naked & (d <= 4)
            far = naked & (d > 4)
            for name, sel in (("film", film), ("naked-near", near), ("naked-far", far)):
                if sel.sum() == 0:
                    continue
                acc[name]["n"] += int(sel.sum())
                acc[name]["a"] += float(gt[sel].sum())
                acc[name]["ah"] += float(ah[sel].sum())
                acc[name]["m"] += float(mh[sel].sum())
            tot += gt.size
            ch = float((ph * np.array(LEVELS)).sum())
            chats.append(ch)
            lv_pred.append(int(ph.argmax()))
            lv_true.append(LEVELS.index(lv))
        for name in ("film", "naked-near", "naked-far"):
            s = acc[name]
            if s["n"] == 0:
                continue
            print(f"{lv:5.1f} {name:>11s} {100*s['n']/tot:6.1f}% "
                  f"{s['a']/s['n']:6.3f} {s['ah']/s['n']:6.3f} {s['m']/s['n']:6.3f}")
        ch_a = np.array(chats)
        print(f"      ĉ 均值 {ch_a.mean():.3f} (逐图 {np.round(ch_a, 2)})  "
              f"预测档 {lv_pred.count(LEVELS.index(lv))}/12 对")
    print("\n(α_GT=膜区真值, α̂/m̂=头输出; naked-near=距膜≤4px 的裸区, naked-far=远离膜的真裸区)")


if __name__ == "__main__":
    main()
