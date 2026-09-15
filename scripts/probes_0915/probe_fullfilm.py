#!/usr/bin/env python3
"""探针: "全贴膜图"（无裸区参照）下档位头的行为。

对 v2.1 每场景每档, 取 GT α<0.95 区域内最大的、全膜（腐蚀 8px 后仍属膜区）正方形裁片,
喂模型 forward_alpha_only, 对比: 全图预测 vs 全膜裁片预测（m̂ 均值 / α̂ 均值 / 档位 argmax）。

用法:
  python3 scripts/probes_0915/probe_fullfilm.py --encoder taps --ckpt runs/ckpt/fac_taps_rmix/ckpt_final.pth --gpu 5
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, label

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, RDNET)
os.chdir(RDNET)
from models.arch.rdnet_atap import FullNet_NLP_ATap_Fac  # noqa: E402

import torch  # noqa: E402

TS = os.path.join(PROJ, "data/test_sets/synth_rrw_v21_0913")
LEVELS = [1.0, 0.6, 0.3]


def largest_film_square(film, size=224):
    """膜区(腐蚀后)内最大的 size² 正方形裁片; 无则返回 None。"""
    er = binary_erosion(film, iterations=8)
    lab, n = label(er)
    if n == 0:
        return None
    # 取最大连通域
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    big = (lab == sizes.argmax())
    H, W = film.shape
    # 用积分图找全 True 的最大正方形（下采样步进省时）
    integ = np.zeros((H + 1, W + 1), np.int32)
    integ[1:, 1:] = big.astype(np.int32).cumsum(0).cumsum(1)
    def s(y, x, k):
        y1, x1 = y + k, x + k
        return (integ[y1, x1] - integ[y, x1] - integ[y1, x] + integ[y, x])
    best = None
    for k in range(min(H, W, 320), size - 1, -8):
        step = max(8, k // 12)
        for y in range(0, H - k + 1, step):
            for x in range(0, W - k + 1, step):
                if s(y, x, k) == k * k:
                    best = (y, x, k); break
            if best: break
        if best: break
    if best is None:
        return None
    y, x, k = best
    k = (k // 32) * 32 or 32
    y0 = min(max(0, y + (best[2] - k) // 2), H - k); x0 = min(max(0, x + (best[2] - k) // 2), W - k)
    return y0, x0, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["taps", "raw"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    net = FullNet_NLP_ATap_Fac(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
                               num_subnet=4, loss_col=4, inject=True,
                               raw_encoder=(args.encoder == "raw"))
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu")["icnn"], strict=False)
    net.eval().to(dev)

    def infer(img):
        x = torch.from_numpy(img.transpose(2, 0, 1) / 255.0)[None].to(dev)
        with torch.no_grad(), torch.cuda.amp.autocast():
            a, m, g, p = net.forward_alpha_only(x)
        return (a.float()[0, 0].cpu().numpy(), m.float()[0, 0].cpu().numpy(),
                int(p.float()[0].argmax()))

    print(f"{'场景':6s}{'档':>5s}{'全膜裁片':>10s}{'m̂':>7s}{'膜内α̂':>8s}{'(真值)':>8s}{'判档':>6s}   {'全图判档':>8s}{'全图膜内α̂':>10s}")
    agg = {lv: dict(crop_a=[], crop_p=[], full_p=[]) for lv in LEVELS}
    for lv in LEVELS:
        for f in sorted(glob.glob(os.path.join(TS, "levels", f"alpha_{lv:g}", "*.png"))):
            stem = os.path.splitext(os.path.basename(f))[0]
            img = np.asarray(Image.open(f).convert("RGB"), np.float32)
            a_gt = np.load(os.path.join(TS, "alpha", f"{stem}_a{lv:g}.npy"))
            film = a_gt < 0.95
            box = largest_film_square(film)
            if box is None:
                print(f"{stem:6s}{lv:5g}  无可裁全膜区"); continue
            y0, x0, k = box
            crop = img[y0:y0 + k, x0:x0 + k]
            ah, mh, pi = infer(crop)
            gt_film = a_gt[y0:y0 + k, x0:x0 + k]
            pred_full = np.load(os.path.join(PROJ, "runs/pred",
                                f"fac_pred_taps_rmix" if args.encoder == "taps" else "fac_pred_raw_rmix",
                                f"alpha_{lv:g}", f"{stem}_a.npy")).astype(np.float32)
            full_film_mean = float(pred_full[film].mean())
            # 全图档位（从 summary 无法逐图取, 这里用 m̂/p 重算代价大→用 csv）
            import csv
            tag = "_rmix" if args.encoder == "taps" else "_rmix"
            rows = list(csv.DictReader(open(os.path.join(PROJ, f"eval/eval_iar_{args.encoder}{tag}.csv"))))
            full_p = int([r for r in rows if r["scene"] == stem and abs(float(r["level"]) - lv) < 1e-6][0]["lv_pred"])
            print(f"{stem:6s}{lv:5g}{f'{k}px':>10s}{mh.mean():7.3f}{ah.mean():8.3f}"
                  f"{gt_film.mean():8.3f}{pi:6d}   {full_p:8d}{full_film_mean:10.3f}")
            agg[lv]["crop_a"].append(ah.mean()); agg[lv]["crop_p"].append(pi); agg[lv]["full_p"].append(full_p)
    print("\n汇总（预测档位序 [1.0,0.6,0.3] → 0/1/2）:")
    for lv in LEVELS:
        cp, fp = np.array(agg[lv]["crop_p"]), np.array(agg[lv]["full_p"])
        print(f"  真档{lv:g}: 全膜裁片 acc={np.mean(cp == LEVELS.index(lv)):.2f} "
              f"判档分布={np.bincount(cp, minlength=3)} | 全图 acc={np.mean(fp == LEVELS.index(lv)):.2f} "
              f"分布={np.bincount(fp, minlength=3)} | 膜内 α̂ 均值={np.mean(agg[lv]['crop_a']):.3f}")


if __name__ == "__main__":
    main()
