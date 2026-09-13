#!/usr/bin/env python3
"""E2 oracle 三对照（v2.1 基准）：量化"空间 α 建模"的得分空间。

对照列（口径与 v2 首跑 2026-09-13 表一致）:
  输入 I             : PSNR(I, 全亮 GT)
  完美去R+全局增益¹   : I−R 后最优全局乘性匹配（无空间 α 建模的理想值）
  完美去R+全局ᾱ      : (I−R)/mean(α(x))，单尺度 α（真值均值，全局 α 上限）
  完美去R+精确α(x)²  : (I−R)/α(x) 用真值逐像素 α，仅统计 a≥0.05 像素
  RDNet / RDNet(GM) / 输入(GM): 读 pred-root 的 ytmt_ucs_sirs_l.png

用法:
  python3 scripts/eval_e2_oracle.py --testset synth_rrw_v21_0913 \
      --pred-root synth_rrw_v21_pred --model v21 --plot E2_oracle_curve_v21.png
"""
import argparse, glob, os
import numpy as np
from PIL import Image


def load255(f):
    return np.asarray(Image.open(f).convert("RGB"), np.float32)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def gain_match(q, g):
    q64, g64 = q.astype(np.float64), g.astype(np.float64)
    s = float((q64 * g64).sum() / (q64 * q64).sum())
    return np.clip(q64 * s, 0, 255).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--model", default="v21")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    lv_dirs = sorted(os.listdir(os.path.join(args.testset, "levels")),
                     key=lambda s: -float(s.split("_")[-1]))
    gts = sorted(glob.glob(os.path.join(args.testset, "gt", "*.png")))
    keys = ["I", "noA", "gA", "exA", "rd", "rdgm", "ingm"]
    print(f"{'level':>5} {'I':>6} {'I-R+gm':>7} {'/a_bar':>7} {'/a(x)2':>7} "
          f"{'RDNet':>6} {'RDGM':>6} {'inGM':>6}  maskf")
    rows = []
    for lv in lv_dirs:
        akey = lv.split("_", 1)[1]
        acc = {k: [] for k in keys}
        maskf = []
        for gt_f in gts:
            sid = os.path.splitext(os.path.basename(gt_f))[0]
            T = load255(gt_f)
            I = load255(os.path.join(args.testset, "levels", lv, sid + ".png"))
            R = load255(os.path.join(args.testset, "r", sid + ".png"))
            A = np.load(os.path.join(args.testset, "alpha", f"{sid}_a{akey}.npy")).astype(np.float32)
            q = I - R  # = αT + n（0-255 域）
            acc["I"].append(psnr(I, T))
            acc["noA"].append(psnr(gain_match(q, T), T))
            acc["gA"].append(psnr(np.clip(q / float(A.mean()), 0, 255), T))
            m = A >= 0.05
            maskf.append(float(m.mean()))
            qe = np.clip(q / A[..., None], 0, 255)
            acc["exA"].append(psnr(qe[m], T[m]))
            P = load255(os.path.join(args.pred_root, "rdnet_" + lv, sid, "ytmt_ucs_sirs_l.png"))
            acc["rd"].append(psnr(P, T))
            acc["rdgm"].append(psnr(gain_match(P, T), T))
            acc["ingm"].append(psnr(gain_match(I, T), T))
        r = {k: float(np.mean(v)) for k, v in acc.items()}
        print(f"{float(akey):>5g} {r['I']:6.2f} {r['noA']:7.2f} {r['gA']:7.2f} {r['exA']:7.2f} "
              f"{r['rd']:6.2f} {r['rdgm']:6.2f} {r['ingm']:6.2f}  {np.mean(maskf):.3f}")
        rows.append((float(akey), r))

    csvp = os.path.join(os.path.dirname(args.testset.rstrip("/")), f"eval_e2_oracle_{args.model}.csv")
    with open(csvp, "w") as fp:
        fp.write("level," + ",".join(keys) + "\n")
        for lvl, r in rows:
            fp.write(f"{lvl:g}," + ",".join(f"{r[k]:.3f}" for k in keys) + "\n")
    print("csv ->", csvp)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            x = [lv for lv, _ in rows]
            fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=160)
            for k, lab, st in [("ingm", "Input (GM)", "--"),
                               ("noA", "Perfect R removal + global gain", "-"),
                               ("gA", "Perfect R removal + global α (mean)", ":"),
                               ("exA", "Perfect R removal + exact α(x)", "-"),
                               ("rdgm", "RDNet (GM)", "-"), ("rd", "RDNet (raw)", "--")]:
                ax.plot(x, [r[k] for _, r in rows], st, marker="o", markersize=3.5, label=lab)
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}" for v in x])
            ax.set_xlabel("α (nominal level)")
            ax.set_ylabel("PSNR (dB) vs full-brightness GT")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(args.plot)
            print("plot ->", args.plot)
        except Exception as e:
            print("plot skipped:", e)


if __name__ == "__main__":
    main()
