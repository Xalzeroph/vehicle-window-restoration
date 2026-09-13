#!/usr/bin/env python3
"""E1 评测: 对固定测试集的每张预测算 PSNR/SSIM(相对全亮度 GT),按 ᾱ 档汇总。

用法:
  python3 eval_e1.py --gt <e1>/gt --pred-root <pred_root> --meta <e1>/meta.json
布局: <pred_root>/<model>_alpha_X/<scene>/...含 sirs_l 图 或 <scene>.png
输出: stdout 表 + <pred_root>/../eval_<model>.csv
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn


def load(f):
    return np.asarray(Image.open(f).convert("RGB"), np.float32)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def find_pred(pred_root, scene, level_dir):
    stem = os.path.splitext(scene)[0]
    base = os.path.join(pred_root, level_dir)
    d = os.path.join(base, stem)
    if os.path.isdir(d):
        cands = [f for f in os.listdir(d) if f.endswith(".png") and "sirs_l" in f]
        if not cands:
            cands = [f for f in os.listdir(d) if f.endswith(".png") and "input" not in f]
        return os.path.join(d, cands[0]) if cands else None
    p = os.path.join(base, scene)
    return p if os.path.exists(p) else None


def gain_match(q, g):
    """最优乘性匹配 q ← g·q, g 由最小二乘给出(把增益误差与内容误差分开)。"""
    g = np.asarray(g, np.float64)
    q = np.asarray(q, np.float64)
    scale = float((q * g).sum() / (q * q).sum())
    return np.clip(q * scale, 0, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", default="rdnet")
    ap.add_argument("--gain-match", action="store_true",
                    help="对每张预测/输入先做最优乘性匹配再算 PSNR/SSIM(指标③ 增益不变对照)")
    ap.add_argument("--ref-input", action="store_true",
                    help="同时评输入图本身(配 --gain-match 作对照下界)")
    args = ap.parse_args()

    with open(args.meta) as fp:
        meta = json.load(fp)
    scenes = sorted(glob.glob(os.path.join(args.gt, "*.png")))
    level_dirs = sorted(os.listdir(os.path.join(args.pred_root)),
                        key=lambda s: float(s.split("_")[-1]))
    tag = "_gm" if args.gain_match else ""
    rows = []
    print(f"{'level':>9} {'PSNR':>6} {'SSIM':>6}  (n)")
    for lv in level_dirs:
        ps, ss = [], []
        ps_in, ss_in = [], []
        for gt_f in scenes:
            scene = os.path.basename(gt_f)
            p = find_pred(args.pred_root, scene, lv)
            if p is None:
                continue
            g = load(gt_f)
            q = load(p)
            if q.shape[:2] != g.shape[:2]:
                g = np.asarray(Image.fromarray(g.astype(np.uint8)).resize(
                    (q.shape[1], q.shape[0])), np.float32)
            if args.gain_match:
                q = gain_match(q, g)
            ps.append(psnr(q, g))
            ss.append(ssim_fn(q, g, channel_axis=2, data_range=255))
            rows.append({"level": lv, "scene": scene,
                         "psnr": round(ps[-1], 3), "ssim": round(ss[-1], 4)})
            if args.ref_input:
                stem = os.path.splitext(scene)[0]
                inp = load(os.path.join(args.pred_root, lv, stem, "m_input.png"))
                if inp.shape[:2] != g.shape[:2]:
                    inp = np.asarray(Image.fromarray(inp.astype(np.uint8)).resize(
                        (g.shape[1], g.shape[0])), np.float32)
                inp = gain_match(inp, g)
                ps_in.append(psnr(inp, g))
                ss_in.append(ssim_fn(inp, g, channel_axis=2, data_range=255))
        line = f"{lv:>9} {np.mean(ps):6.2f} {np.mean(ss):6.3f}  ({len(ps)})"
        if args.ref_input:
            line += f"   input_gm {np.mean(ps_in):6.2f} {np.mean(ss_in):6.3f}"
        print(line)
    csvp = os.path.join(os.path.dirname(args.pred_root.rstrip("/")),
                        f"eval_{args.model}{tag}.csv")
    with open(csvp, "w") as fp:
        fp.write("level,scene,psnr,ssim\n")
        for r in rows:
            fp.write(f"{r['level']},{r['scene']},{r['psnr']},{r['ssim']}\n")
    print("csv ->", csvp)


if __name__ == "__main__":
    main()
