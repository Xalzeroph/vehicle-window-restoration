#!/usr/bin/env python3
"""分离器评测补充面板：常数基线（GT 均值图）+ 结构相关 corr(去均值)。

动机：v2.1/v2.2 的 GT 低对比（σ≈34/255），α 低档时 PSNR/SSIM 会被"平坦图"占据
（常数基线 17.68 dB / SSIM 0.742 反而高于各模型），需常数基线与结构相关共同解读。

常数基线不需要模型（只用 GT 与 levels 图）；corr 读 runs/pred/rdt_<tag>/ 下已存 t_hat。
用法: python3 scripts/eval_rdt_extra.py [--tags freeze,plain,...] [--sets ...]
输出: eval/eval_rdt_extra.csv + 控制台表
"""
import argparse, csv, os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SETS = ["synth_rrw_v21_0913", "synth_rrw_v21_indomain", "synth_rrw_v22_scan"]
DEFAULT_TAGS = ["zeroshot", "pilotB_r2", "freeze", "freeze_s101", "plain", "plain_s101"]


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / ((a - b) ** 2).mean())


def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    ap.add_argument("--sets", default=",".join(DEFAULT_SETS))
    args = ap.parse_args()
    rows = []
    for s in args.sets.split(","):
        root = os.path.join(PROJ, "data/test_sets", s)
        scenes = sorted(os.listdir(os.path.join(root, "gt")))
        levels = sorted(os.listdir(os.path.join(root, "levels")),
                        key=lambda x: -float(x.split("_")[-1]))
        for lv in levels:
            gts = {sc: img(os.path.join(root, "gt", sc)) for sc in scenes}
            p_const = np.mean([psnr(np.full_like(g, g.mean()), g) for g in gts.values()])
            s_const = np.mean([ssim_fn(np.full_like(g, g.mean()), g, channel_axis=2, data_range=255)
                               for g in gts.values()])
            rows.append(dict(set=s, level=lv, tag="const_gt_mean", n=len(scenes),
                             psnr=round(float(p_const), 3), ssim=round(float(s_const), 4), corr=0.0))
            for tag in args.tags.split(","):
                d = os.path.join(PROJ, "runs/pred", f"rdt_{tag}", s, lv)
                if not os.path.isdir(d):
                    continue
                cs, ok = [], True
                for sc in scenes:
                    p = os.path.join(d, os.path.splitext(sc)[0], "t_hat.png")
                    if not os.path.exists(p):
                        ok = False; break
                    cs.append(corr(img(p), gts[sc]))
                if ok:
                    rows.append(dict(set=s, level=lv, tag=tag, n=len(scenes),
                                     psnr="", ssim="", corr=round(float(np.mean(cs)), 4)))
    for r in rows:
        print(f"{r['set']:>24} {r['level']:>11} {r['tag']:>12}  psnr={r['psnr']} ssim={r['ssim']} corr={r['corr']}")
    out = os.path.join(PROJ, "eval/eval_rdt_extra.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["set", "level", "tag", "n", "psnr", "ssim", "corr"])
        w.writeheader(); w.writerows(rows)
    print(f"-> {os.path.relpath(out, PROJ)}")


if __name__ == "__main__":
    main()
