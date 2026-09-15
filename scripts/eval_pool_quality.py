#!/usr/bin/env python3
"""R 池质量核验：残影相关 / 色度分布 / 强度，对照实拍 RDNet R̂。

背景：RRW 池的 frame−GT 提取会把 GT/frame 采集链失配残影混进 R（残差与本场景 GT
负相关、色度随亮度单调升）。本脚本量化三件事，供 v3 色调对齐前后对比：
  ① 残影: corr(R_gray, 本场景 GT_gray) 应→0；对照 corr(R_gray, 其他场景 GT) 应≈0
  ② 色度: chroma=max-min(通道) 的中位/p90 与 随亮度分布；实拍 R̂ 中位≈2/p90≈7 且平坦
  ③ 强度: native R̄ 与入池缩放后 R̄ 分布

用法:
  python3 scripts/eval_pool_quality.py \
      --pools /DataDisk/zmy_code/TintedGlass_Enhance/data/pools/r_rrw_strong \
              /DataDisk/zmy_code/TintedGlass_Enhance/data/pools/r_rrw_strong_v3
"""
import argparse, csv, glob, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_rrw_pool as brp

REAL_GLOBS = ["/home/zmy/vehicle_ssd/rdnet_real426_0909/*/ytmt_ucs_sirs_r.png",
              "/home/zmy/vehicle_ssd/rdnet_real512_0909/*/ytmt_ucs_sirs_r.png",
              "/home/zmy/vehicle_ssd/rdnet_real901_0909/*/ytmt_ucs_sirs_r.png"]


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def chroma_stats(im):
    ch = im.max(axis=2) - im.min(axis=2)
    lum = im.mean(axis=2)
    bins = []
    for b0 in range(0, 256, 32):
        m = (lum >= b0) & (lum < b0 + 32)
        bins.append(float(ch[m].mean()) if m.sum() > 50 else float("nan"))
    return float(np.median(ch)), float(np.percentile(ch, 90)), bins


def load_rgb(p, max_side=1280):
    return brp.downscale(brp.imread_rgb(p), max_side)


def eval_pool(pool_dir, gt_map, max_side):
    meta_path = os.path.join(pool_dir, "meta.csv")
    rows = list(csv.DictReader(open(meta_path)))
    kept = [r for r in rows if r["kept"] == "True"]
    names = [f"{r['sub']}/{r['scene']}" for r in kept]
    gts = []
    for r in kept:
        rg = np.asarray(Image.open(os.path.join(pool_dir, "raw", r["png_raw"])).convert("RGB"),
                        np.float32)
        h, w = rg.shape[:2]
        gt = brp.fit_size(load_rgb(gt_map[f"{r['sub']}/{r['scene']}"], max_side), w, h)
        gts.append(gt)

    own = []
    for i, r in enumerate(kept):
        rg = np.asarray(Image.open(os.path.join(pool_dir, "raw", r["png_raw"])).convert("RGB"),
                        np.float32)
        own.append(corr(rg.mean(axis=2), gts[i].mean(axis=2)))
    own = np.array(own)
    other = []
    for i, r in enumerate(kept):
        rg = np.asarray(Image.open(os.path.join(pool_dir, "raw", r["png_raw"])).convert("RGB"),
                        np.float32)
        h, w = rg.shape[:2]
        js = [(i + k * 7 + 3) % len(kept) for k in range(1, 6)]
        cs = [corr(rg.mean(axis=2),
                   brp.fit_size(gts[j], w, h).mean(axis=2)) for j in js if j != i]
        other.append(float(np.mean(np.abs(cs))))
    other = np.array(other)

    raw_ch, pool_ch = [], []
    for r in kept:
        raw_ch.append(chroma_stats(np.asarray(
            Image.open(os.path.join(pool_dir, "raw", r["png_raw"])).convert("RGB"), np.float32)))
        pool_ch.append(chroma_stats(np.asarray(
            Image.open(os.path.join(pool_dir, "pool", r["png_pool"])).convert("RGB"), np.float32)))
    return dict(name=os.path.basename(pool_dir.rstrip("/")), n=len(kept), names=names,
                own=own, other=other, raw_ch=raw_ch, pool_ch=pool_ch, rows=kept)


def fmt_bins(bs):
    return " ".join("  ." if np.isnan(x) else f"{x:5.1f}" for x in bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/DataDisk/zmy_code/datasets/RRWDataset")
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--worst", type=int, default=5, help="列残影最强的 N 个场景")
    args = ap.parse_args()

    gt_map = {f"{a}/{b}": d for (a, b, c, d) in brp.discover(args.src)}

    results = [eval_pool(p, gt_map, args.max_side) for p in args.pools]

    print("=== ① 残影相关 corr(R_gray, GT_gray) ===")
    print(f"{'pool':<18} {'n':>3} {'own mean':>9} {'own med':>8} {'own min':>8} {'own max':>8} "
          f"{'|other| mean':>12}")
    for r in results:
        print(f"{r['name']:<18} {r['n']:>3} {r['own'].mean():>9.3f} {np.median(r['own']):>8.3f} "
              f"{r['own'].min():>8.3f} {r['own'].max():>8.3f} {r['other'].mean():>12.3f}")
    for r in results:
        order = np.argsort(r["own"])
        print(f"\n[{r['name']}] 残影最强 {args.worst}: " + ", ".join(
            f"{r['names'][i]}={r['own'][i]:.3f}" for i in order[:args.worst]))

    print("\n=== ② 色度 chroma=max-min(通道) ===")
    print(f"{'source':<24} {'median':>7} {'p90':>7}   bins(lum 0-32,32-64,...)")
    real = []
    for g in REAL_GLOBS:
        for p in sorted(glob.glob(g)):
            real.append(chroma_stats(load_rgb(p, args.max_side)))
    if real:
        rmed = [x[0] for x in real]
        rp90 = [x[1] for x in real]
        rbin = np.nanmean(np.array([x[2] for x in real]), axis=0)
        print(f"{'real RDNet R-hat (' + str(len(real)) + ')':<24} {np.median(rmed):>7.1f} "
              f"{np.median(rp90):>7.1f}   {fmt_bins(rbin)}")
    for r in results:
        for tag, key in [("raw", "raw_ch"), ("scaled", "pool_ch")]:
            med = [x[0] for x in r[key]]
            p90 = [x[1] for x in r[key]]
            b = np.nanmean(np.array([x[2] for x in r[key]]), axis=0)
            print(f"{r['name'] + ' ' + tag:<24} {np.median(med):>7.1f} {np.median(p90):>7.1f}   {fmt_bins(b)}")

    print("\n=== ③ 强度 (灰阶) ===")
    for r in results:
        nat = np.array([float(x["r_mean"]) for x in r["rows"]])
        sc = np.array([float(x["r_mean_scaled"]) for x in r["rows"]])
        print(f"{r['name']:<18} native 中位={np.median(nat):5.1f} [{nat.min():.1f},{nat.max():.1f}]  "
              f"入池 中位={np.median(sc):5.1f} [{sc.min():.1f},{sc.max():.1f}]")


if __name__ == "__main__":
    main()
