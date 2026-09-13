#!/usr/bin/env python3
"""E1 固定 ᾱ 分档测试集生成器(论文级,seed 固定后可复现)。

布局:
  <out>/gt/<scene>.png       干净 T(全亮度, 32 对齐)
  <out>/r/<scene>.png        所用反射层
  <out>/f/<scene>.npy        α 空间场 f(x) (与 ᾱ 无关)
  <out>/levels/alpha_X/<scene>.png   合成观测 I
  <out>/meta.json            参数与来源记录

与合成器共用物理: I = ᾱ(1+s·f)·T + R + n, R 取自 r_pool 按 r-mean 缩放。
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image

from synth_tinted_degrade import (group_ts, load, save, tint_field,
                                  pick_mid_r, compose)

ALPHAS = [1.0, 0.6, 0.3, 0.15, 0.1, 0.05]
SEED = 20260909


def align32(im):
    a = np.asarray(Image.fromarray((np.clip(im, 0, 1) * 255).round().astype(np.uint8))
                   .resize((832, 384), Image.LANCZOS), np.float32) / 255.0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="augmented 三件套目录")
    ap.add_argument("--r-pool", required=True, help="R 池(如 r_world_soft)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=0, help="0=全部")
    ap.add_argument("--r-mean", type=float, default=0.06)
    ap.add_argument("--spatial", type=float, default=0.08)
    ap.add_argument("--noise", type=float, default=1.5)
    args = ap.parse_args()

    heads = group_ts(args.src)
    if args.scenes:
        heads = heads[:args.scenes]
    pool = sorted(glob.glob(os.path.join(args.r_pool, "*.*")))

    for sub in ["gt", "r", "f"] + [f"levels/alpha_{a:g}" for a in ALPHAS]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    meta = {"alphas": ALPHAS, "seed": SEED, "spatial": args.spatial, "noise": args.noise,
            "r_mean": args.r_mean, "r_pool": args.r_pool, "src": args.src, "scenes": []}
    for si, t_file in enumerate(heads):
        t = load(t_file)
        rng = np.random.default_rng(SEED + si)
        cand = pool[int(rng.integers(0, len(pool)))]
        r = np.asarray(Image.open(cand).convert("RGB").resize(
            (t.shape[1], t.shape[0]), Image.LANCZOS), np.float32) / 255.0
        r = np.clip(r * (args.r_mean / max(r.mean(), 1e-6)), 0.0, 1.0)

        t32 = align32(t)
        r32 = align32(r)
        rng = np.random.default_rng(SEED + 1000 + si)
        f = tint_field(t32.shape[0], t32.shape[1], rng)
        sid = f"t{si:02d}"
        save(t32, os.path.join(args.out, "gt", sid + ".png"))
        save(r32, os.path.join(args.out, "r", sid + ".png"))
        np.save(os.path.join(args.out, "f", sid + ".npy"), f.astype(np.float16))
        for a in ALPHAS:
            out = compose(t32, r32, a, f, args.spatial, [1, 1, 1], args.noise, SEED + si)
            save(out, os.path.join(args.out, "levels", f"alpha_{a:g}", sid + ".png"))
        meta["scenes"].append({"id": sid, "src_t": t_file, "src_r": cand})
        print(f"{sid}: {os.path.basename(t_file)} <- {os.path.basename(cand)}")
    with open(os.path.join(args.out, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=1, ensure_ascii=False)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
