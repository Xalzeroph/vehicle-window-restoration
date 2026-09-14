#!/usr/bin/env python3
"""Pilot 训练对生成器: 与基准 v2.1 同分布的空间 α 合成 (I = α(x)·T + R + n)。

与基准 v2.1 一致的要素:
  - α 场: coverage_mask(突变膜边) + arc_gradient(弧度缓变), s=0.3, 膜外 α=1
  - R: r_rrw_strong/pool (ρ 已烘焙, 不再归一化) × r_overlay_field 非均匀叠加, 色度压缩 k=0.45
  - 噪声 σ=1.5(0-255 域); 三档 ᾱ=1.0/0.6/0.3 均衡轮转(各 n/3)
  - 输出 I/T/R_eff uint8 + 真值 α(x) float16 npy; 饱和掩膜不落盘(I>=254 现算)

隔离: R 池排除 v2.1 测试集 11 个场景(含其 _alt); T 源 SIR2/nature/scenes 与测试
T(自 augmented 派生)已核验无近重复内容。

用法:
  python3 scripts/build_train_pairs.py --out train_pairs_pilot --n 2400
  python3 scripts/build_train_pairs.py --out train_pairs_pilot --preview 6   # 只出目检 sheet
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image, ImageDraw

from synth_tinted_degrade import load, save, coverage_mask, arc_gradient, \
    alpha_from, r_overlay_field, compose_alpha

LEVELS = [1.0, 0.6, 0.3]
DEFAULT_T_SRC = [
    ("sir2", "/home/zmy/mycode/datasets/SIR2/SIR/transmission_layer"),
    ("nature", "/home/zmy/mycode/datasets/nature/natural_T"),
    ("scenes", "/home/zmy/vehicle_ssd/datasets/scenes"),
]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def glob_images(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*"))
                  if os.path.splitext(f)[1].lower() in IMG_EXT)


def desat(r, k):
    if k >= 0.999:
        return r
    g = r.mean(axis=2, keepdims=True)
    return np.clip(g + k * (r - g), 0.0, 1.0)


def crop_square(im, size, rng):
    h, w = im.shape[:2]
    if min(h, w) < size:
        sc = size / min(h, w)
        im = np.asarray(Image.fromarray((im * 255).round().astype(np.uint8)).resize(
            (max(size, int(round(w * sc))), max(size, int(round(h * sc)))), Image.LANCZOS),
            np.float32) / 255.0
        h, w = im.shape[:2]
    y0 = int(rng.integers(0, h - size + 1))
    x0 = int(rng.integers(0, w - size + 1))
    return im[y0:y0 + size, x0:x0 + size]


def test_scene_ids(meta_path):
    if not os.path.exists(meta_path):
        return set()
    with open(meta_path) as fp:
        meta = json.load(fp)
    return {os.path.basename(s["src_r"])[:-4] for s in meta["scenes"]}


def pool_files(pool_dir, exclude_ids):
    keep = []
    for f in sorted(glob.glob(os.path.join(pool_dir, "*.png"))):
        stem = os.path.basename(f)[:-4]
        sid = stem[:-4] if stem.endswith("_alt") else stem
        if sid in exclude_ids:
            continue
        keep.append(f)
    return keep


def synth_one(t_src, r_src, level, rng, size, s, noise, r_sat, cover_frac, noise_seed):
    t = crop_square(load(t_src), size, rng)
    r = crop_square(load(r_src), size, rng)
    m = coverage_mask(size, size, rng, frac=tuple(cover_frac))
    f_arc = arc_gradient(size, size, rng)
    a = alpha_from(m, f_arc, level, s)
    r_eff = desat(r * r_overlay_field(size, size, rng)[..., None], r_sat)
    i = compose_alpha(t, r_eff, a, noise, noise_seed)
    return t, r_eff, a, i


def make_sheet(rows, path, cw=224):
    pad, lab = 5, 22
    ncol = len(rows[0])
    W = pad + ncol * (cw + pad)
    H = pad + len(rows) * (cw + lab + pad)
    S = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(S)
    for ri, row in enumerate(rows):
        y0 = pad + ri * (cw + lab + pad)
        for ci, (im, name) in enumerate(row):
            x0 = pad + ci * (cw + pad)
            im8 = (np.clip(im, 0, 1) * 255).round().astype(np.uint8)
            S.paste(Image.fromarray(im8), (x0, y0))
            d.text((x0 + 4, y0 + cw + 4), name, fill=(230, 230, 230))
    S.save(path)
    print("sheet ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/DataDisk/zmy_code/TintedGlass_Enhance/train_pairs_pilot")
    ap.add_argument("--n", type=int, default=2400, help="样本数(按 1:1:1 轮转三档)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--spatial", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--r-sat", type=float, default=0.45)
    ap.add_argument("--cover-frac", type=float, nargs=2, default=[0.78, 0.92])
    ap.add_argument("--r-pool", default="/DataDisk/zmy_code/TintedGlass_Enhance/r_rrw_strong/pool")
    ap.add_argument("--test-meta", default="/DataDisk/zmy_code/TintedGlass_Enhance/synth_rrw_v21_0913/meta.json")
    ap.add_argument("--t-weights", default="0.35,0.25,0.40", help="对应 sir2,nature,scenes 抽样权重")
    ap.add_argument("--preview", type=int, default=0, help="只出目检 sheet(行=样本)后退出")
    args = ap.parse_args()

    t_weights = np.array([float(x) for x in args.t_weights.split(",")])
    t_weights = t_weights / t_weights.sum()
    t_files = {name: glob_images(d) for name, d in DEFAULT_T_SRC}
    for name, fs in t_files.items():
        print(f"T[{name}]: {len(fs)}")
        assert fs, f"T 源为空: {name}"
    r_files = pool_files(args.r_pool, test_scene_ids(args.test_meta))
    print(f"R pool: {len(r_files)} (已排除测试场景)")

    def sample_sources(idx, rng):
        ti = int(rng.choice(len(DEFAULT_T_SRC), p=t_weights))
        t_name = DEFAULT_T_SRC[ti][0]
        t_f = t_files[t_name][int(rng.integers(0, len(t_files[t_name])))]
        r_f = r_files[int(rng.integers(0, len(r_files)))]
        return t_name, t_f, r_f

    if args.preview:
        rng0 = np.random.default_rng(args.seed)
        for sub in ["I", "T", "R", "alpha"]:
            os.makedirs(os.path.join(args.out, sub), exist_ok=True)
        rows = []
        for i in range(args.preview):
            rng = np.random.default_rng(args.seed * 1000003 + i)
            lv = LEVELS[i % len(LEVELS)]
            t_name, t_f, r_f = sample_sources(i, rng)
            t, r, a, im = synth_one(t_f, r_f, lv, rng, args.size, args.spatial,
                                    args.noise, args.r_sat, args.cover_frac, args.seed + 999 + i)
            tag = f"{os.path.basename(t_f)[:16]} | {os.path.basename(r_f)[:16]}"
            a_img = (np.repeat(np.clip(a, 0, 1)[..., None], 3, 2) * 255).round().astype(np.uint8)
            a_img = a_img.astype(np.float32) / 255.0
            rows.append([(t, f"T {tag}"), (im, f"I a={lv:g}"),
                         (r, "R_eff"), (a_img, "alpha GT")])
        make_sheet(rows, os.path.join(os.path.dirname(args.out.rstrip("/")),
                                      os.path.basename(args.out.rstrip("/")) + "_preview.png"))
        return

    for sub in ["I", "T", "R", "alpha"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    recs = []
    for i in range(args.n):
        rng = np.random.default_rng(args.seed * 1000003 + i)
        lv = LEVELS[i % len(LEVELS)]
        t_name, t_f, r_f = sample_sources(i, rng)
        t, r, a, im = synth_one(t_f, r_f, lv, rng, args.size, args.spatial,
                                args.noise, args.r_sat, args.cover_frac, args.seed + 999 + i)
        sid = f"{i:06d}"
        save(im, os.path.join(args.out, "I", sid + ".png"))
        save(t, os.path.join(args.out, "T", sid + ".png"))
        save(r, os.path.join(args.out, "R", sid + ".png"))
        np.save(os.path.join(args.out, "alpha", sid + ".npy"), a.astype(np.float16))
        clip_frac = float((im > 1.0 - 1e-6).mean())
        recs.append({"id": sid, "level": lv, "t_src": f"{t_name}:{os.path.basename(t_f)}",
                     "r_src": os.path.basename(r_f), "clip_frac": clip_frac,
                     "alpha_eff": float(a.mean()), "r_mean": float(r.mean())})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{args.n}")

    meta = {"n": args.n, "size": args.size, "seed": args.seed, "levels": LEVELS,
            "spatial": args.spatial, "noise": args.noise, "r_sat": args.r_sat,
            "cover_frac": args.cover_frac, "r_pool": args.r_pool,
            "r_pool_n": len(r_files), "test_meta": args.test_meta,
            "t_weights": list(t_weights),
            "physics": "v2.1: I = alpha(x)*T + R_eff + n (p=0)",
            "samples": recs}
    with open(os.path.join(args.out, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=1, ensure_ascii=False)
    lv_arr = np.array([r["level"] for r in recs])
    cl = np.array([r["clip_frac"] for r in recs])
    print(f"done -> {args.out}  n={args.n}  levels={dict(zip(*np.unique(lv_arr, return_counts=True)))}")
    print(f"clip_frac: mean={cl.mean() * 100:.2f}% max={cl.max() * 100:.2f}%  "
          f"饱和像素(>=254)>0 的样本: {(cl > 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
