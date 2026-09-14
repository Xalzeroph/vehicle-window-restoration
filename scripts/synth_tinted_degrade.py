#!/usr/bin/env python3
"""贴膜退化合成器（E1 pilot）：I = α_c(x)·T + R + n

T/R 取自 vehicle_ssd/datasets/augmented 三件套（label=干净透射层 T，reflection=反射层 R），
α_c(x) = ᾱ · (1 + s·f(x)) · cast_c，f(x) 为每场景共享的低频平滑空间场（贴膜不均匀/渐变），
R 不经衰减直接叠加（表面反射物理），n 可选高斯噪声。

用法示例：
  python3 scripts/synth_tinted_degrade.py \
      --src /home/zmy/vehicle_ssd/datasets/augmented \
      --out /home/zmy/vehicle_ssd/synth_pilot_0909 \
      --scenes 3 --alphas 1.0,0.6,0.3,0.15,0.1,0.05 --spatial 0.3 --noise 2
"""
import argparse, glob, hashlib, csv, os
import numpy as np
from PIL import Image, ImageDraw

COLORS = [(255, 200, 90), (120, 200, 120), (255, 120, 120), (120, 160, 255),
          (200, 120, 255), (255, 160, 120), (160, 255, 200), (255, 255, 160)]


def load(f):
    return np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255.0


def save(im, p):
    Image.fromarray((np.clip(im, 0, 1) * 255).round().astype(np.uint8)).save(p, quality=92)


def smooth_field(h, w, rng, n_cells=22):
    gh, gw = max(2, h // n_cells), max(2, w // n_cells)
    g = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    f = np.asarray(Image.fromarray(g).resize((w, h), Image.BILINEAR), dtype=np.float32)
    return (f - f.mean()) / (f.std() + 1e-6)


def tint_field(h, w, rng):
    """贴膜不均匀场:60% 缓变线性渐变(工艺方向) + 40% 细粒度平滑细节, 单位方差。"""
    lin = (rng.random() * 2 - 1) * np.linspace(-1, 1, h)[:, None]
    lin = lin + (rng.random() * 2 - 1) * np.linspace(-1, 1, w)[None, :]
    lin = (lin - lin.mean()) / (lin.std() + 1e-6)
    blob = smooth_field(h, w, rng)
    f = 0.6 * lin + 0.4 * blob
    return (f - f.mean()) / (f.std() + 1e-6)


def coverage_mask(h, w, rng, frac=(0.78, 0.92), trans_px=(0.8, 1.8)):
    """黑膜覆盖掩膜: 方向随机的平滑边界曲线 + ~4-8px 10-90% 过渡(抗锯齿的突变膜边)。
    frac 为覆盖像素占比, 边界带 1-3 个低频正弦扰动。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = rng.uniform(0, 2 * np.pi)
    nx, ny = np.cos(th), np.sin(th)
    proj = nx * xx + ny * yy
    tang = -ny * xx + nx * yy
    amp = rng.uniform(0.05, 0.11) * min(h, w)
    pert = np.zeros((h, w), np.float32)
    for k in range(int(rng.integers(2, 4))):
        fr = rng.uniform(0.8, 2.0) * 2 * np.pi / max(h, w)
        pert += np.sin(fr * tang + rng.uniform(0, 2 * np.pi)) * amp / (k + 1)
    d = np.percentile(proj + pert, (1.0 - rng.uniform(*frac)) * 100.0)
    return 1.0 / (1.0 + np.exp(-(proj + pert - d) / rng.uniform(*trans_px)))


def arc_gradient(h, w, rng, n_cells=56):
    """玻璃弧度/光照缓变场: 径向弧状渐变为主 + 大尺度平滑场为辅, 单位方差(无斑块感)。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = rng.uniform(-0.5, 1.5) * w
    cy = rng.uniform(-0.5, 1.5) * h
    r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    g1 = (r - r.mean()) / (r.std() + 1e-6)
    g2 = smooth_field(h, w, rng, n_cells=n_cells)
    f = 0.75 * g1 + 0.25 * g2
    return (f - f.mean()) / (f.std() + 1e-6)


def alpha_from(m, f, alpha_bar, s):
    """α(x) = 膜内 ᾱ(1+s·f) + 膜外 1.0(裸玻璃)。"""
    return np.clip(alpha_bar * (1.0 + s * f) * m + (1.0 - m), 0.01, 1.0)


def r_overlay_field(h, w, rng, s_r=0.30, n_cells=64, n_hot=(2, 5)):
    """R 非均匀叠加场: 大尺度渐变衰减(乘性 1+s_r·g) + 金属高光热点(高斯峰 0.3-0.9)。"""
    f = 1.0 + s_r * smooth_field(h, w, rng, n_cells=n_cells)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(*n_hot))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sig = rng.uniform(0.10, 0.28) * min(h, w)
        f = f + rng.uniform(0.3, 0.9) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig * sig))
    return np.clip(f, 0.0, 2.2)


def compose_alpha(t, r, alpha, noise, seed, r_couple=0.0):
    """I = α(x)·T + α(x)^p·R + n。p=0: 反射不随膜衰减; p=1: 反射占比与 α 无关。"""
    rng = np.random.default_rng(seed)
    out = alpha[..., None] * t + (alpha[..., None] ** r_couple) * r
    if noise > 0:
        out += rng.normal(0, noise / 255.0, out.shape)
    return out


def group_ts(src):
    labels = sorted(glob.glob(os.path.join(src, "a*_label.jpg")))
    groups, head = {}, {}
    for f in labels:
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        groups.setdefault(h, []).append(f)
        head.setdefault(h, f)
    return [head[h] for h in groups]


def pick_mid_r(t_file, rng):
    pre = t_file[:-len("_label.jpg")]
    rs = sorted(glob.glob(pre + "_reflection.jpg"))
    if len(rs) <= 1:
        return rs[0] if rs else None
    ms = sorted(((float(np.asarray(Image.open(r), dtype=np.float32)).mean()), r) for r in rs)
    return ms[len(ms) // 2][1]


def compose(t, r, alpha_bar, f, s, cast, noise, seed):
    rng = np.random.default_rng(seed)
    a = alpha_bar * (1.0 + s * f)[..., None] * np.array(cast, np.float32)[None, None, :]
    a = np.clip(a, 0.01, 1.0)
    out = a * t + r
    if noise > 0:
        out += rng.normal(0, noise / 255.0, out.shape)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--alphas", type=str, default="1.0,0.6,0.3,0.15,0.1,0.05")
    ap.add_argument("--spatial", type=float, default=0.08, help="空间场强度 s(缓变,~0.08 即可)")
    ap.add_argument("--r-pool", default=None, help="真实残差 R 池目录(build_real_r_pool 产物);缺省用 augmented 自带 R")
    ap.add_argument("--r-mean", type=float, default=0.06, help="R 缩放目标均值(真实残差 ~0.04-0.06)")
    ap.add_argument("--cast", type=str, default="1,1,1", help="逐通道色偏乘子,如 1,0.95,0.9")
    ap.add_argument("--noise", type=float, default=0, help="高斯噪声 σ(0-255 域)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    cast = [float(x) for x in args.cast.split(",")]
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    heads = group_ts(args.src)
    idx = np.linspace(0, len(heads) - 1, args.scenes).round().astype(int)
    scenes = [heads[i] for i in idx]

    rows = []
    pool = sorted(glob.glob(os.path.join(args.r_pool, "*.*"))) if args.r_pool else None
    for si, t_file in enumerate(scenes):
        t = load(t_file)
        rng = np.random.default_rng(args.seed + si)
        if pool:
            cand = pool[int(rng.integers(0, len(pool)))]
            r = np.asarray(Image.open(cand).convert("RGB").resize(
                (t.shape[1], t.shape[0]), Image.LANCZOS), np.float32) / 255.0
            r = np.clip(r * (args.r_mean / max(r.mean(), 1e-6)), 0.0, 1.0)
            r_file = cand
        else:
            r_file = pick_mid_r(t_file, rng)
            r = load(r_file)
        h, w = t.shape[:2]
        f = tint_field(h, w, rng)
        base = t_file.replace("_label.jpg", "")
        save(r, os.path.join(args.out, f"{os.path.basename(base)}_r.jpg"))
        print(f"\n[scene {si}] {base}  T mean={t.mean():.3f}  R mean={r.mean():.3f} std={r.std():.3f}  <- {os.path.basename(r_file)}")
        for a in alphas:
            out = compose(t, r, a, f, args.spatial, cast, args.noise, args.seed + si)
            p = os.path.join(args.out, f"{os.path.basename(base)}_a{a:g}_s{si}.jpg")
            save(out, p)
            rows.append(dict(scene=os.path.basename(base), alpha=a, t_mean=round(float(t.mean()), 4),
                             r_mean=round(float(r.mean()), 4), i_mean=round(float(out.mean()), 4),
                             i_std=round(float(out.std()), 4),
                             clip_hi=round(float((out > 0.98).mean()), 4),
                             clip_lo=round(float((out < 0.02).mean()), 4)))
            print(f"  ᾱ={a:>4}: I mean={out.mean():.3f} std={out.std():.3f} "
                  f"clip_hi={(out>0.98).mean()*100:.1f}%")

    with open(os.path.join(args.out, "stats.csv"), "w", newline="") as fp:
        cw = csv.DictWriter(fp, fieldnames=list(rows[0]))
        cw.writeheader()
        cw.writerows(rows)

    make_montage(scenes, alphas, args.out, args.spatial, cast)


def make_montage(scenes, alphas, out, s, cast):
    import math
    import re
    cols = ["T", "R"] + [f"ᾱ={a:g}" for a in alphas]
    cw, label_h, m = 168, 26, 12  # cell width, label band, margin
    pad = 4
    sfiles = []
    for f in scenes:
        t_file = load(f)
        rp = os.path.join(out, os.path.basename(f).replace("_label.jpg", "") + "_r.jpg")
        r_im = load(rp) if os.path.exists(rp) else load(pick_mid_r(f, np.random.default_rng(0)))
        sf = [t_file, r_im]
        for a in alphas:
            b = os.path.basename(f).replace("_label.jpg", "")
            sf.append(load(os.path.join(out, f"{b}_a{a:g}_s{scenes.index(f)}.jpg")))
        sfiles.append(sf)
    ch = max(8, int(cw * 386 / 847))  # 按已知 H/W 比例缩放高度
    total_w = cw * len(cols) + (len(cols) + 1) * pad
    total_h = label_h * 2 + len(scenes) * (ch + label_h) + (len(scenes) + 1) * pad
    sheet = Image.new("RGB", (total_w, total_h), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for ci, name in enumerate(cols):
        d.rectangle([pad + ci * (cw + pad), pad, pad + ci * (cw + pad) + cw, pad + label_h], fill=(60, 60, 70))
        d.text((pad + ci * (cw + pad) + 6, pad + 6), name, fill=(255, 255, 255))
    for ri, row in enumerate(sfiles):
        y0 = label_h + pad + ri * (ch + label_h + pad)
        for ci, im in enumerate(row):
            x0 = pad + ci * (cw + pad)
            cell = Image.fromarray((np.clip(im, 0, 1) * 255).round().astype(np.uint8)).resize((cw, ch))
            sheet.paste(cell, (x0, y0))
            lbl = f"m={im.mean()*255:.0f}"
            d.rectangle([x0, y0 + ch, x0 + cw, y0 + ch + label_h], fill=(45, 45, 55))
            d.text((x0 + 5, y0 + ch + 6), lbl, fill=(230, 230, 230))
    p = os.path.join(out, "overview.png")
    sheet.save(p)
    print(f"\nmontage -> {p}")


if __name__ == "__main__":
    main()
