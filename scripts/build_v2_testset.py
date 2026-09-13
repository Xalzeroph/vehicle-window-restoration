#!/usr/bin/env python3
"""基准 v2.1 验证集: RRW 强反射池 + 覆盖突变边界/弧度渐变 α 场 + R 非均匀叠加。

与 v2 (/synth_rrw_v2_0913) 的区别(2026-09-13 实拍修正落地, 见 9.9 补记 §5):
  - α 场: 废小尺度斑块场, 改为「黑膜覆盖不完全的突变边界(2-4.5px 过渡) +
    膜内玻璃弧度/光照缓变(大尺度低频)」; 膜外 α=1.0(裸玻璃)
  - R: 非均匀叠加(大尺度渐变衰减 × [0,2.2] + 金属高光高斯热点), 不再全局均匀
  - 真值 α(x) 图逐档落盘(alpha/{sid}_a{a}.npy), 供 α 头训练/误差-vs-α 分析
  - 定案(2026-09-13 用户目检): α 三档 [1.0,0.6,0.3] (0.05-0.15 人眼不可辨, 判死);
    R 色度压缩 k=0.45(≈实拍最强 R̂ 色度); R 不随 α 衰减 p=0(表面反射物理)
  - 其余一致: 12 个 T 场景复用 E1; ρ 已烘焙不再归一化; R 保比裁剪; s=0.3; σ=1.5

布局: gt/ r/ alpha/ levels/alpha_X/ meta.json(alpha_eff/r_obs/clip_frac)
用法: python3 scripts/build_v2_testset.py --out <dir>
      python3 scripts/build_v2_testset.py --preview 4   # 只出目检 sheet
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image, ImageDraw

from synth_tinted_degrade import load, save, tint_field, compose_alpha, coverage_mask, \
    arc_gradient, alpha_from, r_overlay_field

ALPHAS = [1.0, 0.6, 0.3]
SEED = 20260913


def fit_r(path, w, h):
    """保比中心裁剪到目标长宽比再 LANCZOS 缩放(整池同宽高比时等价于直接 resize)。"""
    im = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    ih, iw = im.shape[:2]
    tar = w / h
    cur = iw / ih
    if abs(cur - tar) > 1e-3:
        if cur > tar:
            nw = int(round(ih * tar))
            x0 = (iw - nw) // 2
            im = im[:, x0:x0 + nw]
        else:
            nh = int(round(iw / tar))
            y0 = (ih - nh) // 2
            im = im[y0:y0 + nh, :]
    return np.asarray(Image.fromarray((im * 255).round().astype(np.uint8)).resize(
        (w, h), Image.LANCZOS), np.float32) / 255.0


def gray(a):
    return np.repeat(a[..., None].astype(np.float32), 3, axis=2)


def scene_fields(t32, pool, si, cover_frac):
    """同场景共享的 R 叠加/覆盖掩膜/弧度场(各档复用, 保证档间只有 ᾱ 变量)。"""
    h, w = t32.shape[:2]
    rng = np.random.default_rng(SEED + si)
    cand = pool[int(rng.integers(0, len(pool)))]
    r_raw = fit_r(cand, w, h)
    rng = np.random.default_rng(SEED + 1000 + si)
    m = coverage_mask(h, w, rng, frac=tuple(cover_frac))
    f_arc = arc_gradient(h, w, rng)
    r_eff = r_raw * r_overlay_field(h, w, rng)[..., None]
    return cand, r_raw, r_eff, m, f_arc


def sheet(rows, col_labels, path, cw=300, bg=(24, 24, 28)):
    pad, label_h, head_h = 5, 22, 24
    shaped = []
    for row in rows:
        cells = []
        for im, lab in row:
            im8 = (np.clip(im, 0, 1) * 255).round().astype(np.uint8)
            pil = Image.fromarray(im8)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            ih = max(1, int(round(cw * pil.height / pil.width)))
            cells.append((pil.resize((cw, ih), Image.LANCZOS), lab))
        shaped.append(cells)
    total_w = pad + len(col_labels) * (cw + pad)
    ys = [head_h + pad]
    for cells in shaped:
        ys.append(ys[-1] + max(c.height for c, _ in cells) + label_h + pad)
    S = Image.new("RGB", (total_w, ys[-1]), bg)
    d = ImageDraw.Draw(S)
    for ci, name in enumerate(col_labels):
        x0 = pad + ci * (cw + pad)
        d.rectangle([x0, pad, x0 + cw, pad + head_h], fill=(60, 60, 70))
        d.text((x0 + 6, pad + 6), name, fill=(255, 255, 255))
    for ri, cells in enumerate(shaped):
        y0 = ys[ri]
        for ci, (pil, lab) in enumerate(cells):
            x0 = pad + ci * (cw + pad)
            S.paste(pil, (x0, y0))
            d.rectangle([x0, y0 + pil.height, x0 + cw, y0 + pil.height + label_h], fill=(45, 45, 55))
            d.text((x0 + 5, y0 + pil.height + 5), lab, fill=(230, 230, 230))
    S.save(path)
    print("sheet ->", path)


def render_preview(args):
    gts = sorted(glob.glob(os.path.join(args.e1, "gt", "*.png")))
    pool = sorted(f for f in glob.glob(os.path.join(args.r_pool, "*.png"))
                  if not f.endswith("_alt.png"))
    rows, crop_rows = [], []
    for si in range(args.preview):
        t32 = load(gts[si])
        h, w = t32.shape[:2]
        cand, r_raw, r_eff, m, f_arc = scene_fields(t32, pool, si, args.cover_frac)
        rng = np.random.default_rng(SEED + 2000 + si)
        f_old = tint_field(h, w, rng)
        a_old3 = np.clip(0.3 * (1.0 + args.spatial * f_old), 0.01, 1.0)
        i_old3 = compose_alpha(t32, r_raw, a_old3, args.noise, SEED + si)
        a_new3 = alpha_from(m, f_arc, 0.3, args.spatial)
        a_new05 = alpha_from(m, f_arc, 0.05, args.spatial)
        i_new3 = compose_alpha(t32, r_eff, a_new3, args.noise, SEED + si, r_couple=args.r_couple)
        i_new05 = compose_alpha(t32, r_eff, a_new05, args.noise, SEED + si, r_couple=args.r_couple)
        sid = os.path.splitext(os.path.basename(gts[si]))[0]
        rows.append([
            (t32, f"{sid}  T m={t32.mean()*255:.0f}"),
            (r_raw, f"R_raw {os.path.basename(cand)[:14]} R̄={r_raw.mean()*255:.1f}"),
            (r_eff, f"R_eff(非均匀) R̄={r_eff.mean()*255:.1f}"),
            (gray(a_old3), f"α_old(0.3) 全场渐变 m={a_old3.mean():.2f}"),
            (i_old3, f"I_old(0.3) m={i_old3.mean()*255:.0f}"),
            (gray(a_new3), f"α_new(0.3) eff={a_new3.mean():.2f} cov={m.mean():.2f}"),
            (i_new3, f"I_new(0.3) m={i_new3.mean()*255:.0f}"),
            (i_new05, f"I_new(0.05) m={i_new05.mean()*255:.0f}"),
        ])
        ys, xs = np.where((m > 0.15) & (m < 0.85))
        cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (h // 2, w // 2)
        cs = min(380, h, w)
        y0 = int(np.clip(cy - cs // 2, 0, h - cs))
        x0 = int(np.clip(cx - cs // 2, 0, w - cs))
        crop_rows.append([
            (t32[y0:y0 + cs, x0:x0 + cs], f"{sid} T 1:1 膜边"),
            (i_old3[y0:y0 + cs, x0:x0 + cs], "I_old(0.3)"),
            (i_new3[y0:y0 + cs, x0:x0 + cs], "I_new(0.3)"),
            (i_new05[y0:y0 + cs, x0:x0 + cs], "I_new(0.05)"),
        ])
        print(f"{sid}: cov={m.mean():.2f} αeff(0.3)={a_new3.mean():.3f} αeff(0.05)={a_new05.mean():.3f} "
              f"R̄ raw={r_raw.mean()*255:.1f} -> eff={r_eff.mean()*255:.1f}")
    sheet(rows, ["T", "R_raw", "R_eff", "α_old(0.3)", "I_old(0.3)", "α_new(0.3)", "I_new(0.3)", "I_new(0.05)"],
          args.preview_out)
    crop_out = os.path.splitext(args.preview_out)[0] + "_crop1to1.png"
    sheet(crop_rows, ["T (1:1)", "I_old(0.3)", "I_new(0.3)", "I_new(0.05)"], crop_out, cw=cs)


def desat(r, k):
    """R 层色度压缩: k=1 不变; 0<k<1 各像素色度(=通道偏离灰度)乘 k。"""
    if k >= 0.999:
        return r
    g = r.mean(axis=2, keepdims=True)
    return np.clip(g + k * (r - g), 0.0, 1.0)


def render_sat_sweep(args):
    gts = sorted(glob.glob(os.path.join(args.e1, "gt", "*.png")))
    pool = sorted(f for f in glob.glob(os.path.join(args.r_pool, "*.png"))
                  if not f.endswith("_alt.png"))
    ks = [float(x) for x in args.sat_sweep.split(",")]
    rows, crop_rows = [], []
    for si in range(2):
        t32 = load(gts[si])
        h, w = t32.shape[:2]
        cand, _raw, r_eff, m, f_arc = scene_fields(t32, pool, si, args.cover_frac)
        amap = alpha_from(m, f_arc, args.sat_level, args.spatial)
        ys, xs = np.where((m > 0.15) & (m < 0.85))
        cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (h // 2, w // 2)
        cs = min(360, h, w)
        y0 = int(np.clip(cy - cs // 2, 0, h - cs))
        x0 = int(np.clip(cx - cs // 2, 0, w - cs))
        for p in (0.0, 1.0):
            row, crow = [], []
            for k in ks:
                rk = desat(r_eff, k)
                I = compose_alpha(t32, rk, amap, args.noise, SEED + si, r_couple=p)
                ch = float((rk.max(2) - rk.min(2)).mean() * 255)
                row.append((I, f"p={p:g} k={k:g} chroma={ch:.1f} m={I.mean()*255:.0f}"))
                crow.append((I[y0:y0 + cs, x0:x0 + cs], f"p={p:g} k={k:g}"))
            rows.append(row)
            crop_rows.append(crow)
        print(f"{os.path.splitext(os.path.basename(gts[si]))[0]} done")
    cols = [f"α={args.sat_level:g} k={k:g}" for k in ks]
    out = os.path.splitext(args.preview_out)[0] + "_satsweep.png"
    sheet(rows, cols, out, cw=340)
    sheet(crop_rows, cols, os.path.splitext(out)[0] + "_crop1to1.png", cw=cs)
    print("行序: sc0 p=0 | sc0 p=1 | sc1 p=0 | sc1 p=1")


def render_p_sweep(args):
    gts = sorted(glob.glob(os.path.join(args.e1, "gt", "*.png")))
    pool = sorted(f for f in glob.glob(os.path.join(args.r_pool, "*.png"))
                  if not f.endswith("_alt.png"))
    ps = [float(x) for x in args.p_sweep.split(",")]
    levels = [0.05, 0.15]
    rows, crop_rows = [], []
    for si in range(max(2, args.preview)):
        t32 = load(gts[si])
        h, w = t32.shape[:2]
        cand, _raw, r_eff, m, f_arc = scene_fields(t32, pool, si, args.cover_frac)
        sid = os.path.splitext(os.path.basename(gts[si]))[0]
        row, crow = [], []
        ys, xs = np.where((m > 0.15) & (m < 0.85))
        cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (h // 2, w // 2)
        cs = min(360, h, w)
        y0 = int(np.clip(cy - cs // 2, 0, h - cs))
        x0 = int(np.clip(cx - cs // 2, 0, w - cs))
        for a in levels:
            amap = alpha_from(m, f_arc, a, args.spatial)
            for p in ps:
                I = compose_alpha(t32, r_eff, amap, args.noise, SEED + si, r_couple=p)
                cov = m > 0.5
                share = r_eff[cov].mean() * (a ** p) / (amap[cov].mean() * t32[cov].mean() + r_eff[cov].mean() * (a ** p) + 1e-9)
                row.append((I, f"α={a:g} p={p:g} share={share*100:.0f}% m={I.mean()*255:.0f}"))
                crow.append((I[y0:y0 + cs, x0:x0 + cs], f"α={a:g} p={p:g}"))
        rows.append(row)
        crop_rows.append(crow)
        print(f"{sid} done")
    cols = [f"α={a:g} p={p:g}" for a in levels for p in ps]
    out = os.path.splitext(args.preview_out)[0] + "_psweep.png"
    sheet(rows, cols, out, cw=340)
    sheet(crop_rows, cols, os.path.splitext(out)[0] + "_crop1to1.png", cw=cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="/home/courseliu/vehicle_ssd/e1_testset")
    ap.add_argument("--r-pool", default="/DataDisk/zmy_code/TintedGlass_Enhance/r_rrw_strong/pool")
    ap.add_argument("--out", default=None)
    ap.add_argument("--spatial", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--cover-frac", type=float, nargs=2, default=[0.78, 0.92])
    ap.add_argument("--r-couple", type=float, default=0.0,
                    help="R 随 α 的耦合指数 p: I = α·T + α^p·R (0=不衰减, 1=比例守恒)")
    ap.add_argument("--r-sat", type=float, default=0.45, help="R 层色度压缩 k (1=不变; 0.45=实拍量级)")
    ap.add_argument("--preview", type=int, default=0, help="只渲染前 N 场景的新旧对比 sheet 后退出")
    ap.add_argument("--p-sweep", type=str, default="", help="如 0,0.5,1: 渲染 p 扫描对比 sheet 后退出")
    ap.add_argument("--sat-sweep", type=str, default="", help="如 1,0.6,0.45,0.3: 渲染色度压缩扫描后退出")
    ap.add_argument("--sat-level", type=float, default=0.3, help="色度扫描用的 α 档")
    ap.add_argument("--preview-out", default="/DataDisk/zmy_code/TintedGlass_Enhance/synth_v21_preview.png")
    args = ap.parse_args()
    if args.sat_sweep:
        render_sat_sweep(args)
        return
    if args.p_sweep:
        render_p_sweep(args)
        return
    if args.preview:
        render_preview(args)
        return
    assert args.out, "--out 必填"

    with open(os.path.join(args.e1, "meta.json")) as fp:
        json.load(fp)
    gts = sorted(glob.glob(os.path.join(args.e1, "gt", "*.png")))
    pool = sorted(f for f in glob.glob(os.path.join(args.r_pool, "*.png"))
                  if not f.endswith("_alt.png"))

    for sub in ["gt", "r", "alpha"] + [f"levels/alpha_{a:g}" for a in ALPHAS]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    meta = {"alphas": ALPHAS, "seed": SEED, "spatial": args.spatial, "noise": args.noise,
            "cover_frac": args.cover_frac, "r_pool": args.r_pool, "e1": args.e1,
            "renormalize": False, "r_couple": args.r_couple, "r_sat": args.r_sat,
            "physics": "v2.1: coverage-step alpha + non-uniform R overlay + I=a*T+a^p*R",
            "scenes": []}
    for si, gt_f in enumerate(gts):
        t32 = load(gt_f)
        cand, _r_raw, r_eff, m, f_arc = scene_fields(t32, pool, si, args.cover_frac)
        r_eff = desat(r_eff, args.r_sat)
        sid = os.path.splitext(os.path.basename(gt_f))[0]
        save(t32, os.path.join(args.out, "gt", sid + ".png"))
        save(r_eff, os.path.join(args.out, "r", sid + ".png"))

        rec = {"id": sid, "src_r": cand, "frac_cov": float(m.mean()),
               "r_mean_raw": float(_r_raw.mean()), "r_mean_eff": float(r_eff.mean()),
               "alpha_eff": {}, "r_obs": {}, "clip_frac": {}}
        for a in ALPHAS:
            amap = alpha_from(m, f_arc, a, args.spatial)
            out = compose_alpha(t32, r_eff, amap, args.noise, SEED + si, r_couple=args.r_couple)
            save(out, os.path.join(args.out, "levels", f"alpha_{a:g}", sid + ".png"))
            np.save(os.path.join(args.out, "alpha", f"{sid}_a{a:g}.npy"), amap.astype(np.float16))
            ae = float(amap.mean())
            rec["alpha_eff"][f"{a:g}"] = ae
            rec["clip_frac"][f"{a:g}"] = float((out > 1.0).mean())
            rec["r_obs"][f"{a:g}"] = float(r_eff.mean() / max(ae * t32.mean(), 1e-6))
        meta["scenes"].append(rec)
        print(f"{sid}: cov={rec['frac_cov']:.2f} R̄={rec['r_mean_raw']*255:.1f}->{rec['r_mean_eff']*255:.1f} "
              f"αeff(0.3)={rec['alpha_eff']['0.3']:.3f} r_obs(1)={rec['r_obs']['1']:.3f} "
              f"clip(1)={rec['clip_frac']['1']*100:.1f}%")
    with open(os.path.join(args.out, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=1, ensure_ascii=False)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
