#!/usr/bin/env python3
"""无泄漏"域匹配" T 池构建（2026-09-15 配方，见 memory: project-pool-rebuild）。

配方（对有内容核验的干净源逐图施加）:
  ① 缩放到短边 384（≈测试 T 的 386）
  ② 径向谱匹配滤波: g(f)=sqrt(P_test(f)/P_src(f))，仅衰减（clip ≤1）
  ③ 对比度匹配: 逐通道 std → 测试 T 的逐通道 std 均值
  ④ 重编码 JPEG: q 自校准到测试 label 的字节/像素（≈0.067）

用法:
  python3 scripts/build_matched_pool.py                 # 全量构建
  python3 scripts/build_matched_pool.py --max-per-src 300 --out matched_pool_0915_small
完成后验证:
  python3 scripts/domain_stats.py --mode native --src matched=matched_pool_0915/*.jpg
  python3 scripts/check_t_src_leak.py --t-src m=matched_pool_0915 --t-glob '*.jpg' \
      --ts-dir synth_rrw_v21_0913/gt --skip-crop
"""
import argparse, glob, json, os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_stats import radial_spectrum  # noqa: E402  (与域表征同一测量口径)

CROP = 224          # 谱测量窗（与 domain_stats 一致）
NB = 112            # 径向分箱数
SHORT = 384         # 目标短边（测试 T=386）
TS_DIR = "synth_rrw_v21_0913"

SOURCES = [
    ("sir2", "/home/zmy/mycode/datasets/SIR2/SIR/transmission_layer/*"),
    ("nature", "/home/zmy/mycode/datasets/nature/natural_T/*"),
    ("scenes", "/home/zmy/vehicle_ssd/datasets/scenes/*"),
    ("real_tl", "/home/zmy/mycode/datasets/real/transmission_layer/*"),
    ("rrw_gt", "/home/zmy/mycode/datasets/RRWDataset/**/*_GT.png"),
]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def resize_short(im, s):
    w, h = im.size
    sc = s / min(h, w)
    if abs(sc - 1.0) > 1e-3:
        im = im.resize((max(s, int(round(w * sc))), max(s, int(round(h * sc)))),
                       Image.LANCZOS)
    return im


def crops_spec(a, rng, n=4):
    h, w = a.shape[:2]
    out = []
    for _ in range(n):
        y0 = int(rng.integers(0, h - CROP + 1)); x0 = int(rng.integers(0, w - CROP + 1))
        out.append(radial_spectrum(a[y0:y0 + CROP, x0:x0 + CROP].mean(2)))
    return np.mean(out, 0)


def gain_map(H, W, gain_bins):
    fy = np.fft.fftfreq(H); fx = np.fft.fftfreq(W)          # cycles/px
    rn = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) * 2.0  # → 与测量同标度 (0..1.41)
    return gain_bins[np.clip((rn * NB).astype(int), 0, NB - 1)]


def image_psd(a):
    """全图 Hann 窗灰度 PSD 的径向平均（与目标测量同标度，样本量远大于小窗）。"""
    g = a.mean(2)
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fft2((g - g.mean()) * win)) ** 2
    fy = np.fft.fftfreq(h); fx = np.fft.fftfreq(w)
    rn = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) * 2.0
    idx = np.clip((rn * NB).astype(int), 0, NB - 1)
    s = np.bincount(idx.ravel(), weights=P.ravel(), minlength=NB)[:NB]
    c = np.bincount(idx.ravel(), minlength=NB)[:NB]
    s = s / np.maximum(c, 1)
    return s / s.sum()   # 归一化，与 radial_spectrum 同口径


def preserve_dc(g, n=3):
    """前 n 个径向 bin（含 DC）及其后 3 个过渡 bin 的增益拉回 1，避免全局亮度/大尺度明暗被改写。"""
    g = g.copy()
    g[:n] = 1.0
    for k in range(3):
        i = n + k
        if i < len(g):
            w = 1.0 - (k + 1) / 4.0   # 0.75/0.5/0.25 过渡
            g[i] = 1.0 + (g[i] - 1.0) * w
    return g


def apply_gain(a, gmap):
    out = np.empty_like(a)
    for c in range(3):
        F = np.fft.fft2(a[..., c])
        out[..., c] = np.real(np.fft.ifft2(F * gmap))
    return out


def test_target(rng):
    files = sorted(glob.glob(os.path.join(TS_DIR, "gt", "*.png")))
    assert files, "测试集 gt 缺"
    specs, stds, bpp = [], [], []
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0
        specs.append(crops_spec(a, rng, 6))
        stds.append(a.reshape(-1, 3).std(0))
    # 字节/像素目标: augmented 池 label（=测试 T 的原生编码）
    for f in sorted(glob.glob("/home/zmy/vehicle_ssd/datasets/augmented/a*_label.jpg")):
        w, h = Image.open(f).size
        bpp.append(os.path.getsize(f) / (w * h))
    return (np.mean(specs, 0), np.mean(stds, 0), float(np.median(bpp)),
            len(files), len(bpp))


def exclude_rrw_testscenes(files):
    """排除与测试 R 同场景的 RRW GT（弱耦合防护）。"""
    with open(os.path.join(TS_DIR, "meta.json")) as fp:
        scenes = {os.path.basename(s["src_r"]).split("__")[-1].rsplit(".", 1)[0]
                  for s in json.load(fp)["scenes"]}
    keep, drop = [], []
    for f in files:
        stem = os.path.basename(f)[:-len("_GT.png")]
        (drop if stem in scenes else keep).append(f)
    return keep, drop, scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="matched_pool_0915")
    ap.add_argument("--max-per-src", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=float, default=0.02, help="滤波增益下限")
    ap.add_argument("--cap", type=float, default=4.0, help="滤波增益上限（源需放大时允许 >1）")
    ap.add_argument("--preblur", type=float, default=0.5, help="滤波后预模糊 σ（压近 Nyquist 伪影/噪声底）")
    ap.add_argument("--corr", default="", help="迭代校准曲线 .npy（长度 NB，乘到增益上）")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    corr = np.load(args.corr) if args.corr else None
    if corr is not None:
        assert corr.shape == (NB,), f"corr 形状应为 ({NB},), 实为 {corr.shape}"

    tgt_spec, tgt_std, tgt_bpp, n_gt, n_lab = test_target(rng)
    print(f"测试 T 目标: 通道std={tgt_std.round(3)}  B/px={tgt_bpp:.3f} "
          f"(gt {n_gt} 张 / label {n_lab} 个)")

    jobs = []           # (src_name, out_name, path)
    for name, pat in SOURCES:
        fs = sorted(f for f in glob.glob(pat, recursive=True)
                    if os.path.splitext(f)[1].lower() in IMG_EXT)
        if name == "rrw_gt":
            fs, drop, scenes = exclude_rrw_testscenes(fs)
            print(f"[{name}] 排除测试 R 同场景 GT: {len(drop)} 张 "
                  f"(命中场景 {sorted({os.path.basename(d)[:-7] for d in drop})[:5]})")
        if len(fs) > args.max_per_src:
            fs = [fs[i] for i in sorted(rng.choice(len(fs), args.max_per_src, replace=False))]
        print(f"[{name}] {len(fs)} 张")
        for i, f in enumerate(fs):
            jobs.append((name, f"{name}_{i:05d}.jpg", f))

    # 预处理（滤波+对比度）到内存样本，用于 JPEG q 自校准
    def prepare(path):
        a = np.asarray(resize_short(Image.open(path).convert("RGB"), SHORT),
                       np.float32) / 255.0
        if args.preblur > 0:   # 先模糊、再测谱、后增益：增益会把模糊一并补偿掉
            a = gaussian_filter(a, (args.preblur, args.preblur, 0))
        # 增益作用于去均值图：DC 不动（亮度严格保持），全 AC 频段参与匹配
        mu = a.mean((0, 1), keepdims=True)
        g = np.sqrt(tgt_spec / (image_psd(a) + 1e-12))
        if corr is not None:
            g = g * corr
        g = np.clip(g, args.floor, args.cap)
        a = apply_gain(a - mu, gain_map(*a.shape[:2], g)) + mu
        s = a.reshape(-1, 3).std(0)
        a = (a - a.mean((0, 1))) * (tgt_std / np.maximum(s, 1e-6)) + a.mean((0, 1))
        return np.clip(a, 0, 1)

    import io
    probe_idx = sorted(rng.choice(len(jobs), min(24, len(jobs)), replace=False))
    probes = [prepare(jobs[i][2]) for i in probe_idx]
    best_q, best_err = None, 1e9
    for q in range(10, 61, 5):
        bpp = []
        for a in probes:
            buf = io.BytesIO()
            Image.fromarray((a * 255).round().astype(np.uint8)).save(buf, "JPEG", quality=q)
            bpp.append(len(buf.getvalue()) / a.shape[0] / a.shape[1])
        err = abs(np.median(bpp) - tgt_bpp)
        print(f"  q={q:3d}  中位 B/px={np.median(bpp):.3f}  |Δ|={err:.3f}")
        if err < best_err:
            best_q, best_err = q, err
    print(f"选定 JPEG q={best_q}")

    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for k, (name, out_name, path) in enumerate(jobs):
        a = prepare(path)
        Image.fromarray((a * 255).round().astype(np.uint8)).save(
            os.path.join(args.out, out_name), "JPEG", quality=best_q)
        counts[name] = counts.get(name, 0) + 1
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(jobs)}", flush=True)

    meta = dict(recipe="谱匹配+对比度匹配+JPEG重编码 (scripts/build_matched_pool.py)",
                short=SHORT, jpeg_q=best_q, tgt_std=tgt_std.tolist(),
                tgt_bpp=tgt_bpp, max_per_src=args.max_per_src, seed=args.seed,
                counts=counts, n_total=len(jobs))
    with open(os.path.join(args.out, "meta.json"), "w") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=1)
    print(f"完成: {args.out}  共 {len(jobs)} 张 {counts}")


if __name__ == "__main__":
    main()
