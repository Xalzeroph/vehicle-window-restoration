#!/usr/bin/env python3
"""T 域统计表征：测试集 T vs 候选训练源的径向功率谱 / 高频占比 / 噪声 / 压缩率。

用途：
  ① 池配方决策（选哪些源、退化多少才能对上测试域）；
  ② 池建成后的验证（谱距离≈0 才算"域匹配"）。

口径：所有图统一缩到短边 224 后中心裁 224²（跨源同尺度可比）；谱为 Hann 窗
功率谱的径向平均，按总能量归一化；距离 = log10 谱曲线在 r_norm∈[0.05,0.9]
上的 RMS 差（以测试 T 为参考）。

用法:
  python3 scripts/domain_stats.py                 # 默认: 测试T + 五个候选源
  python3 scripts/domain_stats.py --n 40 --out fig/domain_spectra_0915.png
"""
import argparse, glob, os
import numpy as np
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
S = 224

DEFAULT_SRC = [
    ("test_v2.1   ", "data/test_sets/synth_rrw_v21_0913/gt/*.png"),
    ("sir2        ", "/home/zmy/mycode/datasets/SIR2/SIR/transmission_layer/*"),
    ("nature      ", "/home/zmy/mycode/datasets/nature/natural_T/*"),
    ("scenes      ", "/home/zmy/vehicle_ssd/datasets/scenes/*"),
    ("real_tl     ", "/home/zmy/mycode/datasets/real/transmission_layer/*"),
    ("rrw_gt      ", "/home/zmy/mycode/datasets/RRWDataset/**/*_GT.png"),
]


def load224(p, mode="norm", rng=None):
    """mode='norm': 短边缩到 S 再中心裁（跨源同尺度）；
    mode='native': 原生分辨率随机裁 224²（模型实际训练口径，短边<S 才放大）。"""
    im = Image.open(p).convert("RGB")
    w, h = im.size
    if mode == "native":
        if min(h, w) < S:
            sc = S / min(h, w)
            im = im.resize((max(S, int(w * sc)), max(S, int(h * sc))), Image.LANCZOS)
            w, h = im.size
        rng = rng or np.random.default_rng(0)
        x0 = int(rng.integers(0, w - S + 1)); y0 = int(rng.integers(0, h - S + 1))
    else:
        if min(h, w) < S:
            sc = S / min(h, w)
            im = im.resize((max(S, int(w * sc)), max(S, int(h * sc))), Image.LANCZOS)
            w, h = im.size
        sc = S / min(h, w)
        im = im.resize((max(S, int(round(w * sc))), max(S, int(round(h * sc)))), Image.LANCZOS)
        w, h = im.size
        x0, y0 = (w - S) // 2, (h - S) // 2
    return np.asarray(im.crop((x0, y0, x0 + S, y0 + S)), np.float32) / 255.0


def radial_spectrum(gray):
    win = np.outer(np.hanning(S), np.hanning(S))
    g = (gray - gray.mean()) * win
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    yy, xx = np.mgrid[-S // 2:S // 2, -S // 2:S // 2]
    rn = np.sqrt(xx ** 2 + yy ** 2) / (S / 2)
    nb = 112
    idx = np.clip((rn * nb).astype(int), 0, nb - 1)
    spec = np.bincount(idx.ravel(), weights=P.ravel(), minlength=nb)[:nb]
    cnt = np.bincount(idx.ravel(), minlength=nb)[:nb]
    spec = spec / np.maximum(cnt, 1)
    tot = spec.sum()
    return spec / tot if tot > 0 else spec


def noise_sigma(gray):
    """Immerkær 快速噪声估计（拉普拉斯核）。"""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    from scipy.ndimage import convolve
    lap = convolve(gray, k, mode="mirror")
    H, W = gray.shape
    return float(np.abs(lap).sum() * np.sqrt(np.pi / 2) / (6 * (W - 2) * (H - 2)))


def stats_of(files, mode="norm", rng=None):
    specs, hf25, hf10, lapv, nsig, sat = [], [], [], [], [], []
    for f in files:
        a = load224(f, mode, rng)
        g = a.mean(2)
        s = radial_spectrum(g)
        specs.append(s)
        rn = np.arange(len(s)) / (len(s) - 1)
        hf25.append(s[rn > 0.25].sum())
        hf10.append(s[rn > 0.10].sum())
        lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]) / 4 - g[1:-1, 1:-1]
        lapv.append(float(lap.var() * 1e3))
        nsig.append(noise_sigma(g))
        mx, mn = a.max(2), a.min(2)
        sat.append(float(np.mean((mx - mn) / np.maximum(mx, 1e-6))))
    return dict(spec=np.mean(specs, 0), hf25=np.mean(hf25), hf10=np.mean(hf10),
                lapv=np.mean(lapv), nsig=np.mean(nsig), sat=np.mean(sat), n=len(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="每源抽样张数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["norm", "native"], default="norm",
                    help="norm=统一缩到短边224比；native=原生分辨率224²裁（训练口径）")
    ap.add_argument("--src", action="append", default=[], metavar="NAME=PATTERN",
                    help="自定义源（可重复）；给了则只测 测试T + 这些源")
    ap.add_argument("--out", default="fig/domain_spectra_0915.png")
    ap.add_argument("--npz", default="fig/domain_spectra_0915.npz")
    args = ap.parse_args()
    if args.mode == "native":
        if args.out.endswith("_0915.png"):
            args.out = args.out.replace("_0915.png", "_native_0915.png")
        if args.npz.endswith("_0915.npz"):
            args.npz = args.npz.replace("_0915.npz", "_native_0915.npz")

    rng = np.random.default_rng(args.seed)
    srcs = list(DEFAULT_SRC)
    if args.src:
        srcs = DEFAULT_SRC[:1]  # 只保留测试 T 作参考
        for s in args.src:
            name, sep, pat = s.partition("=")
            assert sep and pat, f"--src 需 NAME=PATTERN 形式: {s}"
            srcs.append((name.ljust(12), pat))
    rows = {}
    for name, pat in srcs:
        fs = sorted(f for f in glob.glob(pat, recursive=True)
                    if os.path.splitext(f)[1].lower() in IMG_EXT)
        if not fs:
            print(f"{name}: 无文件, 跳过")
            continue
        if len(fs) > args.n:
            fs = [fs[i] for i in sorted(rng.choice(len(fs), args.n, replace=False))]
        bpp = np.median([os.path.getsize(f) / np.prod(Image.open(f).size) for f in fs])
        st = stats_of(fs, args.mode, np.random.default_rng(args.seed + 1))
        st["bpp"] = float(bpp)
        rows[name] = st
        print(f"[{name}] n={st['n']}")

    ref = rows["test_v2.1   "]["spec"]
    rn = np.arange(len(ref)) / (len(ref) - 1)
    band = (rn >= 0.05) & (rn <= 0.9)

    print(f"\n{'源':14s}{'HF(r>0.25)':>11s}{'HF(r>0.10)':>11s}{'lapvar×1e3':>11s}"
          f"{'噪声σ':>8s}{'饱和度':>8s}{'B/px':>8s}{'谱距离':>8s}")
    for name, st in rows.items():
        d = float(np.sqrt(np.mean((np.log10((st['spec'][band] + 1e-12) /
                                            (ref[band] + 1e-12))) ** 2)))
        st["dist"] = d
        print(f"{name:14s}{st['hf25']:11.3f}{st['hf10']:11.3f}{st['lapv']:11.2f}"
              f"{st['nsig']*1000:8.2f}{st['sat']:8.3f}{st['bpp']:8.3f}{d:8.3f}")

    np.savez(args.npz, rn=rn, **{f"spec_{k.strip()}": v["spec"] for k, v in rows.items()})
    print(f"\n谱曲线已存 {args.npz}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        for name, st in rows.items():
            lw, ls = (3.0, "-") if name.startswith("test") else (1.4, "--")
            ax.semilogy(rn, st["spec"] + 1e-12, ls, lw=lw, label=name.strip())
        ax.set_xlabel("radial frequency (norm.)"); ax.set_ylabel("normalized power")
        ax.set_ylim(1e-8, 1e-2); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title("T-domain radial power spectra (short-side 224px)")
        fig.tight_layout(); fig.savefig(args.out)
        print(f"图已存 {args.out}")
    except Exception as e:
        print(f"画图跳过: {e}")


if __name__ == "__main__":
    main()
