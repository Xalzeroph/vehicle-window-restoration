#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高频来源调查：模型的高频 vs 输入的高频，谁更接近 GT？（2026-10-20）

要回答的问题
-----------
已知 PSNR 增益主要来自低频（色调+低频反射），梯度余弦纹丝不动（0.599→0.611）。
那么**高频到底该从哪来？** 三个可能：
  (a) 模型的 HF 比输入的 HF 更接近 GT → 只需更激进地去反射
  (b) 输入的 HF 更接近 GT        → 模型应少动 HF（保频）
  (c) 都不够，需要“部分替换/重加权” → 需要一个高频归属模块

实验设计（零 GPU）
----------------
沿一维家族扫描 α：

$$
\\hat T(\\alpha) = \\mathrm{LP}_\\sigma(\\hat T_{\\text{model}})
                 + \\alpha\\,\\mathrm{HP}_\\sigma(I)
                 + (1-\\alpha)\\,\\mathrm{HP}_\\sigma(\\hat T_{\\text{model}})
$$

- α=0 → 模型原输出
- α=1 → 模型的低频 + **输入的高频**（HF 移植）
- 中间 → 两者混合

若某个 α* 使 **HF-PSNR / 梯度余弦** 同时改善（哪怕 PSNR 略降），
则 (c) 成立：**高频需要重新归属，这就是模块该做的事**。

输出：eval/hf_mix_sweep.json + fig_1019/hf_mix_sweep.png
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")
EVAL = os.path.join(P, "eval")
FIG = os.path.join(EVAL, "fig_1019")

ARMS = {"rdnet_zs": "real_pair_1019_rdnet_zs",
        "ft_s2main": "real_pair_1019_ft_s2main",
        "ft_mixbest": "real_pair_1019_ft_mixbest",
        "lightrr": "real_pair_1019_lightrr"}
SIGMA = 2.0
ALPHAS = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]


def read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im.astype(np.float64) if im is not None else None


def gray(x):
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8),
                        cv2.COLOR_BGR2GRAY).astype(np.float32)


def split(img, sigma=SIGMA):
    """对彩色图逐通道做高斯低/高通分解。"""
    lp = np.stack([cv2.GaussianBlur(img[..., c].astype(np.float32), (0, 0), sigma)
                   for c in range(3)], axis=2).astype(np.float64)
    return lp, img - lp


def psnr_gray(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def psnr_rgb(a, b):
    return float(np.mean([psnr_gray(a[..., c], b[..., c]) for c in range(3)]))


def highpass_g(g, sigma=SIGMA):
    return g - cv2.GaussianBlur(g, (0, 0), sigma)


def grad_cos(a_g, b_g, topk=0.15):
    ax, ay = np.gradient(a_g); bx, by = np.gradient(b_g)
    ma, mb = np.hypot(ax, ay), np.hypot(bx, by)
    m = (ma >= np.percentile(ma, (1 - topk) * 100)) & \
        (mb >= np.percentile(mb, (1 - topk) * 100))
    if m.sum() < 100:
        return float("nan")
    dot = (ax * bx + ay * by)[m]
    return float(np.mean(dot / (ma * mb)[m].clip(1e-6)))


def lapvar(x):
    return float(cv2.Laplacian(gray(x), cv2.CV_32F).var(dtype=np.float64))


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    pairs = []
    for sc in sorted(os.listdir(PAIR)):
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
        G = read(os.path.join(d, "GT.png"))
        if G is None:
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("I_") and f.endswith(".png"):
                I = read(os.path.join(d, f))
                if I is not None:
                    pairs.append((sc, os.path.splitext(f)[0], I, G))
    print(f"配对 {len(pairs)} 组  σ={SIGMA}\n")

    gt_lap = float(np.mean([lapvar(G) for _, _, _, G in pairs]))
    gt_g = [gray(G) for _, _, _, G in pairs]

    res = {"sigma": SIGMA, "gt_lapvar": round(gt_lap, 1), "arms": {}}
    print(f"GT LapVar = {gt_lap:.1f}\n")
    hdr = (f"{'臂':<12}{'α':>6}{'PSNR':>8}{'HF-PSNR':>9}{'梯度余弦':>9}"
           f"{'LapVar':>8}{'%GT':>6}")
    print(hdr); print("-" * len(hdr))

    for arm, sub in ARMS.items():
        recs = []
        for sc, stem, I, G in pairs:
            p = os.path.join(PRED, sub, f"{sc}__{stem}", "t_hat.png")
            if not os.path.exists(p):
                continue
            Pr = read(p)
            if Pr is None:
                continue
            if Pr.shape[:2] != I.shape[:2]:
                Pr = cv2.resize(Pr, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_AREA)
            recs.append((I, Pr, G))
        if not recs:
            continue

        curve = {}
        for a in ALPHAS:
            ps, hf, gc, lv = [], [], [], []
            for I, Pr, G in recs:
                lp_m, hp_m = split(Pr)
                _, hp_i = split(I)
                T = np.clip(lp_m + a * hp_i + (1 - a) * hp_m, 0, 255)
                ps.append(psnr_rgb(T, G))
                hf.append(psnr_gray(highpass_g(gray(T)), highpass_g(gray(G))))
                gc.append(grad_cos(gray(T), gray(G)))
                lv.append(lapvar(T))
            curve[a] = {"psnr": round(float(np.mean(ps)), 3),
                        "hf_psnr": round(float(np.mean(hf)), 3),
                        "grad_cos": round(float(np.nanmean(gc)), 4),
                        "lapvar": round(float(np.mean(lv)), 1)}
            print(f"{arm:<12}{a:>6.2f}{curve[a]['psnr']:>8.2f}{curve[a]['hf_psnr']:>9.2f}"
                  f"{curve[a]['grad_cos']:>9.3f}{curve[a]['lapvar']:>8.1f}"
                  f"{curve[a]['lapvar']/gt_lap*100:>5.0f}%")
        # 找 HF-PSNR 与 GC 的最优点
        best_hf = max(curve, key=lambda k: curve[k]["hf_psnr"])
        best_gc = max(curve, key=lambda k: curve[k]["grad_cos"])
        res["arms"][arm] = {"curve": {str(k): v for k, v in curve.items()},
                            "best_hf_alpha": best_hf,
                            "best_gc_alpha": best_gc,
                            "base": curve[0.0]}
        print(f"  → HF-PSNR 最优 α={best_hf} "
              f"({curve[best_hf]['hf_psnr']:.2f} vs 原 {curve[0.0]['hf_psnr']:.2f}) | "
              f"GC 最优 α={best_gc} ({curve[best_gc]['grad_cos']:.3f} vs "
              f"{curve[0.0]['grad_cos']:.3f})\n")

    json.dump(res, open(os.path.join(EVAL, "hf_mix_sweep.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"写入 {EVAL}/hf_mix_sweep.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for arm in res["arms"]:
            c = res["arms"][arm]["curve"]
            xs = ALPHAS
            ax[0].plot(xs, [c[str(a)]["psnr"] for a in xs], "o-", label=arm)
            ax[1].plot(xs, [c[str(a)]["hf_psnr"] for a in xs], "o-", label=arm)
            ax[2].plot(xs, [c[str(a)]["grad_cos"] for a in xs], "o-", label=arm)
        ax[0].set_title("PSNR"); ax[1].set_title("HF-PSNR"); ax[2].set_title("gradient cosine")
        for a, t in zip(ax, ["PSNR (dB)", "HF-PSNR (dB)", "gradient cosine"]):
            a.set_xlabel("alpha  (0=model HF, 1=input HF)")
            a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(FIG, "hf_mix_sweep.png")
        plt.savefig(p, dpi=140)
        print(f"图 -> {p}")
    except Exception as e:
        print(f"[跳过出图] {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
