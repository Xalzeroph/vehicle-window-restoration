#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""频率选择性去反射：LP_σ(D) 扫描 + 残差频谱诊断（2026-10-20）。

背景
----
用户指出的矛盾（很关键）：
  · 若反射 95% 能量在低频 → 去反射只动低频 → 不该伤高频细节；
  · 但实测模型输出细节掉 24–38%。
  ⇒ 细节损失**不是去反射造成的**，而是模型自身把整幅图平滑了。
  ⇒ "反射偏低频"这条先验解释不了失效机制，也指导不了设计。

本脚本要回答的是**可测的问题**（而不是继续争论先验）：
  1. 各模型的实际残差 D = I − T̂ 里，高频占多少？（若 D 本身就平滑，LP 无意义）
  2. 各模型输出的细节（Laplacian var）掉了多少？
  3. 对 D 做低通再回加：T̂_LP = I − LP_σ(D)，PSNR / SSIM / Laplacian var 怎么变？
     —— 这是"若残差高频有害，则 LP 会同时抬 PSNR 与细节"的直接检验。

输出
----
eval/lp_sweep_1019.json
eval/fig_1019/lp_sweep.png
"""
from __future__ import annotations

import csv
import json
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")
EVAL = os.path.join(P, "eval")
FIG = os.path.join(EVAL, "fig_1019")

ARMS = {"rdnet_zs": "real_pair_1019_rdnet_zs",
        "ft_s2main": "real_pair_1019_ft_s2main",
        "ft_mixbest": "real_pair_1019_ft_mixbest",
        "lightrr": "real_pair_1019_lightrr"}
SIGMAS = [0.0, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
LVL = 3
BANDS = [f"{o}{l}" for l in range(1, LVL + 1) for o in ("LL", "LH", "HL", "HH")]
HF_BANDS = [b for b in BANDS if not b.startswith("LL")]


def read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im.astype(np.float64) if im is not None else None


def psnr(a, b):
    v = []
    for c in range(3):
        mse = float(np.mean((a[..., c] - b[..., c]) ** 2))
        v.append(99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(v))


def ssim(a, b):
    return float(np.mean([ssim_fn(a[..., c], b[..., c], data_range=255)
                          for c in range(3)]))


def lapvar(img):
    g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def dwt_bands(gray, levels=LVL):
    out, cur = {}, gray.astype(np.float32)
    for l in range(1, levels + 1):
        h, w = cur.shape
        cur = cur[:h // 2 * 2, :w // 2 * 2]
        a, b = cur[:, 0::2], cur[:, 1::2]
        lo, hi = (a + b) / np.sqrt(2.0), (a - b) / np.sqrt(2.0)
        c, d = lo[0::2, :], lo[1::2, :]
        LL, LH = (c + d) / np.sqrt(2.0), (c - d) / np.sqrt(2.0)
        c2, d2 = hi[0::2, :], hi[1::2, :]
        HL, HH = (c2 + d2) / np.sqrt(2.0), (c2 - d2) / np.sqrt(2.0)
        out[f"LL{l}"] = LL; out[f"LH{l}"] = LH
        out[f"HL{l}"] = HL; out[f"HH{l}"] = HH
        cur = LL
    return out


def hf_share(gray):
    b = dwt_bands(gray)
    tot = sum(float(np.mean(v.astype(np.float64) ** 2)) for v in b.values())
    if tot <= 1e-9:
        return float("nan")
    hf = sum(float(np.mean(b[k].astype(np.float64) ** 2)) for k in HF_BANDS)
    return hf / tot


def pairs():
    out = []
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
                    out.append((sc, os.path.splitext(f)[0], I, G))
    return out


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    P0 = pairs()
    print(f"配对 {len(P0)} 组\n")

    gt_lap = float(np.mean([lapvar(G) for _, _, _, G in P0]))
    print(f"GT 的 Laplacian var 均值 = {gt_lap:.1f}")
    inp_lap = float(np.mean([lapvar(I) for _, _, I, _ in P0]))
    print(f"INPUT 的 Laplacian var 均值 = {inp_lap:.1f}\n")

    res = {"gt_lapvar": round(gt_lap, 1), "input_lapvar": round(inp_lap, 1),
           "arms": {}}

    for arm, sub in ARMS.items():
        # 收集预测
        recs = []
        for sc, stem, I, G in P0:
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
            print(f"[跳过] {arm}: 无预测")
            continue

        # 诊断：D = I − T̂ 的频谱 & 输出细节
        d_hf, out_lap = [], []
        for I, Pr, G in recs:
            D = (I - Pr).mean(axis=2)
            d_hf.append(hf_share(D))
            out_lap.append(lapvar(Pr))
        d_hf_m = float(np.nanmean(d_hf))
        out_lap_m = float(np.mean(out_lap))

        line = []
        best = None
        for sig in SIGMAS:
            ps, ss, lv = [], [], []
            for I, Pr, G in recs:
                D = I - Pr
                if sig <= 0:
                    T = Pr
                else:
                    T = I - cv2.GaussianBlur(D.astype(np.float32), (0, 0), sig).astype(np.float64)
                T = np.clip(T, 0, 255)
                ps.append(psnr(T, G)); ss.append(ssim(T, G)); lv.append(lapvar(T))
            m = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
                 "lapvar": float(np.mean(lv))}
            line.append((sig, m))
            if best is None or m["psnr"] > best[1]["psnr"]:
                best = (sig, m)

        res["arms"][arm] = {
            "residual_hf_share": round(d_hf_m, 4),
            "output_lapvar": round(out_lap_m, 1),
            "detail_ratio_vs_gt": round(out_lap_m / max(gt_lap, 1e-9), 3),
            "sweep": {str(s): {k: round(v, 4) for k, v in m.items()} for s, m in line},
            "best_sigma": best[0] if best else None,
            "best": {k: round(v, 4) for k, v in best[1].items()} if best else None}
        a = res["arms"][arm]
        print(f"{arm:<12} 残差HF占比={d_hf_m:.3f}  输出Laplacian={out_lap_m:6.1f} "
              f"({a['detail_ratio_vs_gt']*100:.0f}% of GT)")
        print("   σ: " + "  ".join(f"{s}:{m['psnr']:.2f}/{m['lapvar']:.0f}"
                                   for s, m in line))
        print(f"   最佳σ={best[0]} PSNR={best[1]['psnr']:.2f} "
              f"(σ=0 时 {line[0][1]['psnr']:.2f}, 增益 {best[1]['psnr']-line[0][1]['psnr']:+.2f} dB)\n")

    json.dump(res, open(os.path.join(EVAL, "lp_sweep_1019.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"写入 {EVAL}/lp_sweep_1019.json")

    # 出图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for arm in res["arms"]:
            sw = res["arms"][arm]["sweep"]
            xs = sorted(float(k) for k in sw)
            keys = [k for k in sw]
            key_of = {float(k): k for k in keys}
            ax[0].plot(xs, [sw[key_of[x]]["psnr"] for x in xs], "o-", label=arm)
            ax[1].plot(xs, [sw[key_of[x]]["ssim"] for x in xs], "o-", label=arm)
            ax[2].plot(xs, [sw[key_of[x]]["lapvar"] for x in xs], "o-", label=arm)
        ax[2].axhline(gt_lap, color="k", ls="--", lw=1, label="GT detail")
        for a, t in zip(ax, ["PSNR (dB)", "SSIM", "Laplacian var (detail)"]):
            a.set_xlabel("sigma of low-pass applied to residual D")
            a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(FIG, "lp_sweep.png")
        plt.savefig(p, dpi=140)
        print(f"图 -> {p}")
    except Exception as e:
        print(f"[跳过出图] {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
