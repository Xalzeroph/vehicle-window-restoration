#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""洞区 vs 非洞区的分区域评测（2026-10-20）。

洞定义（用户口径）：`|R|` 小（数据上没有反射） ∧ 边缘强（但有结构轮廓）。

核心问题：**模型在洞区是不是更容易改坏？**
若洞区 PSNR 增益显著低于非洞区 → 证明"洞"是困难区域，
这条可成为论文里一张表（masked metric），也是模块设计的直接依据。

输出
----
eval/hole_region_eval.json   逐臂：洞区/非洞区的 PSNR 与增益
_review_1019/hole_masks/     各场景洞掩膜 .png（供后续复用）
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")
MASKDIR = os.path.join(P, "_review_1019/hole_masks")
EVAL = os.path.join(P, "eval")

W = 760
LAMBDA = 12.0
R_LOW_PCT = 30
EDGE_HI_PCT = 80
MIN_AREA = 25

ARMS = {"rdnet_zs": "real_pair_1019_rdnet_zs",
        "ft_s2main": "real_pair_1019_ft_s2main",
        "ft_mixbest": "real_pair_1019_ft_mixbest",
        "lightrr": "real_pair_1019_lightrr"}

# ★ 循环论证警告：洞掩膜由 `|R|` 小定义，而 `R = I−GT`。
# 于是"洞区误差小"是**构造出来的**，不能直接解读为"模型在洞区表现好"。
# 破法：以 **identity（直接输出 I）** 为参照基线，比较各臂的
#   Δgap = gap(model) − gap(identity)
# 洞区相对非洞区被改坏得更多时，Δgap 会显著小于 0。


def load(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im.astype(np.float64) if im is not None else None


def robust_affine(src, ref, keep=0.7, iters=3):
    A = np.ones(3); B = np.zeros(3)
    for c in range(3):
        x = src[..., c].ravel().astype(np.float64)
        y = ref[..., c].ravel().astype(np.float64)
        for _ in range(iters):
            V = np.vstack([y, np.ones_like(y)]).T
            coef, *_ = np.linalg.lstsq(V, x, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            rr = np.abs(x - (a * y + b))
            k = rr <= np.percentile(rr, keep * 100)
            x, y = x[k], y[k]
            if len(x) < 500:
                break
        A[c], B[c] = a, b
    return A, B


def cleanup(m, min_area=MIN_AREA):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) >= min_area:
            out |= (lab == i).astype(np.uint8)
    return out.astype(bool)


def psnr_mask(a, b, mask):
    """只在该掩膜内算 PSNR（逐通道后取均值）。"""
    if mask.sum() < 50:
        return float("nan")
    vals = []
    for c in range(3):
        av = a[..., c][mask].astype(np.float64)
        bv = b[..., c][mask].astype(np.float64)
        mse = float(np.mean((av - bv) ** 2))
        vals.append(99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(vals))


def main() -> int:
    os.makedirs(MASKDIR, exist_ok=True)
    os.makedirs(EVAL, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))

    rows = []
    for sc in scenes:
        d = os.path.join(PAIR, sc)
        I0 = load(os.path.join(d, "I_000000.png"))
        G0 = load(os.path.join(d, "GT.png"))
        if I0 is None or G0 is None:
            continue
        h, w = I0.shape[:2]
        sh = max(2, int(round(h * W / w)))
        I = cv2.resize(I0, (W, sh), interpolation=cv2.INTER_AREA)
        G = cv2.resize(G0, (W, sh), interpolation=cv2.INTER_AREA)

        A, B = robust_affine(I, G)
        I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)
        R_a = np.abs((I_al - G).mean(axis=2))
        Ig = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(Ig, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(Ig, cv2.CV_32F, 0, 1)
        edge = np.sqrt(gx * gx + gy * gy)

        hole = cleanup((R_a <= np.percentile(R_a, R_LOW_PCT))
                       & (edge >= np.percentile(edge, EDGE_HI_PCT)))
        cv2.imwrite(os.path.join(MASKDIR, f"{sc}.png"),
                    (hole.astype(np.uint8) * 255))

        # 把预测图缩放到同一尺寸
        rec = {"scene": sc, "hole_frac": round(float(hole.mean()), 4)}
        nonhole = ~hole
        # identity 基线：直接输出 I
        rec["identity"] = {"psnr_hole": round(psnr_mask(I, G, hole), 3),
                           "psnr_nonhole": round(psnr_mask(I, G, nonhole), 3)}
        for arm, sub in ARMS.items():
            pth = os.path.join(PRED, sub, f"{sc}__I_000000", "t_hat.png")
            if not os.path.exists(pth):
                rec[arm] = None
                continue
            Pr = cv2.imread(pth, cv2.IMREAD_COLOR).astype(np.float64)
            if Pr.shape[:2] != (sh, W):
                Pr = cv2.resize(Pr, (W, sh), interpolation=cv2.INTER_AREA)
            rec[arm] = {"psnr_hole": round(psnr_mask(Pr, G, hole), 3),
                        "psnr_nonhole": round(psnr_mask(Pr, G, nonhole), 3)}
        rows.append(rec)

    # 汇总
    summ = {}
    for arm in ["identity"] + list(ARMS):
        hh, nn = [], []
        for r in rows:
            if r.get(arm):
                hh.append(r[arm]["psnr_hole"]); nn.append(r[arm]["psnr_nonhole"])
        if hh:
            summ[arm] = {"psnr_hole": round(float(np.nanmean(hh)), 3),
                         "psnr_nonhole": round(float(np.nanmean(nn)), 3),
                         "gap": round(float(np.nanmean(hh) - np.nanmean(nn)), 3)}

    base_gap = summ.get("identity", {}).get("gap", 0.0)
    base_h = summ.get("identity", {}).get("psnr_hole", 0.0)
    base_n = summ.get("identity", {}).get("psnr_nonhole", 0.0)

    json.dump({"per_scene": rows, "summary": summ},
              open(os.path.join(EVAL, "hole_region_eval.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"{'臂':<12}{'洞区':>8}{'非洞区':>9}{'gap':>8}"
          f"{'Δ洞区':>9}{'Δ非洞区':>10}{'Δgap':>8}")
    print("-" * 66)
    for arm, s in summ.items():
        dh = s["psnr_hole"] - base_h
        dn = s["psnr_nonhole"] - base_n
        dg = s["gap"] - base_gap
        print(f"{arm:<12}{s['psnr_hole']:>8.2f}{s['psnr_nonhole']:>9.2f}"
              f"{s['gap']:>8.2f}{dh:>9.2f}{dn:>10.2f}{dg:>8.2f}")
    print(f"\n洞区平均占比 {np.mean([r['hole_frac'] for r in rows])*100:.2f}%")
    print("判读：Δgap<0 → 该臂在洞区相对非洞区**被改坏得更多**（相对 identity）")
    print(f"掩膜 -> {MASKDIR}\n统计 -> eval/hole_region_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
