#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞 v11 —— 由"不属于 T 的边缘"围成的区域（2026-10-20）。

思路转变
-------
v9/v10 都从"局部凹陷"出发，只能抓到小碎斑（最大 ~600px），
而用户指出真洞**更大**、v9 选出来只是子集。

按用户定义反推最直接的算法：
  洞 = **被"不属于透射层的边缘"围起来的那块区域**
因为：
  · "不能在 T 中出现这个结构" ⇒ 边界不属于 T
  · "有边缘"     ⇒ 边界在 I 上可见
  · "反射确实弱" ⇒ 区域内部 |R| 比外面低

步骤
----
1. 求"不属于 T 的边缘" `E`：`∇I >= p85` ∧ `∇T <= p55`
2. 形态学闭运算把 `E` 连成闭合轮廓
3. 用 `findContours + 填充` 得到 `E` 围起来的**内部区域**
4. 区域级检验：内部 `|R|` 中位 < 外部环带 `|R|` 中位（反射确实弱）
5. 剥掉轮廓带 → 得洞的**内部**

输出：_review_1019/hole_v11/<scene>.jpg
      eval/hole_v11.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
PAIR = os.path.join(P, "data/test_sets/real_pair_1019")
OUT = os.path.join(P, "_review_1019/hole_v11")
EVAL = os.path.join(P, "eval")
W = 700
E_I_PCT = 82          # 边界：∇I 强
E_T_PCT = 60          # 边界：∇T 弱（不是透射层的结构）
CLOSE_K = 11          # 闭合核，把边缘连成闭合圈
WEAK_RATIO = 1.00     # 内部 |R| 中位 / 外部环带中位 < 该值 → 反射更弱
MIN_AREA = 200
ERODE_PX = 2          # 剥掉轮廓带


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


def grad(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy)


def turbo(x, p=(1, 99.5)):
    lo, hi = np.percentile(x, p); hi = max(hi, lo + 1e-9)
    return cv2.applyColorMap((np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_TURBO)


def title(img, t, h=20):
    w = img.shape[1]
    bar = np.zeros((h, w, 3), np.uint8)
    cv2.putText(bar, t, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))
    res = {}
    print(f"{'场景':<30}{'边界%':>8}{'围成%':>8}{'洞内部%':>9}{'洞数':>6}{'最大洞':>8}"
          f"{'内/外|R|':>10}")
    print("-" * 84)

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
        R_a = np.abs((I_al - G).mean(axis=2)).astype(np.float32)

        T = cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        Ig = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gI, gT = grad(Ig), grad(T)

        E = ((gI >= np.percentile(gI, E_I_PCT))
             & (gT <= np.percentile(gT, E_T_PCT))).astype(np.uint8)
        Ec = cv2.morphologyEx(E, cv2.MORPH_CLOSE,
                              np.ones((CLOSE_K, CLOSE_K), np.uint8))

        # 填充边缘围成的区域
        cont, _ = cv2.findContours(Ec, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(Ec)
        cv2.drawContours(filled, cont, -1, 1, -1)
        inside = filled.astype(bool) & ~Ec.astype(bool)

        # 区域级检验：内弱外强
        n, lab, st, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8), 8)
        keep = np.zeros_like(inside)
        areas = []
        ratios = []
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if a < MIN_AREA:
                continue
            comp = (lab == i)
            ring = (cv2.dilate(comp.astype(np.uint8), np.ones((15, 15), np.uint8))
                    .astype(bool) & ~comp)
            if ring.sum() < 20:
                continue
            ri = float(np.median(R_a[comp])); ro = float(np.median(R_a[ring]))
            ratio = ri / max(ro, 1e-6)
            if ratio <= WEAK_RATIO:
                keep |= comp
                areas.append(a)
                ratios.append(ratio)
        areas.sort(reverse=True)

        inner = keep
        if ERODE_PX > 0 and inner.any():
            k = np.ones((ERODE_PX * 2 + 1, ERODE_PX * 2 + 1), np.uint8)
            e = cv2.erode(inner.astype(np.uint8), k).astype(bool)
            if e.sum() > 0:
                inner = e

        res[sc] = {"edge_pct": round(float(E.mean()) * 100, 2),
                   "inside_pct": round(float(inside.mean()) * 100, 2),
                   "hole_pct": round(float(inner.mean()) * 100, 2),
                   "n_holes": len(areas),
                   "largest": areas[0] if areas else 0,
                   "R_ratio": (round(float(np.median(ratios)), 3) if ratios else None)}
        print(f"{sc:<30}{res[sc]['edge_pct']:>7.2f}%{res[sc]['inside_pct']:>7.2f}%"
              f"{res[sc]['hole_pct']:>8.2f}%{len(areas):>6}{res[sc]['largest']:>8}"
              f"{(f'{np.median(ratios):.2f}' if ratios else '-'):>10}")

        over = I.astype(np.uint8).copy()
        over[inner] = (0.25 * over[inner] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
        kc, _ = cv2.findContours(keep.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, kc, -1, (0, 255, 0), 1)
        cells = [
            title(over, f"HOLE (enclosed by edge-not-in-T): {res[sc]['hole_pct']:.1f}%"),
            title(cv2.cvtColor(Ec * 255, cv2.COLOR_GRAY2BGR),
                  f"edge in I, not in T ({res[sc]['edge_pct']:.1f}%), closed"),
            title(turbo(R_a), f"|R| p50={np.median(R_a):.1f}"),
            title(turbo(gT), "grad T"),
        ]
        sheet = np.hstack(cells)
        ttl = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(ttl, f"{sc}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}.jpg"),
                    np.vstack([ttl, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    json.dump(res, open(os.path.join(EVAL, "hole_v11.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * 84)
    if res:
        hp = [v["hole_pct"] for v in res.values()]
        print(f"洞内部 中位 {np.median(hp):.2f}%  范围 [{min(hp):.2f}%, {max(hp):.2f}%]")
        print("（v9 逐像素判据 1.17%；本次改为区域级，面积应更大）")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
