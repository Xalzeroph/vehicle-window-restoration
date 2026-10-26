#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞检测 v5 —— **物体级尺度**（2026-10-20）。

v4 的问题
---------
v4 在**细纹理尺度**上找暗斑，结果全是树叶枝条之间的小碎块，不是"树影/人影"。
洞是**物体级**的遮挡轮廓，必须先在大尺度上抹平纹理再找。

v5 做法（按"反射层里的暗遮挡物"重新设计）
---------------------------------------
1. `R̂ = |I_a − T|` 反射层估计；
2. **尺度空间**：在 σ ∈ {12, 24, 48} 三个尺度上各算一次 `R̂_bg`，
   取"最佳尺度"= 使 `log((R̂_bg+c)/(R̂+c))` 的**空间连通性**最好的那个
   （用暗斑最大连通块面积占比作判据）；
3. 暗斑判据：`R̂ <= α·R̂_bg`（α=0.55）且 `R̂ < 该图中位`；
4. 形态学：先开（去碎点）再闭（补内洞），最小连通块 800px；
5. **边界清晰度**：暗斑边界带上 ∇R̂ 的**分位数**（相对全图），
   要求 >= 全图 70 分位（说明边界像"物体轮廓"，而不是渐变）。

输出
----
_review_1019/holesB5/<scene>_hole_v5.jpg
eval/hole_b5_stats.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/holesB5"
EVAL = "/DataDisk/zmy_code/TintedGlass_Enhance/eval"
MAXW = 720
SIGMAS = (12, 24, 48)
ALPHA = 0.55
MIN_AREA = 800
SCENES = ["VID_20261019_165126", "VID_20261019_165530", "VID_20261019_170406",
          "VID_20261019_164257", "VID_20261019_165849", "VID_20261019_170758"]


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


def components(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask)
    areas = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            out |= (lab == i)
            areas.append(a)
    areas.sort(reverse=True)
    return out, areas


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    res = {}
    hdr = (f"{'场景':<30}{'尺度σ':>7}{'暗斑占比':>9}{'连通域':>7}"
           f"{'最大块':>9}{'边界分位':>9}")
    print(hdr); print("-" * len(hdr))

    for sc in SCENES:
        d = os.path.join(PAIR, sc)
        if not os.path.isdir(d):
            continue
        I = load(os.path.join(d, "I_000000.png"))
        G = load(os.path.join(d, "GT.png"))
        if I is None or G is None:
            continue
        h, w = I.shape[:2]
        sw = MAXW; sh = max(2, int(round(h * sw / w)))
        I = cv2.resize(I, (sw, sh), interpolation=cv2.INTER_AREA)
        G = cv2.resize(G, (sw, sh), interpolation=cv2.INTER_AREA)

        A, B = robust_affine(I, G)
        I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)
        R = np.abs(I_al - G).mean(axis=2).astype(np.float32)
        c0 = float(np.median(R))
        gy_, gx_ = np.gradient(R)
        gR = np.sqrt(gx_ ** 2 + gy_ ** 2)
        g_med = float(np.median(gR))

        best = None
        for sig in SIGMAS:
            R_bg = cv2.GaussianBlur(R, (0, 0), sig)
            dark = (R <= ALPHA * R_bg) & (R <= np.percentile(R, 60))
            m = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((15, 15), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                                 np.ones((21, 21), np.uint8)).astype(bool)
            m, areas = components(m, MIN_AREA)
            # 边界分位：边界带上 ∇R̂ 在**全图**中的分位
            ker = np.ones((21, 21), np.uint8)
            band = (cv2.dilate(m.astype(np.uint8), ker).astype(bool)
                    & ~cv2.erode(m.astype(np.uint8), ker).astype(bool))
            if band.any() and areas:
                gband = float(np.median(gR[band]))
                pct = float((gR < gband).mean()) * 100
            else:
                pct = 0.0
            cand = {"sigma": sig, "frac": float(m.mean()),
                    "n": len(areas), "largest": areas[0] if areas else 0,
                    "edge_pct": pct, "mask": m,
                    "dip": np.log((R_bg + c0) / (R + c0)).astype(np.float32)}
            score = cand["frac"] * (cand["edge_pct"] / 100.0)
            if best is None or score > best.get("score", -1):
                cand["score"] = score
                best = cand

        m = best["mask"]
        res[sc] = {"R_median": round(float(np.median(R)), 3),
                   "best_sigma": best["sigma"],
                   "hole_frac": round(best["frac"], 4),
                   "n_comp": best["n"], "largest": best["largest"],
                   "edge_pct": round(best["edge_pct"], 1),
                   "pass_edge70": bool(best["edge_pct"] >= 70)}
        print(f"{sc:<30}{best['sigma']:>7}{best['frac']*100:>8.2f}%"
              f"{best['n']:>7}{best['largest']:>9}{best['edge_pct']:>9.1f}")

        # ---- 可视化 ----
        def nz(x, p=(2, 98)):
            lo, hi = np.percentile(x, p); hi = max(hi, lo + 1e-9)
            return np.clip((x - lo) / (hi - lo), 0, 1)
        base = G.astype(np.uint8)
        over = base.copy()
        over[m] = (0.30 * over[m] + 0.70 * np.array([0, 0, 255])).astype(np.uint8)
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cont, -1, (0, 255, 0), 2)
        cells = []
        for nm, img in (
            ("R_hat (reflection layer)", cv2.applyColorMap(
                (nz(R) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)),
            (f"dip @ sigma={best['sigma']}", cv2.applyColorMap(
                (nz(best['dip']) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)),
            (f"HOLE (object-scale): {best['frac']*100:.1f}%  "
             f"edge pct={best['edge_pct']:.0f}", over),
        ):
            ww = img.shape[1]
            bar = np.zeros((20, ww, 3), np.uint8)
            cv2.putText(bar, nm, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(np.vstack([bar, img]))
        hh = min(c.shape[0] for c in cells)
        sheet = np.hstack([c[:hh] for c in cells])
        title = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(title, f"{sc}  HOLE v5 = object-scale dark region in the REFLECTION layer",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_hole_v5.jpg"),
                    np.vstack([title, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])

    json.dump(res, open(os.path.join(EVAL, "hole_b5_stats.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-" * len(hdr))
    if res:
        fr = [v["hole_frac"] for v in res.values()]
        ok = sum(1 for v in res.values() if v["pass_edge70"])
        print(f"暗斑占比 中位 {np.median(fr)*100:.2f}%   "
              f"边界分位>=70 的场景 {ok}/{len(res)}")
    print(f"输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
