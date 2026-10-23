#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反射洞判定（方案 B）：先用 f0 对做逐通道仿射对齐去掉曝光偏置，再判洞。

物理基础（为什么仿射对齐是"对"的）
--------------------------------
    I = a·(T + r·T) = a·T·(1 + r)        a = 曝光/色调增益, r = 反射-透射比
逐通道仿射对齐求 (a, b) 后：
    R_norm = (I − b)/a − T = r·T
即 **R_norm 是"反射-透射比 r 乘以透射层"**，与曝光 a 无关。
→ 幅度按局部亮度归一化（R_rel = R_norm / T），得到的就是 **r 本身**：
    r = (I/a − T) / T
这个量才是"反射有多强"的真正物理量，且**与曝光偏置解耦**。

洞的判据（三条件，全部可测）
--------------------------
  1. **弱反射** : r(p) 小               —— 该处几乎没有反射信息（洞的核心）
  2. **强透射结构** : |∇T(p)| 大        —— 有真实物体结构（否则只是平坦天空/墙）
  3. **偏暗** : T(p) 低于该图中位        —— 与你说的"物体遮蔽环境光→漫反射黑洞"一致

⚠️ 已知局限（必须写进论文的 limitation）
  · "静止深色物体"与"洞"在图像上**可能同形**（都是暗、都有结构、都可能 r 小）。
    本方法无法从单张图把两者分开 —— 这需要在论文里明确定义为
    "**反射弱 + 结构强 + 偏暗的区域**"，而不是断言"物理上的洞"。
  · r 的估计依赖 GT 与 f0 的几何对齐；本数据机位基本未动，已用相位相关校核过。

输出
----
_review_1019/holesB/<scene>_hole.jpg   （对齐后 r 图 / ∇T / 洞掩膜叠加）
eval/hole_b_stats.json
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

PAIR = "/DataDisk/zmy_code/TintedGlass_Enhance/data/test_sets/real_pair_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/holesB"
EVAL = "/DataDisk/zmy_code/TintedGlass_Enhance/eval"

MAXW = 640          # 统计用降采样宽度
R_LO = 30           # 取 r 最小的 30% 作为"弱反射"
G_HI = 70           # 取梯度最强的 30% 作为"强结构"
T_MED = 50          # 取亮度低于中位


def load(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im.astype(np.float64) if im is not None else None


def robust_affine(src, ref, keep=0.7, iters=3):
    """逐通道稳健仿射：src ≈ a*ref + b，返回 (a[3], b[3])。用迭代+截尾抗反射干扰。"""
    A = np.ones(3); B = np.zeros(3)
    for c in range(3):
        x = src[..., c].ravel().astype(np.float64)
        y = ref[..., c].ravel().astype(np.float64)
        for _ in range(iters):
            # 去掉残差最大的部分（那部分更可能是反射，不是曝光）
            V = np.vstack([y, np.ones_like(y)]).T
            coef, *_ = np.linalg.lstsq(V, x, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            r = np.abs(x - (a * y + b))
            k = r <= np.percentile(r, keep * 100)
            x, y = x[k], y[k]
            if len(x) < 500:
                break
        A[c], B[c] = a, b
    return A, B


def phase_shift(a_gray, b_gray):
    win = cv2.createHanningWindow((a_gray.shape[1], a_gray.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a_gray.astype(np.float32),
                                     b_gray.astype(np.float32), win)
    return float(dx), float(dy)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(EVAL, exist_ok=True)
    scenes = sorted(d for d in os.listdir(PAIR) if os.path.isdir(os.path.join(PAIR, d)))
    stats = {}
    print(f"{'场景':<30}{'|g-1|':>8}{'位移px':>8}{'r中位':>8}{'洞占比':>9}{'r_p10':>8}{'r_p90':>8}")
    print("-" * 84)

    for sc in scenes:
        I = load(os.path.join(PAIR, sc, "I_000000.png"))
        G = load(os.path.join(PAIR, sc, "GT.png"))
        if I is None or G is None:
            continue
        # 统一降采样（只用于统计与出图，不影响结论）
        h, w = I.shape[:2]
        sw = MAXW; sh = max(2, int(round(h * sw / w)))
        I = cv2.resize(I, (sw, sh), interpolation=cv2.INTER_AREA)
        G = cv2.resize(G, (sw, sh), interpolation=cv2.INTER_AREA)

        # 几何校核
        dx, dy = phase_shift(cv2.cvtColor(G.astype(np.uint8), cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY))
        shift = float(np.hypot(dx, dy))

        # 逐通道稳健仿射：I ≈ a*G + b  （注意前面推导里 I 是关于 T 的函数）
        A, B = robust_affine(I, G)
        gain_dev = float(np.abs(A - 1).mean())

        # r = (I/a − T) / T ；用 (I-b)/a 作为"去除曝光后的输入"
        I_al = (I - B.reshape(1, 1, 3)) / A.reshape(1, 1, 3)
        T = G
        eps = 8.0
        r = (I_al - T) / np.maximum(T, eps)          # 反射-透射比，逐像素
        r_gray = r.mean(axis=2)
        r_abs = np.abs(r_gray)

        Tg = cv2.cvtColor(T.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(Tg, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(Tg, cv2.CV_32F, 0, 1)
        grad = np.sqrt(gx * gx + gy * gy)
        Tlum = cv2.cvtColor(T.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        # ---- 三条件判洞 ----
        weak = r_abs <= np.percentile(r_abs, R_LO)                    # 弱反射
        strong = grad >= np.percentile(grad, G_HI)                    # 强透射结构
        dark = Tlum <= np.percentile(Tlum, T_MED)                     # 偏暗
        hole = weak & strong & dark

        st = {"scene": sc, "shift_px": round(shift, 3),
              "gain_chan": [round(float(x), 4) for x in A],
              "gain_dev": round(gain_dev, 4),
              "r_median": round(float(np.median(r_abs)), 4),
              "r_p10": round(float(np.percentile(r_abs, 10)), 4),
              "r_p90": round(float(np.percentile(r_abs, 90)), 4),
              "hole_frac": round(float(hole.mean()), 4)}
        stats[sc] = st
        print(f"{sc:<30}{gain_dev:>8.4f}{shift:>8.3f}{st['r_median']:>8.4f}"
              f"{st['hole_frac']*100:>8.2f}%{st['r_p10']:>8.4f}{st['r_p90']:>8.4f}")

        # ---- 可视化 ----
        def norm(x, p=(1, 99)):
            lo, hi = np.percentile(x, p); hi = max(hi, lo + 1e-9)
            return np.clip((x - lo) / (hi - lo), 0, 1)

        vis_r = (norm(r_abs) * 255).astype(np.uint8)
        vis_g = (norm(grad) * 255).astype(np.uint8)
        base = T.astype(np.uint8)
        over = base.copy()
        over[hole] = (0.35 * over[hole] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)

        def strip(img, t):
            ww = img.shape[1]
            bar = np.zeros((22, ww, 3), np.uint8)
            cv2.putText(bar, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            g3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
            return np.vstack([bar, g3])
        cells = [strip(vis_r, f"r = |(I/a-T)/T|  (weak=low)  med={st['r_median']:.3f}"),
                 strip(vis_g, "grad(T)  (strong=high)"),
                 strip(over, f"HOLE = weak&strong&dark : {hole.mean()*100:.1f}%")]
        hh = min(c.shape[0] for c in cells)
        sheet = np.hstack([c[:hh] for c in cells])
        title = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(title, f"{sc}  Method B: affine-aligned r, shift={shift:.2f}px, "
                           f"gain_dev={gain_dev:.3f}", (6, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_hole.jpg"),
                    np.vstack([title, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])

    json.dump(stats, open(os.path.join(EVAL, "hole_b_stats.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    if stats:
        fr = [s["hole_frac"] for s in stats.values()]
        rd = [s["r_median"] for s in stats.values()]
        print("-" * 84)
        print(f"洞占比 中位 {np.median(fr)*100:.2f}%  范围 "
              f"[{min(fr)*100:.2f}%, {max(fr)*100:.2f}%]")
        print(f"r 中位 中位 {np.median(rd):.4f}")
    print(f"\n输出 {OUT}\n      {EVAL}/hole_b_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
