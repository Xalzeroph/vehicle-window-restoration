#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检验"反射洞"的数据驱动定义：用**时间稳定性**替代未知的反射真值（2026-10-20）。

核心物理
--------
每个场景是**同一机位、同一场景**连续拍摄：
  · 透射层 T（窗后物体）→ **全程静止**
  · 反射层 R（操作者/身后环境）→ 随人走动而**变化**
所以对每个像素算**时间标准差**：
  τ(p) 小 → 该像素全程几乎不变 → 反射贡献很小（**弱反射/洞候选**）
  τ(p) 大 → 有东西在变 → 该处有反射（或运动）

再叠加结构项：洞必须是"**透射侧有强结构**"的地方（否则只是平坦的天空/墙面）。
  洞候选 = { τ 小 } ∧ { 梯度强 } ∧ { 偏暗（可选） }

这个定义**不依赖任何模型、不依赖 GT**，可用原始视频独立验证。

输出：_review_1019/holes/<scene>_temporal.jpg（τ 图 + 梯度图 + 洞掩膜叠加）
     _review_1019/holes/temporal_stats.json
"""
from __future__ import annotations

import glob
import json
import os
import re

import cv2
import numpy as np

RAW = "/DataDisk/zmy_code/TintedGlass_Enhance/data/raw_1019"
OUT = "/DataDisk/zmy_code/TintedGlass_Enhance/_review_1019/holes"
MAXN = 40          # 每场景最多抽多少帧算时间统计
MAXW = 480         # 统计时降采样宽度（省内存/时间）


def frames_of(d):
    out = []
    for f in os.listdir(d):
        m = re.match(r"frame_(\d{6})\.jpg$", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    scenes = sorted(s for s in os.listdir(RAW) if s.startswith("VID_20261019_"))
    stats = {}
    for sc in scenes:
        d = os.path.join(RAW, sc)
        fr = frames_of(d)
        if len(fr) < 10:
            continue
        # 均匀抽 MAXN 帧（覆盖整个视频，包含布段——布段会被时间稳定性误判，稍后靠掩膜排除）
        idx = np.linspace(0, len(fr) - 1, min(MAXN, len(fr))).astype(int)
        stack = []
        for i in idx:
            im = cv2.imread(os.path.join(d, fr[i][1]), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            h, w = im.shape
            sw = MAXW
            sh = max(2, int(round(h * sw / w)))
            stack.append(cv2.resize(im, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32))
        if len(stack) < 10:
            continue
        S = np.stack(stack)                    # (n,H,W)
        tau = S.std(axis=0)                    # 时间标准差
        mean = S.mean(axis=0)
        g = cv2.GaussianBlur(mean, (5, 5), 0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        grad = np.sqrt(gx * gx + gy * gy)

        # 洞候选：τ 低（前 30%） ∧ 梯度强（前 30%） ∧ 不特别亮（中位以下）
        t_thr = np.percentile(tau, 30)
        g_thr = np.percentile(grad, 70)
        m_thr = np.median(mean)
        hole = (tau <= t_thr) & (grad >= g_thr) & (mean <= m_thr)

        st = {
            "scene": sc, "n_used": len(stack), "size": [stack[0].shape[1], stack[0].shape[0]],
            "tau_p10": round(float(np.percentile(tau, 10)), 3),
            "tau_p50": round(float(np.percentile(tau, 50)), 3),
            "tau_p90": round(float(np.percentile(tau, 90)), 3),
            "grad_p50": round(float(np.percentile(grad, 50)), 2),
            "hole_frac": round(float(hole.mean()), 4),
            "hole_pixels": int(hole.sum()),
        }
        stats[sc] = st

        # 可视化：τ（归一化） / 梯度（归一化） / 洞掩膜叠在原图上
        def norm(x, p=(1, 99)):
            lo, hi = np.percentile(x, p)
            hi = max(hi, lo + 1e-6)
            return np.clip((x - lo) / (hi - lo), 0, 1)

        vis_tau = (norm(tau) * 255).astype(np.uint8)
        vis_grad = (norm(grad) * 255).astype(np.uint8)
        base = cv2.cvtColor(mean.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        over = base.copy()
        over[hole] = (0.35 * over[hole] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        # 加标题条
        def strip(img, t):
            hh, ww = img.shape[:2]
            bar = np.zeros((22, ww, 3), np.uint8)
            cv2.putText(bar, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            return np.vstack([bar, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img])
        a = strip(vis_tau, "temporal std tau (low=static)")
        b = strip(vis_grad, "gradient (high=structure)")
        c = strip(over, f"hole candidates {hole.mean()*100:.1f}% (red)")
        hh = min(x.shape[0] for x in (a, b, c))
        sheet = np.hstack([a[:hh], b[:hh], c[:hh]])
        title = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(title, f"{sc}   hole = low-tau & high-grad & not-bright",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"{sc}_temporal.jpg"),
                    np.vstack([title, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])

    json.dump(stats, open(os.path.join(OUT, "temporal_stats.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{'场景':<30}{'τ中位':>8}{'梯度中位':>10}{'洞占比':>9}")
    print("-" * 62)
    for sc, s in stats.items():
        print(f"{sc:<30}{s['tau_p50']:>8.2f}{s['grad_p50']:>10.2f}{s['hole_frac']*100:>8.2f}%")
    print(f"\n输出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
