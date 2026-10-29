"""阈值扫描可视化：把 q = I/T 的阈值 δ 从紧到松扫一遍，看哪个 δ 圈出的区域"像人眼认定的遮挡区"。

每个域取"压暗最强"的一对（最可能有人影），每行：
   原图 I ｜ δ=0.95 ｜ δ=1.00 ｜ δ=1.05 ｜ δ=1.10 ｜ δ=1.15
掩码只画**绿色轮廓**（不填色），避免遮住底层内容；每格标注该 δ 下的面积。
掩码流程与 v13 同构：q = luma(I)/luma(T) → 11×11 平滑 → q < δ → 形态学开闭 → 去掉 <0.5% 的小块。

用法：python viz_threshold_sweep.py
"""
import glob
import json
import os

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
OUT = os.path.join(P, "eval", "fig_1019", "threshold_sweep.png")
BUDGETS = [3, 6, 10, 15, 25]   # 面积预算（%），δ 取 q 的对应分位
TILE = 330
DOMAINS = [("1026-apr", "data/train_sets/real_pairs_1026", "apr"),
           ("1020-apr", "data/train_sets/real_pairs_1020", "apr"),
           ("1020-may", "data/train_sets/real_pairs_1020", "may"),
           ("1020-jun", "data/train_sets/real_pairs_1020", "jun")]


def luma(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)


def qfield(I, T):
    """逐帧归一化的 I/T：中位归一到 1，使 δ 可跨域比较（不同帧的曝光/增益差被消掉）。"""
    q = luma(I) / np.maximum(luma(T), 1.0)
    q = cv2.blur(q, (11, 11))
    med = float(np.median(q))
    return q / max(med, 1e-6)


def mask_of(q, d):
    m = (q < d).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m, bool)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] / m.size >= 0.005:
            keep |= (lab == i)
    return keep


def fit(img, w=TILE):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def strongest(rel, dom):
    d = os.path.join(P, rel)
    meta = json.load(open(os.path.join(d, "meta.json")))
    best, bk = -9e9, None
    for k in sorted(meta):
        if meta[k].get("domain") != dom:
            continue
        pi, pt = os.path.join(d, "I", k + ".png"), os.path.join(d, "T", k + ".png")
        if not (os.path.exists(pi) and os.path.exists(pt)):
            continue
        I, T = cv2.imread(pi), cv2.imread(pt)
        if I is None or T is None or I.shape != T.shape:
            continue
        q = qfield(I, T)
        n = int(q.size * 0.2)
        s = float(np.sort(q.ravel())[:n].mean())
        if -s > best:
            best, bk = -s, k
    return d, bk


rows, rowlabels = [], []
for name, rel, dom in DOMAINS:
    d, k = strongest(rel, dom)
    if k is None:
        continue
    I = cv2.imread(os.path.join(d, "I", k + ".png"))
    T = cv2.imread(os.path.join(d, "T", k + ".png"))
    q = qfield(I, T)
    tiles = [fit(I)]
    for k_pct in BUDGETS:
        dv = float(np.percentile(q, k_pct))
        m = mask_of(q, dv)
        ov = I.copy()
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 4)
        t = fit(ov)
        lab = f"top{k_pct}%  d={dv:.3f}  {m.mean()*100:.1f}%"
        cv2.putText(t, lab, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, lab, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    hh = min(t.shape[0] for t in tiles)
    rows.append(np.hstack([t[:hh] for t in tiles]))
    rowlabels.append(f"{name}  {k.split('__')[-1]}")
    print(f"{name:<10} {k:<44} 面积: " + " ".join(f"{mask_of(q, float(np.percentile(q, b))).mean()*100:5.1f}%" for b in BUDGETS))

w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
canvas = np.vstack(rows)
step = rows[0].shape[0]
for i, lab in enumerate(rowlabels):
    y = i * step + 34
    cv2.putText(canvas, lab, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, lab, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(OUT, canvas)
print("写出", OUT, canvas.shape)
