"""把训练池的 Ω 叠到图上看：到底是"人影"，还是一片弥散暗区？

输出一张 PNG 拼图（每行一个配对）：
  列1 I（有反射）｜列2 T（黑布参考）｜列3 I 上叠 Ω（红色半透明 + 绿色轮廓）
并在每行左侧标注池名 / Ω 面积。

用法：python viz_omega_pool.py > /tmp/omega_tiles.log
"""
import glob
import json
import os

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
OUT = os.path.join(P, "eval", "fig_1019", "omega_pool_tiles.png")
POOLS = [
    ("1026-apr(人工确认)", "data/train_sets/real_pairs_1026", "apr", 2),
    ("1020-apr", "data/train_sets/real_pairs_1020", "apr", 1),
    ("1020-may", "data/train_sets/real_pairs_1020", "may", 1),
    ("1020-jun", "data/train_sets/real_pairs_1020", "jun", 1),
]
W = 460


def fit(img, w=W):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


rows, labels = [], []
for title, rel, want, n in POOLS:
    d = os.path.join(P, rel)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    keys = sorted(meta)
    keys = [k for k in keys if meta[k].get("domain") == want]
    for k in keys[:n]:
        pi = os.path.join(d, "I", k + ".png")
        pt = os.path.join(d, "T", k + ".png")
        po = os.path.join(d, "omega", k + ".png")
        if not (os.path.exists(pi) and os.path.exists(pt) and os.path.exists(po)):
            continue
        I, T, Om = cv2.imread(pi), cv2.imread(pt), cv2.imread(po, 0)
        if Om.shape != I.shape[:2]:
            Om = cv2.resize(Om, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = Om > 127
        if m.sum() == 0:
            continue
        ov = I.copy()
        red = np.zeros_like(I); red[..., 2] = 255
        ov[m] = (0.45 * I[m] + 0.55 * red[m]).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 3)
        a, b, c = fit(I), fit(T), fit(ov)
        hh = min(a.shape[0], b.shape[0], c.shape[0])
        row = np.hstack([a[:hh], b[:hh], c[:hh]])
        rows.append(row)
        labels.append(f"{title.split()[0]} | {k.split('__')[-1]} | Ω={m.mean()*100:.1f}%")
        print(f"{title:<34} {k:<44} Ω={m.mean()*100:5.1f}%")

if not rows:
    raise SystemExit("没有可画的配对")

w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
canvas = np.vstack(rows)
for i, lab in enumerate(labels):
    y = i * rows[0].shape[0] + 26
    cv2.putText(canvas, lab, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, lab, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
cv2.imwrite(OUT, canvas)
print("\n写出", OUT, canvas.shape)
