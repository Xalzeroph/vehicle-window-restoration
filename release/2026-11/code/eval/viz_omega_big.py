"""大图目检：每行 I | T | Ω 叠加 | 反射残差(×4 增强)，便于人眼判断遮挡区在哪。

用法：python viz_omega_big.py
"""
import json
import os

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
OUT = os.path.join(P, "eval", "fig_1019", "omega_big.png")
W = 560

# (标签, 池, key 前缀筛选)
PICK = [
    ("1026 max-Om", "data/train_sets/real_pairs_1026", "max"),
    ("1020 may f000400", "data/train_sets/real_pairs_1020", "may__VID_20260512_162203__f000400"),
    ("1020 apr f000525", "data/train_sets/real_pairs_1020", "apr__VID_20260426_172719__f000525"),
]


def fit(img, w=W):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def load_row(d, k):
    I = cv2.imread(os.path.join(d, "I", k + ".png"))
    T = cv2.imread(os.path.join(d, "T", k + ".png"))
    Om = cv2.imread(os.path.join(d, "omega", k + ".png"), 0)
    if I is None or T is None or Om is None:
        return None
    if Om.shape != I.shape[:2]:
        Om = cv2.resize(Om, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
    return I, T, Om > 127


rows, labels = [], []
for label, rel, key in PICK:
    d = os.path.join(P, rel)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    if key == "max":
        best, bk = -1, None
        for k in sorted(meta):
            r = load_row(d, k)
            if r is None:
                continue
            a = float(r[2].mean())
            if a > best:
                best, bk = a, k
        k = bk
    else:
        k = key
    r = load_row(d, k)
    if r is None:
        print("跳过", label, k)
        continue
    I, T, m = r
    ov = I.copy(); red = np.zeros_like(I); red[..., 2] = 255
    ov[m] = (0.4 * I[m] + 0.6 * red[m]).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 3)
    res = np.abs(I.astype(np.float32) - T.astype(np.float32)).mean(axis=2)
    res = np.clip(res * 4, 0, 255).astype(np.uint8)
    res = cv2.applyColorMap(res, cv2.COLORMAP_INFERNO)
    panels = [fit(x) for x in (I, T, ov, res)]
    hh = min(p.shape[0] for p in panels)
    rows.append(np.hstack([p[:hh] for p in panels]))
    lab = f"{label} | {k.split('__')[-1]} | Om={m.mean()*100:.1f}%"
    labels.append(lab)
    print(" ", lab)

w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
canvas = np.vstack(rows)
step = rows[0].shape[0]
for i, lab in enumerate(labels):
    y = i * step + 30
    cv2.putText(canvas, lab, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, lab, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(OUT, canvas)
print("写出", OUT, canvas.shape)
