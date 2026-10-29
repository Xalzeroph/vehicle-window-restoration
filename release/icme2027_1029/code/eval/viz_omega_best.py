"""挑“遮挡最强”的配对来看：按 (I/T)_Ω内 − (I/T)_Ω外 排序（越负 ⇒ 该处被压暗越多 ⇒ 越像真有人影挡光）。

输出 eval/fig_1019/omega_best_tiles.png，每行 I | T | I⊞Ω，标签含 Ω 面积与压暗量。
"""
import json
import os

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
OUT = os.path.join(P, "eval", "fig_1019", "omega_best_tiles.png")
POOLS = [("1026", "data/train_sets/real_pairs_1026"),
         ("1020", "data/train_sets/real_pairs_1020")]
TOPN = 6
W = 430


def fit(img, w=W):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


cands = []
for tag, rel in POOLS:
    d = os.path.join(P, rel)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    for k in sorted(meta):
        pi = os.path.join(d, "I", k + ".png")
        pt = os.path.join(d, "T", k + ".png")
        po = os.path.join(d, "omega", k + ".png")
        if not (os.path.exists(pi) and os.path.exists(pt) and os.path.exists(po)):
            continue
        I = cv2.imread(pi).astype(np.float32)
        T = cv2.imread(pt).astype(np.float32)
        Om = cv2.imread(po, 0)
        if I is None or T is None or Om is None or I.shape != T.shape:
            continue
        if Om.shape != I.shape[:2]:
            Om = cv2.resize(Om, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = Om > 127
        if m.sum() < 500 or (~m).sum() < 500:
            continue
        gi = I.mean(axis=2); gt = T.mean(axis=2)
        valid = gt > 20
        r = np.where(valid, gi / np.maximum(gt, 1e-3), np.nan)
        din = float(np.nanmean(r[m])); dout = float(np.nanmean(r[~m]))
        cands.append(dict(tag=tag, key=k, d=din - dout, area=float(m.mean()),
                          din=din, dout=dout, dir=os.path.join(d, ""), pi=pi, pt=pt, po=po))

cands.sort(key=lambda c: c["d"])
sel = cands[:TOPN]
print(f"候选 {len(cands)} 对，取压暗最强的 {len(sel)} 对：\n")
rows, labels = [], []
for c in sel:
    I = cv2.imread(c["pi"]); T = cv2.imread(c["pt"]); Om = cv2.imread(c["po"], 0)
    if Om.shape != I.shape[:2]:
        Om = cv2.resize(Om, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = Om > 127
    ov = I.copy(); red = np.zeros_like(I); red[..., 2] = 255
    ov[m] = (0.45 * I[m] + 0.55 * red[m]).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 3)
    a, b, cc = fit(I), fit(T), fit(ov)
    hh = min(a.shape[0], b.shape[0], cc.shape[0])
    rows.append(np.hstack([a[:hh], b[:hh], cc[:hh]]))
    lab = f"{c['tag']} {c['key'].split('__')[0]} {c['key'].split('__')[-1]} | Om={c['area']*100:.1f}% | I/T in={c['din']:.3f} out={c['dout']:.3f} d={c['d']:+.3f}"
    labels.append(lab)
    print("  " + lab)

w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
canvas = np.vstack(rows)
step = rows[0].shape[0]
for i, lab in enumerate(labels):
    y = i * step + 26
    cv2.putText(canvas, lab, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, lab, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(OUT, canvas)
print("\n写出", OUT, canvas.shape)
