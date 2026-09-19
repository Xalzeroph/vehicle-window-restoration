#!/usr/bin/env python3
"""反射判定可视化：输入 / GT / 差异图 D / 自适应掩膜 / 块级分档（2026-09-19）。

行 = 4 个代表性场景（近零 → 最强）；列 = 5 张面板。
输出 fig/mask_examples_0919.jpg
"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import robust_gain, compute_mask  # noqa: E402

BS = 64
ROWS = [
    ('may 近零 medD=9.7', 'data/test_sets/real_pair_may_0917', 'GT.png', 'VID_20260512_162623', 'I_000975.png'),
    ('jun 剪影人影', 'data/test_sets/real_pair_0917', 'T.png', 'VID_20260609_170941', 'I.png'),
    ('jun 中强 medD=39', 'data/test_sets/real_pair_0917', 'T.png', 'VID_20260609_172143', 'I.png'),
    ('apr 最强 medD=68', 'data/test_sets/real_pair_apr_0917', 'GT.png', 'VID_20260426_173237', 'I_000013.png'),
]
H = 300


WP = 400


def fit(im, h=H, wp=WP):
    """等比缩放到高 h，再 letterbox 到固定宽 wp（各行对齐）。"""
    im = cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_AREA)
    if im.shape[1] > wp:
        im = im[:, (im.shape[1] - wp) // 2: (im.shape[1] - wp) // 2 + wp]
    canvas = np.zeros((h, wp, 3), np.uint8)
    x = (wp - im.shape[1]) // 2
    canvas[:, x:x + im.shape[1]] = im
    return canvas


def panel_text(img, lines, org=(6, 18)):
    for i, t in enumerate(lines):
        cv2.putText(img, t, (org[0], org[1] + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, t, (org[0], org[1] + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def row_panels(I, T, gt_name, ipath):
    g = robust_gain(I, T)
    D = np.abs(I.astype(np.float64) - T.astype(np.float64) * g.reshape(1, 1, 3)).max(axis=2)
    M, info, _ = compute_mask(I, T)
    thr, sig = info['thr'], info['sigma_d']
    med_d = float(np.median(D[M])) if M.sum() else 0.0

    p1 = I.copy()
    panel_text(p1, ['INPUT I'])
    p2 = T.copy()
    panel_text(p2, ['GT T (no reflection)'])
    heat = np.clip(D / (3 * thr), 0, 1)
    p3 = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    panel_text(p3, [f'D=|I-gT| (thr={thr:.0f}, 3sigma scaled)', f'sigma={sig:.1f}'])
    p4 = I.copy()
    ov = p4.copy()
    ov[M] = (0, 0, 255)
    p4 = cv2.addWeighted(ov, 0.45, p4, 0.55, 0)
    panel_text(p4, [f'MASK (red, frac={info["frac_M"]*100:.0f}%)', f'medD inside={med_d:.1f}'])
    # 块级分档
    p5 = I.copy()
    h, w = D.shape
    t_gray = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float64)
    blocks = []
    for y in range(0, h - BS + 1, BS):
        for x in range(0, w - BS + 1, BS):
            tb = t_gray[y:y + BS, x:x + BS]
            if tb.mean() < 25:
                continue
            blocks.append((y, x, float(D[y:y + BS, x:x + BS].mean() / (tb.mean() + 1e-9))))
    r75 = np.percentile([b[2] for b in blocks], 75) if blocks else 1.0
    ov = p5.copy()
    for y, x, robs in blocks:
        if robs < 0.05:
            c = (0, 200, 0)
        elif robs >= r75:
            c = (0, 0, 255)
        else:
            c = (0, 215, 255)
        ov[y:y + BS, x:x + BS] = c
    p5 = cv2.addWeighted(ov, 0.30, p5, 0.70, 0)
    panel_text(p5, ['block r_obs: green<0.05 / red=top quartile', f'p75={r75:.3f}'])
    return [fit(x.astype(np.uint8)) for x in (p1, p2, p3, p4, p5)]


def main():
    rows = []
    for name, d, gtn, scene, ifn in ROWS:
        T = cv2.imread(os.path.join(d, scene, gtn))
        I = cv2.imread(os.path.join(d, scene, ifn))
        panels = row_panels(I, T, gtn, ifn)
        label = np.zeros((H, 230, 3), np.uint8)
        for i, t in enumerate(name.replace(' ', '\n').split('\n')):
            cv2.putText(label, t, (6, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        rows.append(np.hstack([label] + panels))
    gap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, gap, r])
    os.makedirs('fig', exist_ok=True)
    cv2.imwrite('fig/mask_examples_0919.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'写出 fig/mask_examples_0919.jpg  {out.shape[1]}x{out.shape[0]}')


if __name__ == '__main__':
    main()
