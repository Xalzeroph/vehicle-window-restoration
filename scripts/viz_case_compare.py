#!/usr/bin/env python3
"""两个案例可视化（2026-09-19）：
A. fig/case_rdnet_vs_gt_0919.jpg —— 输入 / RDNet_gm / GT + 掩膜区放大（"掩膜内变好、感知变差"）
B. fig/case_local_highlight_0919.jpg —— 5 月金属高光局部强反射（局部 r_obs=0.73 vs 中位 0.16）
"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import robust_gain, compute_mask, ls_scale, apply_gm  # noqa: E402

H, WP = 300, 400
BS = 64


def fit(im, h=H, wp=WP):
    im = cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_AREA)
    if im.shape[1] > wp:
        x0 = (im.shape[1] - wp) // 2
        im = im[:, x0:x0 + wp]
    canvas = np.zeros((h, wp, 3), np.uint8)
    x = (wp - im.shape[1]) // 2
    canvas[:, x:x + im.shape[1]] = im
    return canvas


def label_col(lines, h=H, w=250):
    img = np.zeros((h, w, 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (6, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def tag(img, txt, org=(6, 20)):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def hstack_rows(rows):
    gap = np.full((8, rows[0].shape[1], 3), 235, np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, gap, r])
    return out


# ---------- A. RDNet vs GT ----------
def fig_a():
    d = 'data/test_sets/real_pair_0917'
    sc = 'VID_20260609_173727'
    T = cv2.imread(f'{d}/{sc}/T.png')
    I = cv2.imread(f'{d}/{sc}/I.png')
    P = cv2.imread(f'runs/pred/real_pair_0917_rdnet/{sc}/ytmt_ucs_sirs_l.png')
    P = cv2.resize(P, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
    Pg = apply_gm(P, ls_scale(P, T)).astype(np.uint8)
    M, info, D = compute_mask(I, T)
    # 最大连通域 bbox 外扩
    n, _, stats, _ = cv2.connectedComponentsWithStats(M.astype(np.uint8), 8)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = stats[i][:4]
    else:
        x, y, w, h = 0, 0, T.shape[1], T.shape[0]
    pad = 80
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(T.shape[1], x + w + pad), min(T.shape[0], y + h + pad)

    ov = I.copy(); ov[M] = (0, 0, 255)
    ov = cv2.addWeighted(ov, 0.45, I, 0.55, 0)
    row1 = [cv2.cvtColor(fit(I), cv2.COLOR_BGR2RGB), cv2.cvtColor(fit(Pg), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(fit(T), cv2.COLOR_BGR2RGB), cv2.cvtColor(fit(ov), cv2.COLOR_BGR2RGB)]
    # 放大行（同窗口裁剪）
    zI, zP, zT, zM = (cv2.cvtColor(fit(im[y0:y1, x0:x1]), cv2.COLOR_BGR2RGB)
                      for im in (I, Pg, T, ov[y0:y1, x0:x1]))
    row2 = [zI, zP, zT, zM]
    t1 = [tag(p, s) for p, s in zip(row1, ['INPUT I', 'RDNET_gm', 'GT T (target)', 'MASK on input'])]
    t2 = [tag(p, s) for p, s in zip(row2, ['zoom: INPUT', 'zoom: RDNET_gm', 'zoom: GT', 'zoom: MASK on input'])]
    lab = label_col(['jun VID_...173727', f'thr={info["thr"]:.0f} fracM={info["frac_M"]*100:.0f}%',
                     'd_perc = +0.084 (worse)', 'd_struct = -1.59 (better)', '', 'row1: full', 'row2: mask zoom'])
    out = hstack_rows([np.hstack([lab, *t1]), np.hstack([lab, *t2])])
    cv2.imwrite('fig/case_rdnet_vs_gt_0919.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('写出 fig/case_rdnet_vs_gt_0919.jpg', out.shape)


# ---------- B. 局部金属高光 ----------
def fig_b():
    d = 'data/test_sets/real_pair_may_0917'
    sc = 'VID_20260512_162901'
    T = cv2.imread(f'{d}/{sc}/GT.png')
    I = cv2.imread(f'{d}/{sc}/I_000875.png')
    g = robust_gain(I, T)
    D = np.abs(I.astype(np.float64) - T.astype(np.float64) * g.reshape(1, 1, 3)).max(axis=2)
    M, info, _ = compute_mask(I, T)
    heat = cv2.applyColorMap((np.clip(D / (2 * info['thr']), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    ov = I.copy(); ov[M] = (0, 0, 255)
    ov = cv2.addWeighted(ov, 0.45, I, 0.55, 0)
    # 局部最强块位置（实测 (y=768, x=1664) 附近）
    yb, xb = 768, 1664
    xi = I.copy()
    cv2.rectangle(xi, (xb - 64, yb - 64), (xb + 128, yb + 128), (0, 255, 255), 4)
    row1 = [cv2.cvtColor(fit(xi), cv2.COLOR_BGR2RGB), cv2.cvtColor(fit(heat), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(fit(ov), cv2.COLOR_BGR2RGB)]
    pad = 160
    x0, y0 = max(0, xb - pad), max(0, yb - pad)
    x1, y1 = min(I.shape[1], xb + 64 + pad), min(I.shape[0], yb + 64 + pad)
    row2 = [cv2.cvtColor(fit(I[y0:y1, x0:x1]), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(fit(heat[y0:y1, x0:x1]), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(fit(ov[y0:y1, x0:x1]), cv2.COLOR_BGR2RGB)]
    t1 = [tag(p, s) for p, s in zip(row1, ['INPUT I (yellow=highlight)', 'D heatmap (2*thr scale)', 'MASK on input'])]
    t2 = [tag(p, s) for p, s in zip(row2, ['zoom: INPUT', 'zoom: D heatmap', 'zoom: MASK'])]
    lab = label_col(['may VID_...162901 I_000875', f'thr={info["thr"]:.0f} fracM={info["frac_M"]*100:.0f}%',
                     'local block r_obs=0.73', 'image median r_obs=0.16', '(max/median = 4.5x)',
                     'global mean would miss it'])
    out = hstack_rows([np.hstack([lab, *t1]), np.hstack([lab, *t2])])
    cv2.imwrite('fig/case_local_highlight_0919.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('写出 fig/case_local_highlight_0919.jpg', out.shape)


if __name__ == '__main__':
    fig_a()
    fig_b()
