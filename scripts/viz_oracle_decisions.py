#!/usr/bin/env python3
"""上帝探针判定拼图（2026-09-20）：输入 / 后端 / 上帝模式 / GT / 逐组件决策。
组件框颜色：绿=动手（标 w），灰=不动。输出 fig/oracle_decisions_0920.jpg
"""
import csv, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import compute_mask, crop_components  # noqa: E402

H, WP = 300, 400
SETS = {
    'jun': ('data/test_sets/real_pair_0917', 'T.png', 'runs/pred/real_pair_0917_rdnet', 'runs/pred/real_pair_0917_oraclev2'),
    'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png', 'runs/pred/apr_rdnet', 'runs/pred/apr_oraclev2'),
    'may': ('data/test_sets/real_pair_may_0917', 'GT.png', 'runs/pred/may_rdnet', 'runs/pred/may_oraclev2'),
}
# 案例：may 动手、may 不动、jun、apr
CASES = [('may', 'VID_20260512_162203__I_000000'), ('may', 'VID_20260512_163859__I_000000'),
         ('jun', 'VID_20260609_170941'), ('apr', 'VID_20260426_173827__I_000009')]


def fit(im, h=H, wp=WP):
    im = cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, wp, 3), np.uint8)
    if im.shape[1] > wp:
        x0 = (im.shape[1] - wp) // 2
        im = im[:, x0:x0 + wp]
    x = (wp - im.shape[1]) // 2
    canvas[:, x:x + im.shape[1]] = im
    return canvas


def tag(img, txt, org=(6, 20)):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    choices = {}
    for r in csv.DictReader(open('eval/eval_probe_oracle_v3_choices.csv')):
        choices.setdefault((r['set'], r['case']), []).append(r)
    rows = []
    for tag_, key in CASES:
        d, gtn, pred_root, ov2_root = SETS[tag_]
        sc = key.split('__')[0]
        ip = os.path.join(d, sc, 'I.png' if '__' not in key else key.split('__')[1] + '.png')
        T = cv2.imread(os.path.join(d, sc, gtn)); I = cv2.imread(ip)
        P = cv2.imread(os.path.join(pred_root, key, 'ytmt_ucs_sirs_l.png'))
        O = cv2.imread(os.path.join(ov2_root, key, 'oraclev2.png'))
        if any(x is None for x in (T, I, P, O)):
            print(f'{key}: 缺文件'); continue
        for x in (P, O):
            if x.shape != T.shape:
                x = cv2.resize(x, (T.shape[1], T.shape[0]))
        dec = I.copy()
        M, _, _ = compute_mask(I, T)
        _, lab, _, _ = cv2.connectedComponentsWithStats(M.astype(np.uint8), 8)
        cs = choices.get((tag_, key), [])
        ov = dec.copy()
        n_act = 0
        for c in cs:
            x0, y0, w0, h0 = int(c['x']), int(c['y']), int(c['w']), int(c['h'])
            sub = lab[y0:y0 + h0, x0:x0 + w0]
            vals, counts = np.unique(sub[sub > 0], return_counts=True)
            if len(vals) == 0:
                continue
            cm = np.zeros(M.shape, bool)
            cm[y0:y0 + h0, x0:x0 + w0] = (sub == vals[np.argmax(counts)])
            wv = float(c['w_star'])
            col = (0, 220, 0) if wv > 0 else (140, 140, 140)
            if wv > 0:
                n_act += 1
            ov[cm] = col
            ys, xs = np.where(cm)
            cv2.putText(dec, f"w={wv:g}", (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        dec = cv2.addWeighted(ov, 0.45, dec, 0.55, 0)
        panels = [tag(fit(I), 'INPUT I'), tag(fit(P), 'BACKEND (rdnet)'), tag(fit(O), 'ORACLE (god mode)'),
                  tag(fit(T), 'GT T'), tag(fit(dec), f'DECISION: act {n_act}/{len(cs)} comps')]
        lab = np.zeros((H, 300, 3), np.uint8)
        lines = [f'{tag_} {key[:26]}', f'comps={len(cs)} acted={n_act}',
                 'green = act (w>0)', 'gray = keep input (w=0)']
        for i, t in enumerate(lines):
            cv2.putText(lab, t, (6, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.hstack([lab] + panels))
    gap = np.full((8, rows[0].shape[1], 3), 235, np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, gap, r])
    cv2.imwrite('fig/oracle_decisions_v3_0920.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('写出 fig/oracle_decisions_v3_0920.jpg', out.shape)


if __name__ == '__main__':
    main()
