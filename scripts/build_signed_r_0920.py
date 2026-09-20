#!/usr/bin/env python3
"""S0：有符号 R 重提（+128 uint8 存储）——我方三套实拍 + RRW 暗反射子集。

- 我方三套：R = I − g·T（逐通道有符号，g=robust_gain）
- RRW camera1：逐帧 frame − g·GT → 时域中值（帧间对齐核验同时输出）
产物：
  data/pools/r_signed_0920/{real,rrw}/<name>.png   （uint8，R+128）
  data/pools/r_signed_0920/meta.csv
  fig/signed_r_check_0920.jpg                      （参考帧核验拼图：I | T | 有符号R热图 | Ω覆盖）
"""
import csv, glob, os, re, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import robust_gain  # noqa: E402

OUT = 'data/pools/r_signed_0920'
RRW = '/DataDisk/zmy_code/datasets/RRWDataset'
SETS = {'jun': ('data/test_sets/real_pair_0917', 'T.png'),
        'apr': ('data/test_sets/real_pair_apr_0917', 'GT.png'),
        'may': ('data/test_sets/real_pair_may_0917', 'GT.png')}
NEG_THR = 20.0   # Ω 定义：R < −20


def save_signed(R, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.clip(R + 128.0, 0, 255).astype(np.uint8))


def r_stats(R):
    Rn = R.mean(axis=2) if R.ndim == 3 else R
    neg = Rn < -8
    return dict(abs_mean=round(float(np.abs(Rn).mean()), 2),
                neg_frac=round(float(neg.mean()), 4),
                omega_frac=round(float((Rn < -NEG_THR).mean()), 4))


def main():
    rows = []
    # ---- 我方三套 ----
    for tag, (d, gtn) in SETS.items():
        for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
            sd = os.path.join(d, sc)
            T = cv2.imread(os.path.join(sd, gtn))
            for fn in (sorted(f for f in os.listdir(sd) if f.startswith('I')) or ['I.png']):
                I = cv2.imread(os.path.join(sd, fn))
                if I is None or T is None or I.shape != T.shape:
                    continue
                g = robust_gain(I, T)
                R = I.astype(np.float64) - T.astype(np.float64) * g.reshape(1, 1, 3)
                name = f'real__{tag}__{sc}' + ('' if fn == 'I.png' else f'__{fn[:-4]}')
                save_signed(R, f'{OUT}/real/{name}.png')
                rows.append(dict(name=name, src=tag, n_frames=1, **r_stats(R)))
    # ---- RRW camera1 暗反射子集（用 DeRef txt 的配对帧，勿用目录前 30 帧）----
    txt = open(os.path.join(RRW, 'DeRef_camera1.txt')).read().splitlines()
    by_scene = {}
    for l in txt:
        ps = [re.sub(r'/+', '/', x) for x in l.split()]
        if len(ps) < 2:
            continue
        by_scene.setdefault(ps[0].split('/')[-2], []).append((ps[0].lstrip('/'), ps[1].lstrip('/')))
    for sc, plist in sorted(by_scene.items()):
        if not sc.startswith('camera_objects'):
            continue
        T = cv2.imread(os.path.join(RRW, plist[0][1]))
        if T is None:
            print(f'{sc}: GT 缺失 -> 跳过'); continue
        Rs = []
        for a, _ in plist[:30]:
            I = cv2.imread(os.path.join(RRW, a))
            if I is None or I.shape != T.shape:
                continue
            g = robust_gain(I, T)
            Rs.append(I.astype(np.float64) - T.astype(np.float64) * g.reshape(1, 1, 3))
        if not Rs:
            continue
        R = np.median(np.stack(Rs), axis=0)
        name = f'rrw__{sc}'
        save_signed(R, f'{OUT}/rrw/{name}.png')
        rows.append(dict(name=name, src='rrw', n_frames=len(Rs), **r_stats(R)))
        print(f'{sc}: {len(Rs)} 帧(txt配对) | |R|均值 {r_stats(R)["abs_mean"]} | neg(<-8) {r_stats(R)["neg_frac"]*100:.1f}%')
    os.makedirs(OUT, exist_ok=True)
    with open(f'{OUT}/meta.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    n_neg = sum(1 for r in rows if r['neg_frac'] > 0.01)
    print(f'\n共 {len(rows)} 层（我方 {sum(1 for r in rows if r["src"]!="rrw")} + RRW {sum(1 for r in rows if r["src"]=="rrw")}）'
          f'，有负反射(>1%)的 {n_neg} | 写出 {OUT}/')

    # ---- 参考帧核验拼图 ----
    H, WP = 300, 400
    def fit(im):
        x = cv2.resize(im, (max(1, int(im.shape[1] * H / im.shape[0])), H), interpolation=cv2.INTER_AREA)
        c = np.zeros((H, WP, 3), np.uint8)
        if x.shape[1] > WP:
            x = x[:, :WP]
        c[:, :x.shape[1]] = x
        return c
    picks = [('jun', 'VID_20260609_170941', 'I.png'), ('apr', 'VID_20260426_173827', 'I_000009.png'),
             ('may', 'VID_20260512_162203', 'I_000000.png')]
    rows_img = []
    for tag, sc, fn in picks:
        d, gtn = SETS[tag]
        I = cv2.imread(f'{d}/{sc}/{fn}'); T = cv2.imread(f'{d}/{sc}/{gtn}')
        if I is None or T is None:
            continue
        g = robust_gain(I, T)
        R = (I.astype(np.float64) - T.astype(np.float64) * g.reshape(1, 1, 3)).mean(axis=2)
        heat = cv2.applyColorMap(np.clip((-R) / 60 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        om = (R < -NEG_THR)
        ovs = []
        for base in (I, T):
            ov = base.copy(); ov[om] = (0, 0, 255)
            ovs.append(cv2.addWeighted(ov, 0.45, base, 0.55, 0))
        lab = np.zeros((H, 250, 3), np.uint8)
        for i, t in enumerate([f'{tag}/{sc[:14]}', '列: I+Ω | T+Ω | -R热图', '核对: T的Ω处应无人影']):
            cv2.putText(lab, t, (6, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows_img.append(np.hstack([lab, fit(ovs[0]), fit(ovs[1]), fit(heat)]))
    if rows_img:
        gap = np.full((8, rows_img[0].shape[1], 3), 235, np.uint8)
        out = rows_img[0]
        for r in rows_img[1:]:
            out = np.vstack([out, gap, r])
        os.makedirs('fig', exist_ok=True)
        cv2.imwrite('fig/signed_r_check_0920.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print('写出 fig/signed_r_check_0920.jpg')


if __name__ == '__main__':
    main()
