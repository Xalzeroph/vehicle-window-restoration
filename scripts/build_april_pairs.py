#!/usr/bin/env python3
"""4 月（车窗户外，强反射档）实拍配对集：GT = 用户手工挑选的无人影帧（label.txt 中唯一帧号），
输入 = 各场景均匀抽取、且与 GT 对齐可信的帧（边缘域全搜索，峰值比 >= 4 才保留）。

产出 data/test_sets/real_pair_apr_0917/{scene}/{GT.png,I_*.png} + meta.json
口径与 6 月两套一致：对齐到 GT，稳健增益不写入（评测时现算）。
"""
import argparse, json, os, re, sys, glob
import cv2
import numpy as np

COARSE, SEARCH_PX, MIN_PEAK = 0.25, 16, 4.0


def edge_mag(g):
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def align(ref_gray, mov_gray):
    R = cv2.resize(edge_mag(ref_gray), None, fx=COARSE, fy=COARSE, interpolation=cv2.INTER_AREA)
    M = cv2.resize(edge_mag(mov_gray), None, fx=COARSE, fy=COARSE, interpolation=cv2.INTER_AREA)
    R = (R - R.mean()) / (R.std() + 1e-6)
    M = (M - M.mean()) / (M.std() + 1e-6)
    rad, pad = int(SEARCH_PX * COARSE), int(SEARCH_PX * COARSE) + 2
    rc = R[pad:-pad, pad:-pad]
    vals, best = [], (-9, 0, 0)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            sub = M[pad + dy:pad + dy + rc.shape[0], pad + dx:pad + dx + rc.shape[1]]
            if sub.shape != rc.shape:
                continue
            v = float((rc * sub).mean())
            vals.append(v)
            if v > best[0]:
                best = (v, dx, dy)
    v = np.array(vals)
    pr = float((best[0] - np.median(v)) / (v.std() + 1e-6))
    dx, dy = best[1], best[2]
    sy, sx = dy + rad, dx + rad
    cc = np.array([[v.reshape(2 * rad + 1, 2 * rad + 1)[sy, sx - 1], v.reshape(2 * rad + 1, 2 * rad + 1)[sy, sx + 1]],
                   [v.reshape(2 * rad + 1, 2 * rad + 1)[sy - 1, sx], v.reshape(2 * rad + 1, 2 * rad + 1)[sy + 1, sx]]]) if False else None
    return dx / COARSE, dy / COARSE, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/My_Reflection_Videos_cropped')
    ap.add_argument('--out', default='data/test_sets/real_pair_apr_0917')
    ap.add_argument('--per-scene', type=int, default=3)
    ap.add_argument('--scene-prefix', default='VID_20260426')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for sc in sorted(d for d in os.listdir(args.src) if d.startswith(args.scene_prefix)):
        d = os.path.join(args.src, sc)
        gt = re.search(r'^(\d{6})', open(os.path.join(d, 'label.txt')).read(), re.M).group(1)
        gtp = os.path.join(d, f'frame_{gt}.jpg')
        if not os.path.exists(gtp):
            print(f'{sc}: GT 帧缺 -> 跳过'); continue
        GT = cv2.imread(gtp)
        gGT = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
        h, w = GT.shape[:2]
        fs = sorted(glob.glob(os.path.join(d, 'frame_*.jpg')))
        idx = np.linspace(0, len(fs) - 1, args.per_scene * 3).astype(int)
        od = os.path.join(args.out, sc)
        os.makedirs(od, exist_ok=True)
        cv2.imwrite(os.path.join(od, 'GT.png'), GT)
        meta = dict(scene=sc, gt_frame=gt, size=[w, h], inputs={})
        kept = 0
        for i in idx:
            if kept >= args.per_scene:
                break
            p = fs[i]
            if p.endswith(f'frame_{gt}.jpg'):
                continue
            I = cv2.imread(p)
            if I.shape[:2] != (h, w):
                continue
            dx, dy, pr = align(gGT, cv2.cvtColor(I, cv2.COLOR_BGR2GRAY))
            if pr < MIN_PEAK:
                continue
            Iw = cv2.warpAffine(I, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fn = os.path.basename(p)[6:12]
            cv2.imwrite(os.path.join(od, f'I_{fn}.png'), Iw)
            D = np.abs(GT.astype(np.float32) - Iw.astype(np.float32)).max(axis=2)
            meta['inputs'][fn] = dict(frame=fn, shift=float(np.hypot(dx, dy)), peak_ratio=pr,
                                      D_med=float(np.median(D)), D_p95=float(np.percentile(D, 95)))
            kept += 1
        json.dump(meta, open(os.path.join(od, 'meta.json'), 'w'), indent=1)
        rows.append(meta)
        print(f"{sc} GT={gt} 保留输入 {kept}: " + ', '.join(
            f"{k}(峰值{v['peak_ratio']:.1f},D_p95={v['D_p95']:.0f})" for k, v in meta['inputs'].items()))
    print(f'\n=== {len(rows)} 场景，共 {sum(len(r["inputs"]) for r in rows)} 个输入 ===')
    with open(os.path.join(args.out, 'DO_NOT_USE_IN_TRAINING.md'), 'w') as fh:
        fh.write('# 评测专用（eval-only）。4 月车窗实拍，GT 为用户手工挑选的无人影帧。\n')


if __name__ == '__main__':
    sys.exit(main())
