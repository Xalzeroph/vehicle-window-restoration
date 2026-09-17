#!/usr/bin/env python3
"""实拍配对测试集 v2（2026-09-17 用户口径修正版）：
  **6 月 GT = 无反射带补光帧**（明亮、无反射的"理想输出"），输入 = 其余三类各取一张。
  5 月无补光拍摄，其 GT 才是低光无反射（本脚本不处理 5 月）。

对齐：每个输入帧对齐到 GT 帧（边缘域归一化相关全搜索 + 抛物线亚像素，无迭代不发散）。
配准/光度口径沿用本日 v1（scripts/build_real_pair_testset.py）。

产出 data/test_sets/real_pair_litgt_0917/{scene}/{GT,I_*.png} + meta.json
"""
import argparse, json, os, re, sys
import cv2
import numpy as np

COARSE, FINE, SEARCH_PX = 0.25, 0.5, 16
CATS = [('refl_low', '低光反射'), ('refl_norm', '正常光带反射'), ('none_low', '低光无反射')]


def parse_labels(p):
    txt = open(p).read()
    ent = [(m.group(1), m.group(2).strip()) for m in re.finditer(r'^(\d{6})\s*(.*)$', txt, re.M)]
    refl = [e for e in ent if '反' in e[1] and '无' not in e[1] and '不' not in e[1]]
    out = {}
    out['refl_low'] = next((e for e in refl if '低光' in e[1]), refl[0] if refl else None)
    out['refl_norm'] = next((e for e in refl if '正常' in e[1]), None)
    out['none_low'] = next((e for e in ent if '无' in e[1] and '反' in e[1] and '补' not in e[1] and '加' not in e[1]), None)
    out['supp'] = next((e for e in ent if '无' in e[1] and ('补' in e[1] or '加' in e[1])), None)
    return out


def edge_mag(g):
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def align(ref_gray, mov_gray, scale, radius):
    r = cv2.resize(edge_mag(ref_gray), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = cv2.resize(edge_mag(mov_gray), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    r = (r - r.mean()) / (r.std() + 1e-6)
    m = (m - m.mean()) / (m.std() + 1e-6)
    rad = max(1, int(round(radius * scale)))
    pad = rad + 2
    rc = r[pad:-pad, pad:-pad]
    surf = np.full((2 * rad + 1, 2 * rad + 1), -np.inf)
    best, bxy = -np.inf, (0, 0)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            sub = m[pad + dy:pad + dy + rc.shape[0], pad + dx:pad + dx + rc.shape[1]]
            if sub.shape != rc.shape:
                continue
            v = float((rc * sub).mean())
            surf[dy + rad, dx + rad] = v
            if v > best:
                best, bxy = v, (dx, dy)
    valid = surf[np.isfinite(surf)]
    peak_ratio = float((best - np.median(valid)) / (valid.std() + 1e-6))
    dx, dy = bxy
    sy, sx = dy + rad, dx + rad
    if 0 < sx < surf.shape[1] - 1 and 0 < sy < surf.shape[0] - 1:
        denx = surf[sy, sx - 1] - 2 * surf[sy, sx] + surf[sy, sx + 1]
        deny = surf[sy - 1, sx] - 2 * surf[sy, sx] + surf[sy + 1, sx]
        cx = 0.5 * (surf[sy, sx - 1] - surf[sy, sx + 1]) / denx if abs(denx) > 1e-9 else 0.0
        cy = 0.5 * (surf[sy - 1, sx] - surf[sy + 1, sx]) / deny if abs(deny) > 1e-9 else 0.0
        dx, dy = dx + float(np.clip(cx, -0.5, 0.5)), dy + float(np.clip(cy, -0.5, 0.5))
    return dx / scale, dy / scale, peak_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/My_Reflection_Videos_cropped')
    ap.add_argument('--out', default='data/test_sets/real_pair_litgt_0917')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for sc in sorted(d for d in os.listdir(args.src) if d.startswith('VID_20260609_')):
        d = os.path.join(args.src, sc)
        ann = parse_labels(os.path.join(d, 'label.txt'))
        if not ann['supp']:
            print(f'{sc}: 无补光标注 -> 跳过'); continue
        nums = sorted(int(os.path.basename(p)[6:12]) for p in __import__('glob').glob(os.path.join(d, 'frame_*.jpg')))
        fp = lambda f: os.path.join(d, 'frame_%06d.jpg' % f)
        gt_num, note = int(ann['supp'][0]), ''
        if gt_num not in nums:
            after = [x for x in nums if x > gt_num]
            if after:
                gt_num, note = after[0], f'标注{ann["supp"][0]}缺→取后一帧{after[0]:06d}'
            else:
                gt_num, note = nums[-1], f'标注{ann["supp"][0]}超末帧→取末帧{nums[-1]:06d}'
        GT = cv2.imread(fp(gt_num))
        h, w = GT.shape[:2]
        gGT = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
        od = os.path.join(args.out, sc)
        os.makedirs(od, exist_ok=True)
        cv2.imwrite(os.path.join(od, 'GT.png'), GT)
        meta = dict(scene=sc, gt_frame=f'{gt_num:06d}', gt_note=note, size=[w, h], inputs={})
        for key, label in CATS:
            e = ann[key]
            if not e:
                continue
            f = int(e[0])
            if f not in nums:
                after = [x for x in nums if x > f]
                if not after:
                    continue
                f = after[0]
            I = cv2.imread(fp(f))
            if I.shape[:2] != (h, w):
                continue
            dx0, dy0, pr = align(gGT, cv2.cvtColor(I, cv2.COLOR_BGR2GRAY), COARSE, SEARCH_PX)
            M = np.float32([[1, 0, dx0], [0, 1, dy0]])
            I_w = cv2.warpAffine(I, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            dxf, dyf, prf = align(gGT, cv2.cvtColor(I_w, cv2.COLOR_BGR2GRAY), FINE, 3)
            M2 = np.float32([[1, 0, dx0 + dxf], [0, 1, dy0 + dyf]])
            I_w = cv2.warpAffine(I, M2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(os.path.join(od, f'I_{key}.png'), I_w)
            D = np.abs(GT.astype(np.float32) - I_w.astype(np.float32)).max(axis=2)
            gi = float((GT.astype(np.float64).ravel() * I_w.astype(np.float64).ravel()).sum() /
                       ((I_w.astype(np.float64).ravel() ** 2).sum() + 1e-9))
            meta['inputs'][key] = dict(label=label, frame=f'{f:06d}', shift=float(np.hypot(dx0 + dxf, dy0 + dyf)),
                                       peak_ratio=pr, gain_gt_over_input=round(gi, 3),
                                       D_p95=float(np.percentile(D, 95)), mean_gt=float(GT.mean()), mean_in=float(I.mean()))
            print(f"{sc} {key:<10} 帧{f:06d} 位移{np.hypot(dx0 + dxf, dy0 + dyf):5.2f}px 增益{gi:5.2f} "
                  f"D_p95={np.percentile(D, 95):5.1f} 亮度 {I.mean():5.1f}->{GT.mean():5.1f} {note}")
        json.dump(meta, open(os.path.join(od, 'meta.json'), 'w'), indent=1)
        rows.append(meta)
    print(f'\n=== {len(rows)} 场景 ===')
    with open(os.path.join(args.out, 'DO_NOT_USE_IN_TRAINING.md'), 'w') as fh:
        fh.write('# 本目录为评测专用（eval-only），禁止进入任何训练集/锚点/调参流程（2026-09-17 轮决议）\n\n'
                 '本集 GT = 6 月实拍「无反射带补光」帧，与 `vehicle_ssd/dataset/train/` 训练数据同源（同一批拍摄）。\n')


if __name__ == '__main__':
    sys.exit(main())
