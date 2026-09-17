#!/usr/bin/env python3
"""实拍配对测试集构建（2026-09-17 轮）: 同机位「有反射 I / 无反射 T」配对
→ 边缘域暴力搜索对齐(无迭代、不会发散) + 稳健光度增益 + 无反射段抖动底诊断。

判据设计（2026-09-17 两轮修订，见 docs/）：
  - ECC 在低纹理玻璃墙上会发散（实测 174546 → 101px），故改用边缘域全搜索；
  - 「补光无反射」帧是**故意打光**，其差异(54–107)比反射差异还大，不能当漂移基线；
  - 改用**无反射状态段内部**的帧间抖动作为噪声/漂移底 σ_map —— 与待检差异 I−gT 相互独立，非循环。

G-1''（逐场景）:  ① 对齐峰值显著 (peak_ratio>=3) 且位移<=15px
                  ② 反射信号显著: frac(D>25) >= 5%
                  ③ 非反射区达到抖底: median(D[~M]) <= max(8, 2*floor)
                  其中 M = D > max(3*floor, 8)
G-2: 通过场景数 >= 10/13
产出 data/test_sets/real_pair_0917/{scene}/{I,T}.png + meta.json + DO_NOT_USE_IN_TRAINING.md
"""
import argparse, json, os, re, sys
import cv2
import numpy as np

COARSE = 0.25
FINE = 0.5
SEARCH_PX = 16      # 粗搜半径（原图像素）
SEG_MAX = 6         # 抖动底用的无反射段帧数上限


def parse_labels(label_path):
    txt = open(label_path).read()
    ent = [(m.group(1), m.group(2).strip()) for m in re.finditer(r'^(\d{6})\s*(.*)$', txt, re.M)]
    refl = [e for e in ent if '反' in e[1] and '无' not in e[1] and '不' not in e[1]]
    low_refl = [e for e in refl if '低光' in e[1]] or refl
    no_refl = [e for e in ent if '无' in e[1] and '反' in e[1] and '补' not in e[1] and '加' not in e[1]]
    no_refl_supp = [e for e in ent if '无' in e[1] and ('补' in e[1] or '加' in e[1])]
    return ent, (low_refl[0] if low_refl else None), (no_refl[0] if no_refl else None), (no_refl_supp[0] if no_refl_supp else None)


def edge_mag(gray):
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return cv2.magnitude(sx, sy)


def align_bruteforce(ref_gray, mov_gray, scale, radius):
    """边缘域归一化相关全搜索（粗+细）。返回 (dx, dy, peak_ratio, surf_peak)。dx,dy 原图像素。"""
    r = cv2.resize(edge_mag(ref_gray), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = cv2.resize(edge_mag(mov_gray), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    r = (r - r.mean()) / (r.std() + 1e-6)
    m = (m - m.mean()) / (m.std() + 1e-6)
    rad = max(1, int(round(radius * scale)))
    # 只取中心区域做匹配，避免边界效应
    pad = rad + 2
    rc = r[pad:-pad, pad:-pad]
    best, best_xy = -np.inf, (0, 0)
    surf = np.full((2 * rad + 1, 2 * rad + 1), -np.inf)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            sub = m[pad + dy:pad + dy + rc.shape[0], pad + dx:pad + dx + rc.shape[1]]
            if sub.shape != rc.shape:
                continue
            v = float((rc * sub).mean())
            surf[dy + rad, dx + rad] = v
            if v > best:
                best, best_xy = v, (dx, dy)
    valid = surf[np.isfinite(surf)]
    peak_ratio = float((best - np.median(valid)) / (valid.std() + 1e-6))
    dx, dy = best_xy
    # 亚像素：峰值邻域抛物线拟合
    sy, sx = dx + rad, dy + rad
    if 0 < sx < surf.shape[1] - 1 and 0 < sy < surf.shape[0] - 1:
        denx = surf[sy, sx - 1] - 2 * surf[sy, sx] + surf[sy, sx + 1]
        deny = surf[sy - 1, sx] - 2 * surf[sy, sx] + surf[sy + 1, sx]
        cx = 0.5 * (surf[sy, sx - 1] - surf[sy, sx + 1]) / denx if abs(denx) > 1e-9 else 0.0
        cy = 0.5 * (surf[sy - 1, sx] - surf[sy + 1, sx]) / deny if abs(deny) > 1e-9 else 0.0
        dx, dy = dx + float(np.clip(cx, -0.5, 0.5)), dy + float(np.clip(cy, -0.5, 0.5))
    return dx / scale, dy / scale, peak_ratio, best


def robust_gain(I, T, keep_frac=0.7, iters=3):
    g = np.ones(3)
    for c in range(3):
        i = I[..., c].ravel().astype(np.float64)
        t = T[..., c].ravel().astype(np.float64)
        for _ in range(iters):
            den = float((t * t).sum())
            gc = float((i * t).sum() / den) if den > 1e-9 else 1.0
            res = np.abs(i - gc * t)
            keep = res <= np.percentile(res, keep_frac * 100)
            i, t = i[keep], t[keep]
            if len(i) < 100:
                break
            g[c] = gc
    return g


def warp(im, dx, dy, hw):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(im, M, hw, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/My_Reflection_Videos_cropped')
    ap.add_argument('--out', default='data/test_sets/real_pair_0917')
    ap.add_argument('--scene-prefix', default='VID_20260609_')
    args = ap.parse_args()

    scenes = sorted(d for d in os.listdir(args.src) if d.startswith(args.scene_prefix))
    print(f'场景数 {len(scenes)}')
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for sc in scenes:
        d = os.path.join(args.src, sc)
        ent, refl, no_refl, supp = parse_labels(os.path.join(d, 'label.txt'))
        if not (refl and no_refl):
            print(f'{sc}: 标注不全 -> 跳过')
            continue
        fp = lambda f: os.path.join(d, f'frame_{f}.jpg')
        if not (os.path.exists(fp(refl[0])) and os.path.exists(fp(no_refl[0]))):
            print(f'{sc}: 主帧缺失 -> 跳过')
            continue
        I, T = cv2.imread(fp(refl[0])), cv2.imread(fp(no_refl[0]))
        h, w = I.shape[:2]
        gI, gT = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY), cv2.cvtColor(T, cv2.COLOR_BGR2GRAY)

        # ---- 对齐：粗搜(0.25x, ±16px) → 细搜(0.5x, ±3px)
        dx0, dy0, pr0, _ = align_bruteforce(gI, gT, COARSE, SEARCH_PX)
        dxf, dyf, prf, _ = align_bruteforce(gI, warp(T, dx0, dy0, (w, h)), FINE, 3)
        dx, dy = dx0 + dxf, dy0 + dyf
        shift = float(np.hypot(dx, dy))
        T_w = warp(T, dx, dy, (w, h))

        # ---- 光度：稳健逐通道增益
        g = robust_gain(I, T_w)
        D = np.abs(I - T_w * g.reshape(1, 1, 3)).max(axis=2)

        # ---- 无反射段抖动底（与 D 独立）：段内多帧各自对齐后取逐像素 MAD
        t_num = int(no_refl[0])
        nxt = min([int(e[0]) for e in ent if int(e[0]) > t_num], default=10**9)
        seg_frames = []
        step = 1
        while len(seg_frames) < SEG_MAX and t_num + step * 7 < nxt:
            cand = t_num + step * 7
            if os.path.exists(fp(f'{cand:06d}')):
                seg_frames.append(cv2.imread(fp(f'{cand:06d}')))
            step += 1
        if len(seg_frames) >= 3:
            stack = []
            for f in seg_frames:
                ddx, ddy, _, _ = align_bruteforce(gT, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), COARSE, SEARCH_PX)
                fw = warp(f, ddx, ddy, (w, h))
                gf = robust_gain(T_w, fw)
                stack.append(np.abs(T_w - fw * gf.reshape(1, 1, 3)).max(axis=2))
            sigma_map = np.median(np.stack(stack), axis=0)
            floor = float(np.median(sigma_map))
        else:
            sigma_map, floor = None, None

        # ---- 判据
        thr = max(3 * floor, 8.0) if floor is not None else 25.0
        M = D > thr
        outside = ~M
        frac_M = float(M.mean())
        d_out = float(np.median(D[outside])) if outside.any() else float('nan')
        d_in = float(np.median(D[M])) if M.any() else float('nan')
        ok1 = (pr0 >= 3.0) and (shift <= 15.0)
        ok2 = float((D > 25).mean()) >= 0.05
        ok3 = (not np.isnan(d_out)) and d_out <= max(8.0, 2 * (floor or 0))
        passed = bool(ok1 and ok2 and ok3)

        met = dict(g=[float(x) for x in g], shift_px=shift, peak_ratio=float(pr0), fine_peak_ratio=float(prf),
                   n_seg=len(seg_frames), floor=floor,
                   D_med=float(np.median(D)), D_p95=float(np.percentile(D, 95)),
                   frac_D_gt25=float((D > 25).mean()), frac_M=frac_M,
                   D_med_outside=d_out, D_med_inside=d_in,
                   bg_frac=float((D < 8).mean()))
        od = os.path.join(args.out, sc)
        os.makedirs(od, exist_ok=True)
        cv2.imwrite(os.path.join(od, 'I.png'), I)
        cv2.imwrite(os.path.join(od, 'T.png'), T_w)
        meta = dict(scene=sc, frames=dict(I=refl, T=no_refl, N=supp), size=[w, h],
                    gates=dict(align=bool(ok1), signal=bool(ok2), bg=bool(ok3), passed=passed), **met)
        json.dump(meta, open(os.path.join(od, 'meta.json'), 'w'), indent=1)
        rows.append(meta)
        print(f"{sc} shift={shift:5.2f} peak={pr0:7.1f} floor={'n/a' if floor is None else format(floor, '.1f'):>5} "
              f"D_p95={met['D_p95']:5.1f} fracM={frac_M*100:4.1f}% D_out={d_out:6.2f} D_in={d_in:6.1f} "
              f"{'PASS' if passed else 'FAIL' + str([ok1, ok2, ok3])}")

    npass = sum(r['gates']['passed'] for r in rows)
    json.dump(dict(n_scenes=len(rows), n_pass=int(npass), gate='G-1\'\' + G-2(>=10/13)',
                   scenes={r['scene']: r['gates']['passed'] for r in rows}),
              open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
    print(f'\n=== 通过 {npass}/{len(rows)} ===')
    with open(os.path.join(args.out, 'DO_NOT_USE_IN_TRAINING.md'), 'w') as fh:
        fh.write('# 本目录为评测专用（eval-only），禁止进入任何训练集/锚点/调参流程（2026-09-17 轮决议）\n\n'
                 '理由：本集 T 帧与 `vehicle_ssd/dataset/train/` 的监督配对训练数据**字节级同源**'
                 '（md5 12/12 命中），且与合成测试集 v2/v2.1/v2.2 的 GT 内容同源（6 月实拍同一房间）。\n'
                 '一旦这些帧进入训练/微调/锚点标定/超参选择，本集的任何测试数字全部作废。\n')


if __name__ == '__main__':
    sys.exit(main())
