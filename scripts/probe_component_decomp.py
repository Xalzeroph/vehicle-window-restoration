#!/usr/bin/env python3
"""车窗数据的"反射成分分解"诊断（2026-10-08）。

动机：用户描述画面里有三类东西
  ① 物体反射（带色彩 / 色彩弱）
  ② 高亮炫光（本数据没有）
  ③ 漫反射——整幅铺一层白（全局对比度/饱和损失）
  ④ 由此挡出来的"反射洞"（局部暗洞）

本脚本用**全局**统计量分别量化 ①③④，避免像上一版那样只在 ROI 里找全局效应。

指标：
  G1 全局对比损失   —— I 的局部 std 中位 / T 的局部 std 中位（漫反射会压低）
  G2 全局饱和度损失 —— I 的 HSV-S 均值 / T 的（贴膜+漫反射都会降饱和）
  G3 全局亮度偏移   —— I 与 T 的 luma 中位差（曝光+贴膜）
  G4 高光炫光检查   —— I 中"接近饱和(>=250)且成片"的像素占比（炫光签名）
  C1 色度结构       —— I 与 T 的 a*/b* 通道差的空间结构（物体反射带色 vs 灰纱）
  H1 洞的暗度       —— 用户 ROI 内 luma 相对全局中位的偏移（洞签名）
用法: python3 scripts/probe_component_decomp.py
"""
import csv, json, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
N = 5
K = 9
SETS = {
    'apr': (os.path.join(PROJ, 'data/test_sets/real_pair_apr_0917'), 'GT.png'),
    'may': (os.path.join(PROJ, 'data/test_sets/real_pair_may_0917'), 'GT.png'),
    'jun': (os.path.join(PROJ, 'data/test_sets/real_pair_0917'), 'T.png'),
}


def local_std(g):
    m = cv2.blur(g, (K, K))
    m2 = cv2.blur(g * g, (K, K))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def cellmask(shape, cells):
    H, W = shape
    m = np.zeros((H, W), bool)
    for c in cells:
        col, row = ord(c[0]) - 65, int(c[1]) - 1
        m[row * H // N:(row + 1) * H // N, col * W // N:(col + 1) * W // N] = True
    return m


def main():
    J = json.load(open(f'{PROJ}/data/annot_hole_cells.json', encoding='utf-8'))
    rows = []
    for k, v in J.items():
        dom, rest = k.split('/', 1)
        if dom not in SETS:
            continue
        root, gtn = SETS[dom]
        sc = rest.split('__', 1)[0]
        T = cv2.imread(os.path.join(root, sc, gtn))
        num = rest.split('__', 1)[1]
        fn = 'I.png' if dom == 'jun' else (num if num.startswith('I_') else 'I_' + num) + '.png'
        I = cv2.imread(os.path.join(root, sc, fn))
        if I is None or T is None or I.shape != T.shape:
            continue
        H, W = I.shape[:2]

        gI = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gT = cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # G1 全局对比
        cI, cT = float(np.median(local_std(gI))), float(np.median(local_std(gT)))
        # G2 全局饱和度
        sI = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
        sT = cv2.cvtColor(T, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
        # G3 亮度
        # G4 高光成片（炫光签名）：>=250 的像素占比 + 其最大连通域占比
        hot = (gI >= 250).astype(np.uint8)
        hot_frac = float(hot.mean())
        n, lab, st, _ = cv2.connectedComponentsWithStats(hot, 8)
        big = float(st[1:, 4].max() / hot.size) if n > 1 else 0.0
        # C1 色度结构：Lab 的 a/b 与 T 的差，其空间 std（有结构 vs 均匀灰纱）
        labI = cv2.cvtColor(I, cv2.COLOR_BGR2LAB).astype(np.float32)
        labT = cv2.cvtColor(T, cv2.COLOR_BGR2LAB).astype(np.float32)
        dab = labI[:, :, 1:] - labT[:, :, 1:]
        chroma_struct = float(np.median(np.abs(dab - np.median(dab))))
        # H1 洞的暗度（用户 ROI）
        cells = v.get('pos') or []
        if cells:
            Om = cellmask((H, W), cells)
            hole_off = float(gI[Om].mean() - np.median(gI))
        else:
            hole_off = float('nan')

        rows.append(dict(set=dom, case=rest,
                         contrast_I=round(cI, 2), contrast_T=round(cT, 2),
                         G1_contrast_ratio=round(cI / max(cT, 1e-9), 3),
                         sat_I=round(float(sI.mean()), 1), sat_T=round(float(sT.mean()), 1),
                         G2_sat_ratio=round(float(sI.mean()) / max(float(sT.mean()), 1e-9), 3),
                         luma_I=round(float(np.median(gI)), 1), luma_T=round(float(np.median(gT)), 1),
                         G3_luma_shift=round(float(np.median(gI) - np.median(gT)), 1),
                         G4_hot_frac=round(hot_frac, 4), G4_hot_bigcc=round(big, 4),
                         C1_chroma_struct=round(chroma_struct, 2),
                         H1_hole_offset=round(hole_off, 1)))

    out = os.path.join(PROJ, 'eval/probe_component_decomp.csv')
    keys = sorted({k for r in rows for k in r})
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f'写出 eval/probe_component_decomp.csv   n={len(rows)}\n')

    for dom in SETS:
        sub = [r for r in rows if r['set'] == dom]
        if not sub:
            continue
        m = lambda k: float(np.median([r[k] for r in sub]))
        print(f'=== {dom}  n={len(sub)} ===')
        print(f'  G1 全局对比比 I/T        {m("G1_contrast_ratio"):.3f}   '
              f'(<1 = I 比 T 更平 → 漫反射/贴膜压对比)')
        print(f'  G2 全局饱和度比 I/T      {m("G2_sat_ratio"):.3f}   '
              f'(<1 = I 更灰)')
        print(f'  G3 亮度偏移 I−T          {m("G3_luma_shift"):+.1f}')
        print(f'  G4 高光像素占比          {m("G4_hot_frac")*100:.3f}%   最大连通域 {m("G4_hot_bigcc")*100:.3f}%'
              f'   (炫光应成片且占比高)')
        print(f'  C1 色度结构(ab 差的中位绝对偏差) {m("C1_chroma_struct"):.2f}')
        print(f'  H1 洞相对全局中位的偏移  {m("H1_hole_offset"):+.1f} 灰度')
        print()


if __name__ == '__main__':
    main()
