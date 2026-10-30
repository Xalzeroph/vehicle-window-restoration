"""合成遮挡演示 v2：3 种形状（S1 / S2 全身 / S2 上半身）× 3 档"暗度"，供人工目检。

用法：python demo_synth_occlusion_v2.py --out demo.png
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_occlusion_v2 as S  # noqa: E402

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
AMP_LEVELS = [(0.10, 0.22), (0.02, 0.12), (0.00, 0.05)]   # 亮 / 中（新默认）/ 很暗


def real_pair(size):
    cand = sorted(glob.glob(os.path.join(P, "data/train_sets/real_pairs_1026/I/*.png")))
    if not cand:
        raise SystemExit("找不到真实配对")
    key = os.path.basename(cand[0])
    base = os.path.dirname(os.path.dirname(cand[0]))
    Ti = cv2.imread(os.path.join(base, "T", key))
    Ii = cv2.imread(os.path.join(base, "I", key))
    R = np.clip(Ii.astype(np.float32) - Ti.astype(np.float32), 0, 255)
    return cv2.resize(Ti, (size, size)), cv2.resize(R, (size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_synth_v2.png")
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    T, R = real_pair(a.size)
    lib = S.load_lib()
    rng = np.random.default_rng(a.seed)

    rows, labels = [], []
    # 固定三个形状（同一 seed 抽到同一形状族），只改暗度
    shapes = {}
    shapes["S1-宽"] = S.coverage(np.random.default_rng(101), "s1", a.size, a.size, 0.14)
    m, up = S.sample_S2(np.random.default_rng(202), lib, a.size, a.size, 0.14, upper_p=0.0)
    shapes["S2-全身"] = S._soft_edge(S._place(rng, m, a.size, a.size), np.random.default_rng(203))
    m2, up2 = S.sample_S2(np.random.default_rng(303), lib, a.size, a.size, 0.14, upper_p=1.0)
    shapes["S2-上半身"] = S._soft_edge(S._place(rng, m2, a.size, a.size, 0.95), np.random.default_rng(204))

    for name, c in shapes.items():
        cells = []
        for amp in AMP_LEVELS:
            S.R_OCC_AMP = amp
            I, Tt, _, om = S.apply_occlusion(T, R, c, np.random.default_rng(5), b=1.0, eta=0.109)
            st = S._stats(Tt, I, om)
            cells.append(I)
            last = st
        ov = cells[1].copy()
        cnts, _ = cv2.findContours((c > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
        rows.append(np.hstack([T.astype(np.uint8)] + cells + [ov]))
        labels.append(f"{name}  Om={last['area']*100:.1f}%  I/T in={last['ratio_in']:.3f} out={last['ratio_out']:.3f}")
        print("  " + labels[-1])

    canvas = np.vstack(rows)
    step = rows[0].shape[0]
    for i, lab in enumerate(labels):
        y = i * step + 20
        cv2.putText(canvas, lab, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, lab, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    hdr = np.zeros((26, canvas.shape[1], 3), np.uint8)
    cv2.putText(hdr, "T' | occ AMP=0.10-0.22 (old) | AMP=0.02-0.12 (new default) | AMP=0.00-0.05 (darkest) | Omega",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA)
    canvas = np.vstack([hdr, canvas])
    cv2.imwrite(a.out, canvas)
    print("写出", a.out, canvas.shape)


if __name__ == "__main__":
    main()
