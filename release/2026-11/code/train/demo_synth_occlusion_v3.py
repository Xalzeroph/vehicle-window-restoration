"""合成遮挡演示 v3：看"边缘锐度 + 暗度"，含 Ω 局部 2× 放大格。

3 行 = 三档照度下降（η = 10.9% / 18% / 25%），均带方案 B 的暗反射替换（I/T_in ≈ 1.02）。
每行 5 格：T' | I | I 的 Ω 局部 2× 放大 | I 的 Ω 局部 4× 放大 | Ω 轮廓

用法：python demo_synth_occlusion_v3.py --out demo.png
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
IOT = [0.95, 0.87, 0.78]   # Ω 内 I/T 的目标值（暗度三档）
NAMES = ["I/T in=0.95 (mild)", "I/T in=0.87 (calibrated mean)", "I/T in=0.78 (strong)"]


def real_pair(size):
    cand = sorted(glob.glob(os.path.join(P, "data/train_sets/real_pairs_1026/I/*.png")))
    key = os.path.basename(cand[0])
    base = os.path.dirname(os.path.dirname(cand[0]))
    Ti = cv2.imread(os.path.join(base, "T", key)).astype(np.float32)
    Ii = cv2.imread(os.path.join(base, "I", key)).astype(np.float32)
    a = float(Ii.mean() / max(Ti.mean(), 1e-6))       # 逐图增益归一后再估反射
    R = np.clip(Ii - a * Ti, 0, 255)
    return cv2.resize(Ti, (size, size)), cv2.resize(R, (size, size))


def zoom_crop(img, om, k, pad=8):
    ys, xs = np.nonzero(om)
    if len(ys) == 0:
        return cv2.resize(img, (img.shape[1] * k, img.shape[0] * k))
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (crop.shape[1] * k, crop.shape[0] * k), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_synth_v3.png")
    ap.add_argument("--size", type=int, default=220)
    a = ap.parse_args()

    T, R = real_pair(a.size)
    lib = S.load_lib()

    # 固定一个"上半身"形状，三行共用，保证暗度对比公平
    m, up = S.sample_S2(np.random.default_rng(77), lib, a.size, a.size, 0.15, upper_p=1.0)
    c = S._soft_edge(S._place(np.random.default_rng(78), m, a.size, a.size, 0.95), np.random.default_rng(79))

    rows, labels = [], []
    for iot, name in zip(IOT, NAMES):
        I, Tt, Reff, om = S.apply_occlusion(T, R, c, np.random.default_rng(5), b=1.0, i_over_t_in=iot)
        st = S._stats(Tt, I, om)
        ov = I.copy()
        cnts, _ = cv2.findContours(om.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 1)
        z2 = zoom_crop(I, om, 2)
        z4 = zoom_crop(I, om, 4)
        H = Tt.shape[0]
        z2 = cv2.resize(z2, (H, H)) if z2.shape[0] != H else z2
        z4 = cv2.resize(z4, (H, H)) if z4.shape[0] != H else z4
        rows.append(np.hstack([Tt, I, z2, z4, ov]))
        labels.append(f"{name}  Om={st['area']*100:.1f}%  I/T in={st['ratio_in']:.3f}  out={st['ratio_out']:.3f}")
        print("  " + labels[-1])

    canvas = np.vstack(rows)
    step = rows[0].shape[0]
    for i, lab in enumerate(labels):
        y = i * step + 18
        cv2.putText(canvas, lab, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, lab, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    hdr = np.zeros((24, canvas.shape[1], 3), np.uint8)
    cv2.putText(hdr, "T' | I | Omega zoom x2 | Omega zoom x4 | Omega contour", (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 255), 1, cv2.LINE_AA)
    canvas = np.vstack([hdr, canvas])
    cv2.imwrite(a.out, canvas)
    print("写出", a.out, canvas.shape)


if __name__ == "__main__":
    main()
