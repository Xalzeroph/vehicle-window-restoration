#!/usr/bin/env python3
"""洞形态合成器（M1 训练集扩充）：真 R × 前景（洞）mask → 带洞合成图 + 精确 mask 标签。

物理：人影挡住反射光 → 洞区没有反射（透射完整）。
公式：I' = clip(j·(T + b·R·(1−m)), 0, 1)；m = 前景 mask（1=洞=无反射）。
  洞形状来源：apr Ω 真形状（341 个）× 随机仿射（缩放 0.5–1.5 + 平移）；
  T 源 = VOC_s320/SIR2/nature/scenes；R 白名单 = apr(16) + RRW(9)；b~logU[0.4,1.5]；j~U[0.5,1]。
负样本：同流程 m≡0（无洞合成）。
产物: data/train_sets/train_mask_0921/{I,T,mask}/synth_XXXX.png（直接并入 M1 训练集目录）
用法: python3 scripts/build_holed_synth_0921.py [--n-hole 1500] [--n-neg 600] [--workers 16]
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT = os.path.join(PROJ, 'data/train_sets/train_mask_0921')
REAL = os.path.join(PROJ, 'data/pools/r_signed_0920/real')
RRW = os.path.join(PROJ, 'data/pools/r_signed_0920/rrw')
T_SRCS = [os.path.join(PROJ, 'data/external/VOC2012/VOC2012_s320'),
          '/home/zmy/mycode/datasets/SIR2/SIR/transmission_layer',
          '/home/zmy/mycode/datasets/nature/natural_T',
          '/home/zmy/vehicle_ssd/datasets/scenes']
SZ = 320


def one(job):
    idx, kind = job
    rng = np.random.default_rng(20260921 + idx)
    # T
    d = T_SRCS[rng.integers(len(T_SRCS))]
    f = sorted(os.listdir(d))[rng.integers(0, len(os.listdir(d)))]
    T = cv2.imread(os.path.join(d, f))
    if T is None:
        return None
    h, w = T.shape[:2]
    sc = SZ / min(h, w) * rng.uniform(1.0, 1.6)
    T = cv2.resize(T, (max(SZ, int(w * sc)), max(SZ, int(h * sc))), interpolation=cv2.INTER_AREA)
    h, w = T.shape[:2]
    y0, x0 = int(rng.integers(0, h - SZ + 1)), int(rng.integers(0, w - SZ + 1))
    T = T[y0:y0 + SZ, x0:x0 + SZ]
    # R（+128 解码）
    rf = (os.path.join(RRW, sorted(os.listdir(RRW))[rng.integers(len(os.listdir(RRW)))])
          if rng.random() < 0.36 else
          os.path.join(REAL, sorted(os.listdir(REAL))[rng.integers(len(os.listdir(REAL)))]))
    R = cv2.imread(rf).astype(np.float32) / 255.0 - 128.0 / 255.0
    h, w = R.shape[:2]
    sc = SZ / min(h, w) * rng.uniform(1.0, 1.3)
    R = cv2.resize(R, (max(SZ, int(w * sc)), max(SZ, int(h * sc))), interpolation=cv2.INTER_AREA)
    h, w = R.shape[:2]
    y0, x0 = int(rng.integers(0, h - SZ + 1)), int(rng.integers(0, w - SZ + 1))
    R = R[y0:y0 + SZ, x0:x0 + SZ]
    ks = int(rng.choice([5, 7, 9, 11], p=[0.1, 0.2, 0.3, 0.4]))
    R = cv2.GaussianBlur(R, (ks, ks), float(rng.uniform(2, 5)))
    # 洞 mask（真形状 + 随机仿射）
    m = np.zeros((SZ, SZ), np.float32)
    if kind == 'hole':
        mf = os.path.join(OUT, 'mask', sorted(os.listdir(os.path.join(OUT, 'mask')))[
            rng.integers(len(os.listdir(os.path.join(OUT, 'mask'))))])
        M0 = (cv2.imread(mf, 0) > 127).astype(np.uint8)
        ys0, xs0 = np.nonzero(M0)
        if len(ys0) > 0:
            M0 = M0[ys0.min():ys0.max() + 1, xs0.min():xs0.max() + 1]   # 裁到 bbox
            frac = float(rng.uniform(0.10, 0.30))                       # 洞目标面积占比（对齐 apr Ω 分布）
            sc2 = float(np.sqrt(frac * SZ * SZ / max(int(M0.sum()), 1)))
            M0 = cv2.resize(M0 * 255, (max(1, int(M0.shape[1] * sc2)), max(1, int(M0.shape[0] * sc2))),
                            interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
            th, tw = M0.shape
            py, px = int(rng.integers(-th // 2, SZ - th // 2 + 1)), int(rng.integers(-tw // 2, SZ - tw // 2 + 1))
            ys, ye = max(0, py), min(SZ, py + th); xs, xe = max(0, px), min(SZ, px + tw)
            if ye > ys and xe > xs:
                m[ys:ye, xs:xe] = M0[ys - py:ye - py, xs - px:xe - px]
    m3 = m[..., None]
    b = float(np.exp(rng.uniform(np.log(0.4), np.log(1.5))))
    j = float(rng.uniform(0.5, 1.0))
    Iv = T.astype(np.float32) + b * R * (1.0 - m3)
    hi = float(np.percentile(Iv, 99.5))
    if hi * j > 255:
        j = max(0.4, 254.0 / hi)
    I = np.clip(j * Iv, 0, 255).astype(np.uint8)
    name = f'synth_{idx:05d}'
    cv2.imwrite(os.path.join(OUT, 'I', name + '.png'), I)
    cv2.imwrite(os.path.join(OUT, 'T', name + '.png'),
                np.clip(j * T, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(OUT, 'mask', name + '.png'), (m * 255).astype(np.uint8))
    return dict(name=name, kind=kind, hole_frac=round(float(m.mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-hole', type=int, default=1500)
    ap.add_argument('--n-neg', type=int, default=600)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    jobs = [(i, 'hole') for i in range(args.n_hole)] + \
           [(args.n_hole + i, 'neg') for i in range(args.n_neg)]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=32)):
            if r:
                rows.append(r)
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(jobs)}', flush=True)
    # 并回 meta.json（Dataset 依赖它）
    meta_p = os.path.join(OUT, 'meta.json')
    with open(meta_p) as fp:
        meta = json.load(fp)
    for r in rows:
        meta[r['name']] = dict(scene='synth', frame=r['kind'],
                               omega_frac=r['hole_frac'], ok=int(r['hole_frac'] > 0.01))
    with open(meta_p, 'w') as fp:
        json.dump(meta, fp, indent=1)
    hf = [r['hole_frac'] for r in rows if r['kind'] == 'hole']
    print(f'完成 {len(rows)} 张（洞 {len(hf)}，洞占比中位 {np.median(hf):.3f}）-> 已并入 meta.json')


if __name__ == '__main__':
    main()
