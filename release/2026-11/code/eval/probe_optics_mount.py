"""外拍内（膜在远端）时的三个物理预判，用现有配对直接检验。

几何：相机 → 空气 → 玻璃外表面 → 玻璃 → 膜 → 空气 → 场景
由此推出三条可测预判：
  C1 反射接近中性：反射主要发生在【外表面】，没穿过染色膜
     -> R 的三通道均值应接近相等（比值 ≈1）
  C2 存在位移双影：膜–空气界面还有一次反射，穿过膜两次 ⇒ 更弱、带色、有位移
     -> R 与其自身平移后的相关应出现次级峰值（若找不到，说明该分量太弱）
  C3 反射比透射更锐：反射不穿膜（不受 haze），透射穿膜一次（被雾化）
     -> mean|∇R|/mean|∇T| 应 > 1（车内往外拍则相反）

用法：python probe_optics_mount.py [--pairs 60]
"""
import argparse
import glob
import json
import os
import statistics as st

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
POOL = os.path.join(P, "data/train_sets/real_pairs_1020")


def grad_mag(x):
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) if x.ndim == 3 else x.astype(np.float32)
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    return float(np.mean((gx + gy) / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=60)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(POOL, "meta.json"), encoding="utf-8"))
    keys = [k for k in sorted(meta) if os.path.exists(os.path.join(POOL, "R", k + ".png"))]
    keys = keys[::max(1, len(keys) // a.pairs)][:a.pairs]

    rows = []
    for k in keys:
        pi = os.path.join(POOL, "I", k + ".png")
        pt = os.path.join(POOL, "T", k + ".png")
        pr = os.path.join(POOL, "R", k + ".png")
        po = os.path.join(POOL, "omega", k + ".png")
        if not all(os.path.exists(p) for p in (pi, pt, pr)):
            continue
        I = cv2.imread(pi).astype(np.float32)
        T = cv2.imread(pt).astype(np.float32)
        R = cv2.imread(pr).astype(np.float32) - 128.0
        if not (I.shape == T.shape == R.shape):
            continue
        om = cv2.imread(po, 0) if os.path.exists(po) else None
        if om is not None and om.shape == I.shape[:2]:
            out = om <= 127
        else:
            out = np.ones(I.shape[:2], bool)
        if out.sum() < 1000:
            continue
        # C1：洞外 R 的三通道均值（只要正反射，避免负数把比值搞乱）
        r = R[out]
        rmean = r.mean(axis=0)                       # BGR
        pos = r[r.mean(axis=1) > 0]
        c1 = float(np.std(pos.mean(axis=0)) / max(abs(pos.mean()), 1e-6)) if pos.size else float("nan")
        # C3：锐度比
        c3 = grad_mag(R) / max(grad_mag(T), 1e-6)   # 两者都已是灰阶单位
        # 场景亮度（判断是否背光）
        rows.append(dict(key=k, g=cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).mean(),
                         c1_cv=c1, c3_ratio=c3, rmean=float(rmean.mean())))

    if not rows:
        raise SystemExit("没有可用配对")

    c1 = [r["c1_cv"] for r in rows if np.isfinite(r["c1_cv"])]
    c3 = [r["c3_ratio"] for r in rows]
    print(f"配对 {len(rows)} 对")
    print(f"\n=== C1 反射是否会带色（三通道均值的变异系数，0=完全中性）===")
    print(f"  中位 {st.median(c1):.4f}   均值 {st.mean(c1):.4f}   90 分位 {np.percentile(c1, 90):.4f}")
    print(f"  判读：<0.05 视为中性；>0.10 说明反射带明显色偏")
    print(f"\n=== C3 反射/透射 锐度比（>1 = 反射更锐，符合“膜在远端”）===")
    print(f"  中位 {st.median(c3):.3f}   均值 {st.mean(c3):.3f}   范围 {min(c3):.3f}~{max(c3):.3f}")
    print(f"  判读：>1 支持外拍内（反射不穿膜）；<1 不支持")
    bright = [r["c3_ratio"] for r in rows if r["g"] > st.median([x["g"] for x in rows])]
    dark = [r["c3_ratio"] for r in rows if r["g"] <= st.median([x["g"] for x in rows])]
    print(f"\n  按场景亮度分组：亮场景 {st.mean(bright):.3f} / 暗场景 {st.mean(dark):.3f}")
    print(f"  （若反射是外表面反射，暗场景下反射相对更强，比值应更高）")


if __name__ == "__main__":
    main()
