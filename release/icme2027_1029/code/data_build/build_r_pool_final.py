"""最终 R 池：判断"色偏是否系统性"，并据此建池。

判定口径（系统性 vs 内容色）：
    逐图算 LAB 色度 (a*, b*) 的均值；一个池的**系统性色偏**表现为
      |mean(a*)|、|mean(b*)| 相对标准误很大（t = mean / (sd/√n)），即整池朝同一方向偏。
    内容色（如绿草地反光）是逐图随机方向，t 接近 0。

池子：
    A  RRW frames（真实配对数据集，普通玻璃；跨场景抽帧、按内容去重）
    B  SIR² Wildscene r.jpg（实拍反射图）
    C  自采 4 月（贴膜玻璃，仅作对照，不入池）

输出：data/pools/r_final_1029/*.png（逐图归一化到统一亮度）+ meta.json（含色偏统计与来源）
用法：python build_r_pool_final.py [--max-per-scene 3]
"""
import argparse
import glob
import json
import os
import statistics as st

import cv2
import numpy as np

RRW = "/DataDisk/zmy_code/datasets/RRWDataset"
SIR2 = "/DataDisk/zmy_code/datasets/SIR2"
SELF = "/DataDisk/zmy_code/TintedGlass_Enhance/data/train_sets/train_pairs_omega_0920/R"
OUT_DEFAULT = "/DataDisk/zmy_code/TintedGlass_Enhance/data/pools/r_final_1029"


def tint_stats(paths, label, maxn=300):
    A, B = [], []
    for p in paths[:maxn]:
        im = cv2.imread(p)
        if im is None:
            continue
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
        A.append(lab[..., 1].mean() - 128.0)
        B.append(lab[..., 2].mean() - 128.0)
    if not A:
        print(f"  {label:<28} (无样本)")
        return None
    n = len(A)
    ma, mb = float(st.mean(A)), float(st.mean(B))
    sa, sb = (float(st.stdev(A)) if n > 1 else 0.0), (float(st.stdev(B)) if n > 1 else 0.0)
    ta = float(ma / (sa / n ** 0.5)) if sa > 1e-6 else 0.0
    tb = float(mb / (sb / n ** 0.5)) if sb > 1e-6 else 0.0
    print(f"  {label:<28} n={n:4d}  a*={ma:+6.2f}±{sa:5.2f} (t={ta:+5.1f})   "
          f"b*={mb:+6.2f}±{sb:5.2f} (t={tb:+5.1f})")
    return dict(label=label, n=n, a_mean=ma, a_sd=sa, a_t=ta, b_mean=mb, b_sd=sb, b_t=tb)


def dhash(im, n=8):
    g = cv2.cvtColor(cv2.resize(im, (32, 32)), cv2.COLOR_BGR2GRAY)
    d = cv2.dct(g.astype(np.float32))[:n, :n]
    return int("".join("1" if v > d.mean() else "0" for v in d.ravel()), 2)


def norm_bright(im, p95_target=0.35):
    """逐图归一化：把 95 分位亮度拉到 p95_target，避免"残差很暗"导致合成里没有反射。"""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p95 = float(np.percentile(g, 95))
    if p95 < 1e-3:
        return im
    k = p95_target / p95
    return np.clip(im.astype(np.float32) * k, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--max-per-scene", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = []

    # ---------- 0) 色偏系统性判定 ----------
    print("=== 色偏：系统性 or 内容色 ===")
    rrw_sample = sorted(glob.glob(os.path.join(RRW, "reflection_camera1", "camera_objects", "*.jpg")))[:200]
    rrw_sample += sorted(glob.glob(os.path.join(RRW, "reflection_camera2", "indoor_object", "*.jpg")))[:100]
    sir2_sample = sorted(glob.glob(os.path.join(SIR2, "Wildscene", "*", "r.jpg")))
    self_sample = sorted(glob.glob(os.path.join(SELF, "*.png")))
    stats = [tint_stats(p, "RRW frames") for p in (rrw_sample,)]
    stats.append(tint_stats(sir2_sample, "SIR2 Wildscene r.jpg"))
    stats.append(tint_stats(self_sample, "自采 4 月（对照）"))

    # ---------- 1) A: SIR2 blended - T（去重 + 逐图归一化）----------
    n_a = 0
    seen = set()
    for gname, d in [("sir", os.path.join(SIR2, "SIR")),
                     ("postcard", os.path.join(SIR2, "postcard/postcard199"))]:
        bd, td = os.path.join(d, "blended"), os.path.join(d, "transmission_layer")
        if not (os.path.isdir(bd) and os.path.isdir(td)):
            continue
        for f in sorted(set(os.listdir(bd)) & set(os.listdir(td))):
            B = cv2.imread(os.path.join(bd, f))
            T = cv2.imread(os.path.join(td, f))
            if B is None or T is None or B.shape != T.shape:
                continue
            R = np.clip(B.astype(np.float32) - T.astype(np.float32), 0, 255).astype(np.uint8)
            h = dhash(R)
            if h in seen:
                continue
            seen.add(h)
            name = f"sir2{gname}__{f}"
            cv2.imwrite(os.path.join(a.out, name), norm_bright(R))
            meta.append(dict(name=name, group=f"sir2_{gname}_residue", src=os.path.join(bd, f)))
            n_a += 1
    print(f"\n  A SIR2 blended-T: {n_a} 张（去重后）")

    # ---------- 2) B: SIR2 Wildscene ----------
    n_b = 0
    for p in sir2_sample:
        im = cv2.imread(p)
        if im is None:
            continue
        sc = os.path.basename(os.path.dirname(p))
        name = f"sir2wild__{sc}.png"
        cv2.imwrite(os.path.join(a.out, name), norm_bright(im))
        meta.append(dict(name=name, group="sir2_wildscene", src=p))
        n_b += 1
    print(f"  B SIR2 Wildscene: {n_b} 张")

    json.dump(dict(n=len(meta), pool="r_final_1029",
                   sources={"sir2_blended_minus_T": n_a, "sir2_wildscene": n_b},
                   excluded=["self-captured 2026-04 (tinted transmission; R itself neutral, kept out by protocol)",
                             "RRW frames: systematic colour cast a*=+6.7 (t=+69), b*=+10.8 (t=+25)"],
                   tint=stats, items=meta),
              open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  合计 {len(meta)} 张 → {a.out}")
    print(f"  （旧池 25 张；本池 {len(meta)} 张，提升 {len(meta)/25:.0f}×）")


if __name__ == "__main__":
    main()
