"""may+jun vs apr+jun：亮度来源核查 + 配对统计。

输出两件事：
  1. 每个臂的 raw PSNR 与「闭式逐图 affine 校正后」PSNR ⇒ 判断 2 dB 是否主要来自光度
  2. 三种子池化后的配对检验（vs RRW / 官方 / identity），n=15 场景

用法（服务器）：
  python scripts/tone_pool_compare.py
"""
import glob
import math
import os
import statistics as st

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
PANEL = os.path.join(P, "data/test_sets/real_pair_1019")
PRED = os.path.join(P, "runs/pred")

GROUPS = {
    "apr+jun": ["real_pair_1019_aprjun_aprjun_s101",
                "real_pair_1019_aprjun_aprjun_s202",
                "real_pair_1019_aprjun_aprjun_s303"],
    "may+jun": ["real_pair_1019_mayjun15_s101",
                "real_pair_1019_mayjun15_s202",
                "real_pair_1019_mayjun15_s303"],
}
BASE = {"identity": "real_pair_1019_identity",
        "rdnet_zs": "real_pair_1019_rdnet_zs",
        "RRW": "real_pair_1019_zhu"}


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(255.0 ** 2 / mse)


def fit_affine(x, y):
    """逐通道最小二乘 a,b（x=预测, y=目标）。"""
    A, B = [], []
    for c in range(3):
        xc = x[..., c].astype(np.float64).ravel()
        yc = y[..., c].astype(np.float64).ravel()
        mx, my = xc.mean(), yc.mean()
        vx = ((xc - mx) ** 2).mean()
        cov = ((xc - mx) * (yc - my)).mean()
        a = cov / max(vx, 1e-9)
        A.append(a)
        B.append(my - a * mx)
    return np.asarray(A), np.asarray(B)


def apply_affine(x, a, b):
    y = x.astype(np.float64) * a.reshape(1, 1, 3) + b.reshape(1, 1, 3)
    return np.clip(y, 0, 255)


def scenes():
    for d in sorted(glob.glob(os.path.join(PANEL, "*"))):
        if not os.path.isdir(d):
            continue
        g = os.path.join(d, "GT.png")
        ips = sorted(glob.glob(os.path.join(d, "I_*.png")))
        if os.path.exists(g) and ips:
            yield os.path.basename(d), ips[0], g


def arm_scores(dirname):
    """返回 {scene: (raw_psnr, corr_psnr)}"""
    out = {}
    for sc, ip, gp in scenes():
        key = os.path.join(PRED, dirname, os.path.basename(os.path.dirname(ip)) + "__" +
                           os.path.splitext(os.path.basename(ip))[0], "t_hat.png")
        if not os.path.exists(key):
            cand = glob.glob(os.path.join(PRED, dirname, sc + "__*", "t_hat.png"))
            key = cand[0] if cand else None
        if not key or not os.path.exists(key):
            continue
        th = cv2.imread(key)
        gt = cv2.imread(gp)
        if th is None or gt is None:
            continue
        if th.shape[:2] != gt.shape[:2]:
            th = cv2.resize(th, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        a, b = fit_affine(th, gt)
        out[sc] = (psnr(th, gt), psnr(apply_affine(th, a, b), gt))
    return out


def main():
    print("=== 1) 亮度来源核查（15 场景，逐图 affine 为事后上界）===")
    print(f"{'组 / 臂':<26}{'raw':>8}{'affine':>9}{'增益':>8}")
    store = {}
    for gname, arms in GROUPS.items():
        raws, corrs = [], []
        for a in arms:
            s = arm_scores(a)
            if not s:
                print(f"  {a}: 预测缺失")
                continue
            r = st.mean(v[0] for v in s.values())
            c = st.mean(v[1] for v in s.values())
            raws.append(r); corrs.append(c)
            store[a] = s
            print(f"  {a:<24}{r:>8.3f}{c:>9.3f}{c - r:>+8.3f}")
        if raws:
            print(f"  {'  → ' + gname + ' 均值':<24}{st.mean(raws):>8.3f}{st.mean(corrs):>9.3f}"
                  f"{st.mean(corrs) - st.mean(raws):>+8.3f}")
            print(f"     （种子 sd: raw {st.stdev(raws):.3f} / corr {st.stdev(corrs):.3f}）")
    for bn, bd in BASE.items():
        s = arm_scores(bd)
        if s:
            r = st.mean(v[0] for v in s.values())
            c = st.mean(v[1] for v in s.values())
            store[bd] = s
            print(f"  {bn:<24}{r:>8.3f}{c:>9.3f}{c - r:>+8.3f}")

    print("\n=== 2) 三种子池化配对检验（n=15 场景）===")
    for gname, arms in GROUPS.items():
        arms = [a for a in arms if a in store]
        if not arms:
            continue
        for bn, bd in BASE.items():
            if bd not in store:
                continue
            ks = sorted(set(store[arms[0]]) & set(store[bd]))
            diffs = []
            for k in ks:
                v = [store[a][k][0] for a in arms if k in store[a]]
                diffs.append(sum(v) / len(v) - store[bd][k][0])
            n = len(diffs); m = sum(diffs) / n
            sd = st.stdev(diffs); se = sd / math.sqrt(n)
            tc = 2.145 if n == 15 else 2.2
            wins = sum(1 for x in diffs if x > 0)
            print(f"  {gname:>8} − {bn:<9} n={n:>2}  Δ={m:+.3f}  sd={sd:.3f}  "
                  f"95%CI=[{m - tc * se:+.3f},{m + tc * se:+.3f}]  胜 {wins}/{n}")


if __name__ == "__main__":
    main()
