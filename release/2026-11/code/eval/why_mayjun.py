"""为什么 may+jun 池能到 22.70 而 apr+jun 只有 20.97？

三个可测的诊断：
  A. 池的"物理难度"：反射强度 mean|I−T|、亮度关系 g、Ω 面积
  B. 池与测试面板的"内容距离"：池内 T 图与面板 GT 的最大内容相关（越高越像同一类场景）
  C. 逐场景增益：mayjun − aprjun 的 Δ 是均匀的还是集中在少数场景
    （均匀 ⇒ 全局色调/结构效应；集中 ⇒ 场景特异的匹配）
"""
import glob
import json
import os
import random
import statistics as st

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
PANEL = os.path.join(P, "data/test_sets/real_pair_1019")
POOL = os.path.join(P, "data/train_sets/real_pairs_1020")
PRED = os.path.join(P, "runs/pred")
random.seed(0)


def desc(im, s=24):
    if im is None:
        return None
    g = cv2.cvtColor(cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    g -= g.mean()
    n = np.linalg.norm(g)
    return (g / n).ravel() if n > 1e-6 else g.ravel()


def main():
    meta = json.load(open(os.path.join(POOL, "meta.json"), encoding="utf-8"))
    by = {}
    for k, v in meta.items():
        by.setdefault(v.get("domain"), []).append(k)

    panels = {}
    for d in sorted(glob.glob(os.path.join(PANEL, "*"))):
        g = os.path.join(d, "GT.png")
        if os.path.exists(g):
            panels[os.path.basename(d)] = desc(cv2.imread(g))
    Pmat = np.stack([v for v in panels.values() if v is not None])
    print(f"面板场景数: {len(Pmat)}")

    print("\n=== A. 池的物理难度 + B. 与面板的内容距离 ===")
    print(f"{'池':<6}{'n':>5}{'|I-T|均值':>10}{'g中位':>8}{'Ω中位':>8}{'与面板最大相关(均值/最高)':>26}")
    for dom in ("apr", "may", "jun"):
        keys = by.get(dom, [])
        if not keys:
            continue
        pick = random.sample(keys, min(40, len(keys)))
        diffs, gs, oms, sims = [], [], [], []
        for k in pick:
            idir = os.path.join(POOL, "I", k + ".png")
            if not os.path.exists(idir):
                cand = glob.glob(os.path.join(POOL, "I", k.replace("__", "*__") + "*"))
                idir = cand[0] if cand else None
            if not idir or not os.path.exists(idir):
                continue
            I = cv2.imread(idir)
            gt = glob.glob(os.path.join(POOL, "T", os.path.basename(idir)))
            if I is None or not gt:
                continue
            T = cv2.imread(gt[0])
            if T is None or T.shape != I.shape:
                continue
            diffs.append(float(np.abs(I.astype(np.float32) - T.astype(np.float32)).mean()))
            gs.append(float(T.mean()) / max(float(I.mean()), 1e-6))
            v = meta.get(k, {})
            if v.get("omega_frac") is not None:
                oms.append(float(v["omega_frac"]))
            dT = desc(T)
            if dT is not None:
                sims.append(float((Pmat @ dT).max()))
        if not diffs:
            print(f"{dom:<6} 读取失败")
            continue
        print(f"{dom:<6}{len(diffs):>5}{st.mean(diffs):>10.2f}"
              f"{(st.median(gs) if gs else float('nan')):>8.3f}"
              f"{(st.median(oms) * 100 if oms else float('nan')):>7.1f}%"
              f"{st.mean(sims):>16.4f} / {max(sims):.4f}")

    print("\n=== C. 逐场景 may+jun − apr+jun（三种子均值）===")
    def arm_mean(dirnames):
        out = {}
        for d in dirnames:
            for f in glob.glob(os.path.join(PRED, d, "*", "t_hat.png")):
                sc = os.path.basename(os.path.dirname(f)).split("__")[0]
                gt = os.path.join(PANEL, sc, "GT.png")
                if not os.path.exists(gt):
                    continue
                th = cv2.imread(f)
                G = cv2.imread(gt)
                if th.shape != G.shape:
                    th = cv2.resize(th, (G.shape[1], G.shape[0]))
                mse = float(np.mean((th.astype(np.float64) - G.astype(np.float64)) ** 2))
                out.setdefault(sc, []).append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
        return {k: st.mean(v) for k, v in out.items()}

    a = arm_mean(["real_pair_1019_aprjun_aprjun_s101", "real_pair_1019_aprjun_aprjun_s202",
                  "real_pair_1019_aprjun_aprjun_s303"])
    m = arm_mean(["real_pair_1019_mayjun15_s101", "real_pair_1019_mayjun15_s202",
                  "real_pair_1019_mayjun15_s303"])
    ks = sorted(set(a) & set(m))
    deltas = [(k, m[k] - a[k]) for k in ks]
    deltas.sort(key=lambda x: -x[1])
    print(f"{'场景':<26}{'apr+jun':>9}{'may+jun':>9}{'Δ':>8}")
    for k, d in deltas:
        print(f"{k:<26}{a[k]:>9.3f}{m[k]:>9.3f}{d:>+8.3f}")
    ds = [d for _, d in deltas]
    print(f"\nΔ 均值 {st.mean(ds):+.3f}  sd {st.stdev(ds):.3f}  最小 {min(ds):+.3f}  "
          f"最大 {max(ds):+.3f}  全为正: {all(d > 0 for d in ds)}")


if __name__ == "__main__":
    main()
