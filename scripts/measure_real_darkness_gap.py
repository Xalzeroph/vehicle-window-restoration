"""量真实数据的"Ω 内外落差"分布，作为合成器的标定目标。

定义（与合成器一致、且是方法真正看到的量）：
    r(x) = I(x)/T(x)                      逐像素的相对反射含量
    R_tot = mean(r | Ω) / mean(r | ¬Ω)    Ω 内相对 Ω 外的比值（<1 表示 Ω 内更暗）
合成时按"每样本目标 R_tot ~ 真实分布"反解 η，而不是固定 η。

用法：python measure_real_darkness_gap.py [--pool real_pairs_1026]
"""
import argparse
import json
import os
import statistics as st

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="real_pairs_1026")
    ap.add_argument("--maxn", type=int, default=200)
    a = ap.parse_args()
    d = os.path.join(P, "data/train_sets", a.pool)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    keys = sorted(meta)[: a.maxn]

    rows = []
    for k in keys:
        pi = os.path.join(d, "I", k + ".png")
        pt = os.path.join(d, "T", k + ".png")
        po = os.path.join(d, "omega", k + ".png")
        if not all(os.path.exists(p) for p in (pi, pt, po)):
            continue
        I = cv2.imread(pi).astype(np.float32)
        T = cv2.imread(pt).astype(np.float32)
        Om = cv2.imread(po, 0)
        if I is None or T is None or Om is None or I.shape != T.shape:
            continue
        if Om.shape != I.shape[:2]:
            Om = cv2.resize(Om, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
        om = Om > 127
        if om.sum() < 500 or (~om).sum() < 500:
            continue
        gi, gt = I.mean(2), T.mean(2)
        valid = gt > 20
        r = np.where(valid, gi / np.maximum(gt, 1e-6), np.nan)
        r_in = float(np.nanmean(r[om]))
        r_out = float(np.nanmean(r[~om]))
        if r_out <= 0:
            continue
        rows.append(dict(key=k, area=float(om.mean()), r_in=r_in, r_out=r_out, R_tot=r_in / r_out))

    if not rows:
        raise SystemExit("无可用样本")
    gaps = [x["R_tot"] for x in rows]
    drops = [1 - g for g in gaps]
    print(f"池 {a.pool}  有效样本 {len(rows)} 对")
    print(f"  Ω 面积中位          {st.median([x['area'] for x in rows])*100:6.1f} %")
    print(f"  r_out 中位          {st.median([x['r_out'] for x in rows]):6.3f}")
    print(f"  r_in  中位          {st.median([x['r_in'] for x in rows]):6.3f}")
    print(f"  R_tot = r_in/r_out  中位 {st.median(gaps):.3f}  mean {st.mean(gaps):.3f}  sd {st.stdev(gaps):.3f}")
    q = np.percentile(gaps, [5, 25, 50, 75, 95])
    print(f"  分位 5/25/50/75/95  {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f} / {q[3]:.3f} / {q[4]:.3f}")
    print(f"  ⇒ 落差 1-R_tot     中位 {st.median(drops)*100:.1f} %  mean {st.mean(drops)*100:.1f} %  "
          f"sd {st.stdev(drops)*100:.1f} %")
    print("\n  标定建议（合成器按此采样）：")
    print(f"    R_tot ~ 截断正态(mean={st.mean(gaps):.3f}, sd={st.stdev(gaps):.3f}) 或直接用经验分位")
    out = os.path.join(P, "eval", f"real_darkness_gap_{a.pool}.json")
    json.dump(dict(pool=a.pool, n=len(rows), R_tot_mean=st.mean(gaps), R_tot_sd=st.stdev(gaps),
                   R_tot_median=st.median(gaps), pct=list(map(float, q)), rows=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 {out}")


if __name__ == "__main__":
    main()
