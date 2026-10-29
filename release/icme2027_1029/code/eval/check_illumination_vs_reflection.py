"""决定性检查：Ω 内的"变暗"到底来自**照度下降**还是**反射被替换**？

逻辑：
    r(x) = I/T
    若照度真的下降（T 本身变暗），则**参考图 T** 在 Ω 内就应比 Ω 外暗；
    若只是反射被换成更暗的反射，则 T 内外应无系统差异，而 r 在 Ω 内更低。

输出：T_in/T_out、I_in/I_out、r_in/r_out，以及三者的一致性检验。

用法：python check_illumination_vs_reflection.py --pool real_pairs_1026
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

    rows = []
    for k in sorted(meta)[: a.maxn]:
        pi, pt, po = (os.path.join(d, s, k + ".png") for s in ("I", "T", "omega"))
        if not all(os.path.exists(p) for p in (pi, pt, po)):
            continue
        I = cv2.imread(pi).astype(np.float64).mean(2)
        T = cv2.imread(pt).astype(np.float64).mean(2)
        Om = cv2.imread(po, 0)
        if I is None or T is None or Om is None or I.shape != T.shape:
            continue
        if Om.shape != I.shape[:2]:
            Om = cv2.resize(Om, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
        om = Om > 127
        if om.sum() < 500 or (~om).sum() < 500:
            continue
        T_in, T_out = float(T[om].mean()), float(T[~om].mean())
        I_in, I_out = float(I[om].mean()), float(I[~om].mean())
        if min(T_in, T_out, I_in, I_out) < 1:
            continue
        rows.append(dict(key=k, area=float(om.mean()),
                         T_in=T_in, T_out=T_out, I_in=I_in, I_out=I_out,
                         t_ratio=T_in / T_out, i_ratio=I_in / I_out,
                         r_ratio=(I_in / T_in) / (I_out / T_out)))

    if not rows:
        raise SystemExit("无样本")
    def m(k): return st.mean([x[k] for x in rows])
    def md(k): return st.median([x[k] for x in rows])
    print(f"池 {a.pool}  有效 {len(rows)} 对   Ω 面积中位 {md('area')*100:.1f}%")
    print()
    print(f"  T 亮度    Ω内 {m('T_in'):7.2f}   Ω外 {m('T_out'):7.2f}   ⇒ T_in/T_out = {m('t_ratio'):.3f}（中位 {md('t_ratio'):.3f}）")
    print(f"  I 亮度    Ω内 {m('I_in'):7.2f}   Ω外 {m('I_out'):7.2f}   ⇒ I_in/I_out = {m('i_ratio'):.3f}")
    print(f"  r=I/T    比值                                  ⇒ r_in/r_out = {m('r_ratio'):.3f}")
    print()
    tr = m("t_ratio")
    print("  判读：")
    if abs(tr - 1.0) < 0.03:
        print(f"    T_in/T_out = {tr:.3f} ≈ 1  ⇒ **参考图在 Ω 内没有系统性变暗**")
        print("    ⇒ 所谓『照度下降』不能由 T 支持；Ω 内的变暗主要来自『反射被换成更暗的反射』")
    elif tr < 0.97:
        print(f"    T_in/T_out = {tr:.3f} < 1  ⇒ 参考图本身在 Ω 内更暗 ⇒ **照度下降确实存在**")
        print(f"    幅度约 {(1-tr)*100:.1f}%（这是 T 自身的内外差，可与 r 的差分开）")
    else:
        print(f"    T_in/T_out = {tr:.3f} > 1  ⇒ 参考图在 Ω 内反而更亮（与照度下降相反）")
    print(f"    r_in/r_out = {m('r_ratio'):.3f} ⇒ 反射含量在 Ω 内低 {(1-m('r_ratio'))*100:.1f}%")
    print()
    print("  注意：T_in/T_out 若显著≠1，也可能来自『掩码偏向选择暗场景区域』（选择偏差），")
    print("        而非遮挡物挡光。二者在这里无法区分——需要人工标注的掩码才能定。")


if __name__ == "__main__":
    main()
