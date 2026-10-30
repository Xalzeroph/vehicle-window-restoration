"""合成器标定核对：把合成样本的统计量与真实数据并排比对。

真实（2026-10-29 实测）：
  1026 池 82 对：T_in/T_out = 0.963，I_in/I_out = 0.803，I/T_in ≈ 0.87，I/T_out = 1.10
  1020 池 200 对：T_in/T_out = 0.935，I_in/I_out = 0.802，I/T_in ≈ 0.955/1.097≈0.87
合成应按 I/T 对齐：Ω 外 ≈1.045，Ω 内 ≈0.87。

用法：python verify_synth_calibration.py
"""
import os
import statistics as st
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_occlusion_v2 as S  # noqa: E402

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")


def main():
    rng = np.random.default_rng(0)
    lib = S.load_lib()
    H = W = 224
    rows = []
    for _ in range(120):
        base = float(rng.uniform(60, 200))
        T = np.full((H, W, 3), base, np.float32) * float(rng.uniform(0.9, 1.1))
        R = np.clip(rng.normal(base * 0.10, base * 0.05, (H, W, 3)), 0, 255).astype(np.float32)
        c = S.coverage(rng, "s2", H, W, 0.15, lib=lib)
        I, Tt, _, om = S.apply_occlusion(T, R, c, rng, b=1.0)
        gi = I.mean(2).astype(np.float64)
        gt = Tt.mean(2).astype(np.float64)
        valid = gt > 20
        if om.sum() < 300 or (~om).sum() < 300 or not valid.all():
            continue
        r = gi / np.maximum(gt, 1e-6)
        rows.append(dict(
            area=float(om.mean()),
            t_ratio=float(gt[om].mean() / gt[~om].mean()),
            i_ratio=float(gi[om].mean() / gi[~om].mean()),
            rt_in=float(r[om].mean()), rt_out=float(r[~om].mean()),
        ))
    if not rows:
        raise SystemExit("无样本")

    def m(k):
        return st.mean([x[k] for x in rows])

    def sd(k):
        return st.stdev([x[k] for x in rows]) if len(rows) > 1 else float("nan")

    print(f"合成样本 {len(rows)} 个（S2 剪影，目标面积 15%）")
    print(f"  Ω 面积         {m('area')*100:6.1f} %")
    print(f"  T_in/T_out     {m('t_ratio'):6.3f}   （真实 0.963 / 0.935）")
    print(f"  I_in/I_out     {m('i_ratio'):6.3f}   （真实 0.803 / 0.802）")
    print(f"  I/T in         {m('rt_in'):6.3f} ± {sd('rt_in'):.3f}   （真实 ≈0.87）")
    print(f"  I/T out        {m('rt_out'):6.3f} ± {sd('rt_out'):.3f}   （真实 ≈1.10）")
    print()
    ok = (abs(m("i_ratio") - 0.80) < 0.06) and (abs(m("rt_out") - 1.045) < 0.08)
    print("  标定结论：", "通过（与真实分布对齐）" if ok else "偏离，需要再调")


if __name__ == "__main__":
    main()
