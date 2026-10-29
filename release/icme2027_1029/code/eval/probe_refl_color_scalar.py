"""§7-1 清洁口径：反射到底带不带色？（标量增益，避开逐通道 g 的吸收）

问题：池里的 R 由 build_train_real_pairs.py 用**逐通道** g 生成，色偏会被 g 吸收，
      所以之前测到的"反射色偏 0.044"是偏向中性的有偏估计。

本脚本用**标量** g 重做，并加两个对照：
  · 正对照：透射层 T 的色偏（贴膜是染色膜，T 必然带色）→ 方法必须能测出来
  · 独立旁证：RRW 的 R 白名单（不是从我们数据反推的）→ 看"真实反射层"本身的色偏水平

三个量都只用**强反射像素**（R 灰阶 > 80 分位）统计，避免弱信号把色偏稀释。

用法：python probe_refl_color_scalar_src.py
"""
import glob
import json
import os
import statistics as st

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
POOL = os.path.join(P, "data/train_sets/real_pairs_1020")
RRW_R = os.path.join(P, "data/pools/r_signed_0920/rrw")


def cv_channels(v):
    """三通道均值的变异系数（0 = 中性）。"""
    m = np.asarray(v, np.float64)
    return float(m.std() / max(abs(m.mean()), 1e-6))


def ratio_channels(v):
    m = np.asarray(v, np.float64)
    return float(m.max() / max(abs(m.min()), 1e-6))


def main():
    meta = json.load(open(os.path.join(POOL, "meta.json"), encoding="utf-8"))
    keys = [k for k in sorted(meta)
            if os.path.exists(os.path.join(POOL, "I", k + ".png"))
            and os.path.exists(os.path.join(POOL, "T", k + ".png"))]

    r_cv, r_ratio, t_cv, t_ratio, n_used = [], [], [], [], 0
    for k in keys:
        I = cv2.imread(os.path.join(POOL, "I", k + ".png")).astype(np.float64)
        T = cv2.imread(os.path.join(POOL, "T", k + ".png")).astype(np.float64)
        if I.shape != T.shape:
            continue
        # 标量增益（全通道联合最小二乘）：色偏不会被吸收
        g = float((I * T).sum() / max((T * T).sum(), 1e-9))
        R = I - g * T
        rg = R.mean(axis=2)
        hi = np.percentile(rg, 80)
        m = rg > max(hi, 1.0)
        if m.sum() < 500:
            continue
        rm = R[m].mean(axis=0)                     # 强反射像素的三通道均值（BGR）
        tm = T.reshape(-1, 3).mean(axis=0)         # 透射层整体三通道均值
        r_cv.append(cv_channels(rm)); r_ratio.append(ratio_channels(rm))
        t_cv.append(cv_channels(tm)); t_ratio.append(ratio_channels(tm))
        n_used += 1

    print(f"=== 标量增益口径（{n_used} 对，强反射像素 R>p80）===")
    print(f"  反射层色偏   CV 中位 {st.median(r_cv):.4f}   通道比 max/min 中位 {st.median(r_ratio):.4f}")
    print(f"  透射层色偏   CV 中位 {st.median(t_cv):.4f}   通道比 max/min 中位 {st.median(t_ratio):.4f}")
    print(f"  ⇒ 判读：若 反射 CV ≪ 透射 CV，说明反射确实近中性，而膜把色偏留在了透射层")

    files = sorted(glob.glob(os.path.join(RRW_R, "*.png")))[:60]
    if files:
        cvs, ratios = [], []
        for f in files:
            R = cv2.imread(f).astype(np.float64) - 128.0
            rg = R.mean(axis=2)
            m = rg > max(np.percentile(rg, 80), 1.0)
            if m.sum() < 500:
                continue
            rm = R[m].mean(axis=0)
            cvs.append(cv_channels(rm)); ratios.append(ratio_channels(rm))
        print(f"\n=== 独立旁证：RRW 的 R 白名单（{len(cvs)} 张，非本数据反推）===")
        print(f"  反射层色偏   CV 中位 {st.median(cvs):.4f}   通道比中位 {st.median(ratios):.4f}")
        print(f"  ⇒ 若与本数据反射的 CV 同量级，说明“反射近中性”是真实反射层的普遍性质")
    else:
        print("\n（RRW 白名单不存在，跳过旁证）")


if __name__ == "__main__":
    main()
