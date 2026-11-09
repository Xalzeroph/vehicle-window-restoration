"""合成遮挡区（occlusion synthesis）—— 独立模块，暂不接入训练。

现状：在线合成只有"加法反射 + 失焦 + 曝光"，**没有遮挡区**（Ω 全 0），
      所以 S2 的 Ω 损失只在 30% 的真实样本上生效，且真实样本的 Ω 面积不可控。

本模块把真实玻璃照片里的三件事一起合成出来：
      I = (1 − η·c)·T + (1 − c)·β·R + c·R_occ + 曝光
其中 c 是随机遮挡盘（面积比可控），η 是照度下降（真实统计 10.9%±7.3%），
      R_occ 是遮挡物自身的淡反射（可用真实 R 的低频部分，或很小的常数纹理）。

产出：(I, T, R, omega) 四件与真实配对同构，可直接喂给现有 Dataset。

用法
----
    from synth_occlusion import make_occluder, apply_occlusion
    c = make_occluder(rng, H, W, area_frac=0.12)
    I, T, R, om = apply_occlusion(T, R, c, rng, eta=0.109, eta_sd=0.073)

自检
----
    python synth_occlusion.py --selftest
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

# 由 82 对确认池实测（像素域）：Ω 内额外压暗 10.9% ± 7.3%
ETA_MEAN, ETA_SD = 0.109, 0.073


def make_occluder(rng: np.random.Generator, H: int, W: int,
                  area_frac: float = 0.12, soft: int = 0) -> np.ndarray:
    """生成一个圆形遮挡盘（面积比 ≈ area_frac），返回 [0,1] 的软掩码 HxW。

    圆盘而不是矩形：真实遮挡物（拍摄者/手持物）的投影接近椭圆盘，
    且圆盘没有"沿窗框的细条"，避免把边界像素权重放大（那是我们已判定测不出的路径）。
    """
    area_frac = float(np.clip(area_frac, 0.02, 0.35))
    r = max(4, int(np.sqrt(area_frac * H * W / np.pi)))
    cy = int(rng.integers(r, max(r + 1, H - r)))
    cx = int(rng.integers(r, max(r + 1, W - r)))
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if soft > 0:                       # 软边：模拟有限的遮挡物-玻璃距离
        c = np.clip((r + soft - d) / (2.0 * soft), 0.0, 1.0).astype(np.float32)
    else:
        c = (d <= r).astype(np.float32)
    return c


def apply_occlusion(T: np.ndarray, R: np.ndarray, c: np.ndarray,
                    rng: np.random.Generator, eta: float | None = None,
                    eta_sd: float = ETA_SD, beta: float = 1.0,
                    occluder_refl: float = 0.02) -> tuple:
    """把遮挡效果合成进 (T, R)。

    T: HxWx3 [0,1] 透射层；R: HxWx3 有符号反射层（可正可负）
    c: HxW [0,1] 遮挡掩码
    eta: 照度下降幅度；None 时从 N(ETA_MEAN, ETA_SD) 截断采样
    occluder_refl: 遮挡物自身淡反射的幅度（高频随机纹理）
    返回 (I, T_target, R_eff, omega) —— omega 为 HxW 的 0/1 掩码，与真实池同构。
    """
    if eta is None:
        eta = float(np.clip(rng.normal(ETA_MEAN, eta_sd), 0.0, 0.35))
    c3 = c[..., None]
    # 遮挡物自身淡反射：低频噪声纹理（真实遮挡物是粗糙的皮肤/布料）
    hf = rng.normal(0.0, 1.0, T.shape[:2]).astype(np.float32)
    hf = cv2.GaussianBlur(hf, (0, 0), 6.0)[..., None] * c3 * occluder_refl

    R_eff = (1.0 - c3) * beta * R                                  # 反射被遮挡削弱
    T_dark = (1.0 - eta * c3) * T                                  # 照度同时下降
    I = T_dark + R_eff + hf                                        # 观测
    omega = (c > 0.5).astype(np.float32)
    return (np.clip(I, 0, 1).astype(np.float32),
            T_dark.astype(np.float32),                             # 目标仍是"含遮挡阴影的透射层"
            R_eff.astype(np.float32),
            omega)


def _selftest() -> int:
    rng = np.random.default_rng(0)
    H = W = 224
    T = rng.random((H, W, 3)).astype(np.float32)
    R = (rng.random((H, W, 3)).astype(np.float32) - 0.5)
    checks = []
    for frac in (0.05, 0.12, 0.25):
        c = make_occluder(rng, H, W, frac)
        got = float(c.mean())
        checks.append((f"遮挡面积 {frac:.2f} → {got:.3f}",
                       abs(got - frac) < 0.08, got))
    c = make_occluder(rng, H, W, 0.12)
    I, Td, Re, om = apply_occlusion(T, R, c, rng, eta=0.109)
    checks.append(("omega 与掩码一致", float(np.abs(om - (c > 0.5)).max()) == 0.0, 1.0))
    inside = c > 0.5
    d_in = float(Td[inside].mean() - T[inside].mean())
    d_out = float(Td[~inside].mean() - T[~inside].mean())
    checks.append((f"Ω 内被压暗 {d_in:+.3f}，Ω 外 {d_out:+.3f}", d_in < -0.02 and abs(d_out) < 1e-6, d_in))
    r_in = float(np.abs(Re[inside]).mean()); r_out = float(np.abs(Re[~inside]).mean())
    checks.append((f"Ω 内反射降到 {r_in / max(r_out, 1e-6):.2f}×", r_in < 0.2 * r_out, r_in))
    checks.append(("I 落在 [0,1]", 0.0 <= I.min() and I.max() <= 1.0, float(I.max())))
    ok = True
    for name, passed, val in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<34} value={val:+.4f}")
        ok &= passed
    print("自检:", "全部通过" if ok else "**有失败**")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    print(__doc__)
