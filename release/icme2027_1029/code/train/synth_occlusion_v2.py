"""合成遮挡区（Ω）注入：S1 随机平滑闭曲线 + S2 真实行人剪影（变换后作为投影轮廓）。

物理（与实测口径对齐，见 2026.10.26_物理模型_最终版 与 2026.10.28_遮蔽区定义档）：
    c(x)  遮挡覆盖率场（0..1，边界软过渡）
    T'    = (1 - eta*c) * T                <- 照度下降（目标本身更暗，实测 10.9%±7.3%）
    R'    = (1 - c) * R + c * R_occ        <- 亮反射被"暗的、离焦的自身反射"替换
    I     = T' + b * R'
    om    = c > 0.5                        <- 精确掩码（由构造给出，无需标注）

用法：
    python synth_occlusion_v2.py --selftest
    python synth_occlusion_v2.py --demo --demo-out demo.png
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

LIB_DEFAULT = os.environ.get(
    "TINTED_SILHOUETTES",
    "/DataDisk/zmy_code/TintedGlass_Enhance/data/silhouettes/lib")

# ---- 实测锚点（2026-10-29 重新标定，见 measure_real_darkness_gap / check_illumination_vs_reflection）----
# 关键修正：真实数据里 Ω 内的"暗"主要不是目标变暗，而是**输入相对目标暗**：
#   1026 池 82 对：T_in/T_out = 0.963，I_in/T_out = 0.803 ⇒ I/T_in ≈ 0.87
#   ⇒ 网络必须在 Ω 内**局部提亮**约 13%，而这一点是任何全局增益都做不到的
I_OVER_T_IN = (0.87, 0.10)             # Ω 内 I/T 的目标分布（截断到 [0.55, 1.05]）
I_OVER_T_OUT = 1.045                   # Ω 外 I/T（反射贡献 +4.5%）
R_OCC_RATIO = (0.015, 0.030)           # Ω 内"遮挡物自身暗反射"占 T 的比例（方案 B）
R_OCC_SIGMA = (10.0, 24.0)             # 离焦半径
EDGE_PX = (0.6, 1.2)                   # 边界软过渡宽度（用户：真实边缘更清晰）
TARGET_DARKEN = 0.0                    # 目标是否也压暗（实测只有 3.7%，默认不加）

# ------------------------------------------------------------------ 形状
def _pad_to(img, H, W):
    out = np.zeros((H, W), np.float32)
    h, w = img.shape
    if h > H or w > W:
        s = min(H / h, W / w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape
    y, x = (H - h) // 2, (W - w) // 2
    out[y:y + h, x:x + w] = img
    return out


def _place(rng, m, H, W, bottom_bias=0.7):
    """把形状放到画布上：多数贴下缘/侧缘（拍摄者通常站在取景框边缘），少数任意位置。"""
    h, w = m.shape
    if h > H or w > W:                      # 超界先等比缩进画布
        k = min(H / h, W / w) * 0.95
        m = cv2.resize(m, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        h, w = m.shape
    canvas = np.zeros((H, W), np.float32)
    if rng.random() < bottom_bias:
        y = H - h - rng.integers(0, max(1, H // 12))
        side = int(rng.integers(0, 3))
        if side == 0:
            x = rng.integers(0, max(1, W // 6))
        elif side == 1:
            x = W - w - rng.integers(0, max(1, W // 6))
        else:
            x = rng.integers(0, max(1, W - w + 1))
    else:
        y = rng.integers(0, max(1, H - h + 1))
        x = rng.integers(0, max(1, W - w + 1))
    y = int(min(max(0, int(y)), H - h))      # 硬夹紧，避免切片越界
    x = int(min(max(0, int(x)), W - w))
    canvas[y:y + h, x:x + w] = m
    return canvas


def _soft_edge(m, rng, edge_px=None):
    """把二值掩码变成带软过渡的覆盖率场 c∈[0,1]（内正外负的有符号距离）。"""
    e = float(rng.uniform(*EDGE_PX)) if edge_px is None else float(edge_px)
    b = (m > 0.5).astype(np.uint8)
    d_in = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)
    return np.clip(0.5 + (d_in - d_out) / (2.0 * e), 0.0, 1.0).astype(np.float32)


def _elastic(rng, m, amp_frac=0.05, k=3):
    """低频弹性形变：模仿投影面上的非线性拉伸/挤压。"""
    H, W = m.shape
    small = (6, 6)
    dx = rng.normal(0, 1, (small[0], small[1])).astype(np.float32)
    dy = rng.normal(0, 1, (small[0], small[1])).astype(np.float32)
    dx = cv2.resize(dx, (W, H), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy, (W, H), interpolation=cv2.INTER_CUBIC)
    a = amp_frac * min(H, W)
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(m, xx + a * dx, yy + a * dy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _affine_shape(rng, m, widen=(1.20, 1.70), squash=(0.80, 1.05), shear=0.10):
    """把剪影"放宽"成该视角下的投影轮廓。

    用户定案（2026-10-29）：**不做旋转** —— 旋转会破坏语义（人看起来是歪的）；
    改为各向异性放宽（x 拉宽 1.2–1.7×，y 轻微压缩 0.8–1.05×）+ 小幅剪切。
    """
    h, w = m.shape
    sx = float(rng.uniform(*widen))
    sy = float(rng.uniform(*squash))
    sh = float(rng.uniform(-shear, shear))
    l00, l01 = sx, sx * sh
    l10, l11 = 0.0, sy
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    tx = cx - (l00 * cx + l01 * cy)
    ty = cy - (l10 * cx + l11 * cy)
    M = np.array([[l00, l01, tx], [l10, l11, ty]], dtype=np.float64)
    return cv2.warpAffine(m.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def sample_S1(rng, H, W, area_frac):
    """S1：随机平滑闭曲线（极坐标傅里叶级数），可控面积。"""
    n = int(rng.integers(3, 7))
    th = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(0.55, 1.0, n)
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rn = np.interp(t, np.r_[th, 2 * np.pi], np.r_[r, r[0]])
    k = rng.integers(2, 5)
    rn = rn * (1 + 0.18 * np.sin(k * t + rng.uniform(0, 6.28)))
    R0 = np.sqrt(area_frac * H * W / np.pi)
    cx, cy = rng.uniform(0.25, 0.75) * W, rng.uniform(0.3, 0.8) * H
    xs = np.clip(cx + R0 * rn * np.cos(t), 0, W - 1)
    ys = np.clip(cy + R0 * rn * np.sin(t), 0, H - 1)
    m = np.zeros((H, W), np.float32)
    cv2.fillPoly(m, [np.stack([xs, ys], 1).astype(np.int32)], 1.0)
    return _bbox_crop(m)


def _bbox_crop(m):
    ys, xs = np.nonzero(m > 0.5)
    if len(ys) < 20:
        return m
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def sample_S2(rng, lib, H, W, area_frac, upper_p=0.5, widen=None):
    """S2：真实行人剪影 → 放宽（不旋转）→ 可选只保留上半身 → 按目标面积缩放并贴边。

    upper_p：以该概率只取"上半身"（顶部 35–80%），此时底边是画框切边（真实拍摄者常被画框切掉）。
    返回 (mask, anchored_bottom)。
    """
    m = lib[int(rng.integers(0, len(lib)))].astype(np.float32)
    if m.max() > 1.5:
        m = (m > 127).astype(np.float32)
    m = _bbox_crop(m)
    m = _affine_shape(rng, m) if widen is None else _affine_shape(rng, m, widen=widen)
    if rng.random() < 0.7:
        m = _elastic(rng, m, amp_frac=float(rng.uniform(0.015, 0.040)))
    m = _bbox_crop(m)

    upper = rng.random() < upper_p
    if upper:
        f = float(rng.uniform(0.35, 0.80))
        h0 = max(8, int(m.shape[0] * f))
        m = m[:h0]
        m = _bbox_crop(m)

    # 按目标面积缩放（先粗调，再按实际面积精调一次）
    for _ in range(2):
        a_now = max(float((m > 0.5).sum()), 1.0)
        k = np.sqrt(area_frac * H * W / a_now)
        m = cv2.resize(m, (max(1, int(m.shape[1] * k)), max(1, int(m.shape[0] * k))),
                       interpolation=cv2.INTER_NEAREST)
    m = (m > 0.5).astype(np.float32)
    return m, upper


def coverage(rng, mode, H, W, area_frac=(0.05, 0.25), lib=None):
    """返回覆盖率场 c(x) ∈[0,1]（同一形状，边界软过渡）。"""
    a = float(rng.uniform(*area_frac)) if np.ndim(area_frac) else float(area_frac)
    if mode == "s1":
        m = sample_S1(rng, H, W, a)
        m = _place(rng, m, H, W)
    elif mode == "s2":
        if not lib:
            raise ValueError("S2 需要剪影库")
        m, upper = sample_S2(rng, lib, H, W, a)
        m = _place(rng, m, H, W, bottom_bias=0.95 if upper else 0.7)
    else:
        raise ValueError(mode)
    return _soft_edge(m, rng)


# ------------------------------------------------------------------ 注入
def apply_occlusion(T, R, c, rng, b=None, i_over_t_in=I_OVER_T_IN,
                    r_occ_ratio=R_OCC_RATIO, target_darken=TARGET_DARKEN):
    """把遮挡注入到 (T, R) 上，返回 I, T_target, R_eff, om。

    标定口径（2026-10-29）：直接对齐**观测量** I/T。
      Ω 外：I/T ≈ 1.045（反射贡献 +4.5%）
      Ω 内：I/T ≈ 0.87 ± 0.10  ——即输入比目标暗约 13%，这是"目标没变、输入变暗"的情形，
            需要网络做**局部提亮**；任何全局增益都无法同时满足内外两侧。
    分解：I/T_in = (1 - eta_in) + ratio_occ，其中 ratio_occ ~ R_OCC_RATIO（方案 B 的暗反射）。
    """
    H, W = T.shape[:2]
    T = T.astype(np.float32)
    R = R.astype(np.float32)
    c3 = c[..., None]
    om = (c > 0.5)

    Tt = (1.0 - target_darken * c3) * T                   # 目标（默认不压暗）
    b = float(np.exp(rng.uniform(np.log(0.4), np.log(1.5)))) if b is None else float(b)

    # 先按"Ω 外 I/T = 1.045"定反射整体强度
    denom = float((R * (1 - c3)).sum())
    if denom > 1e-3:
        beta = (I_OVER_T_OUT - 1.0) * float((Tt * (1 - c3)).sum()) / denom
        beta = float(np.clip(beta, 0.0, 5.0))
    else:
        beta = 0.0

    # Ω 内：目标 I/T 由分布给定，拆成 (1-eta_in) + ratio_occ
    tgt = float(np.clip(rng.normal(*i_over_t_in), 0.55, 1.05)) if np.ndim(i_over_t_in) else float(i_over_t_in)
    ratio = float(rng.uniform(*r_occ_ratio)) if np.ndim(r_occ_ratio) else float(r_occ_ratio)
    eta_in = float(np.clip(1.0 - tgt + ratio, 0.0, 0.6))

    sig = float(rng.uniform(*R_OCC_SIGMA))
    R_occ = cv2.GaussianBlur(R, (0, 0), sig)
    m = float(R_occ[om].mean()) if om.any() else 0.0
    if m > 1e-3 and om.any():
        R_occ = R_occ * (ratio * float(Tt[om].mean()) / m)
    else:
        R_occ = np.zeros_like(R)

    I = (1.0 - eta_in * c3) * Tt + b * beta * (1.0 - c3) * R + c3 * R_occ
    I = np.clip(I, 0, 255)
    return I.astype(np.uint8), Tt.astype(np.uint8), (R * beta), om


def load_lib(lib_dir=None):
    d = lib_dir or LIB_DEFAULT
    ps = sorted(glob.glob(os.path.join(d, "*.png")))
    if not ps:
        raise FileNotFoundError(f"剪影库为空: {d}")
    lib = []
    for p in ps:
        m = cv2.imread(p, 0)
        if m is None:
            continue
        # 库内 PNG 存的是 0/255，统一归一到 0/1（面积计算依赖这一点）
        lib.append((m > 127).astype(np.float32))
    return lib


# ------------------------------------------------------------------ 自检 / 演示
def _stats(T, I, om):
    gT = T.mean(2).astype(np.float64)
    gI = I.mean(2).astype(np.float64)
    valid = gT > 20
    r = np.where(valid, gI / np.maximum(gT, 1e-6), np.nan)
    return dict(area=float(om.mean()), ratio_in=float(np.nanmean(r[om])),
                ratio_out=float(np.nanmean(r[~om])))


def selftest():
    rng = np.random.default_rng(0)
    ok = 0
    # 1) S1 面积可控
    for a in (0.05, 0.10, 0.20):
        c = coverage(rng, "s1", 224, 224, a)
        got = float((c > 0.5).mean())
        assert 0.55 * a < got < 1.6 * a, f"S1 面积失控 {a}->{got}"
        ok += 1
    # 2) 软边在 [0,1]
    c = coverage(rng, "s1", 128, 128, 0.1)
    assert c.min() >= 0 and c.max() <= 1 and (c > 0).any() and (c < 1).any()
    ok += 1
    # 3) 注入后：Ω 内 T 变暗、I/T 比值落在合理区间
    T = np.full((224, 224, 3), 150, np.float32)
    R = np.full((224, 224, 3), 40, np.float32)
    c = coverage(rng, "s1", 224, 224, 0.12)
    I, Tt, _, om = apply_occlusion(T, R, c, rng, b=1.0, eta=0.109)
    # 只看 Ω 内：应落在 1-eta 附近（全图均值被面积摊薄，不能用来判）
    d_in = float(Tt[om].mean() / T[om].mean())
    assert 0.83 < d_in < 0.95, f"Ω 内照度下降异常 {d_in}"
    ok += 1
    st = _stats(Tt, I, om)
    assert st["ratio_in"] < st["ratio_out"], f"Ω 内反射未减弱 {st}"
    ok += 1
    # 4) S2 形状来自真实剪影（需库）
    try:
        lib = load_lib()
        c = coverage(rng, "s2", 224, 224, 0.12, lib=lib)
        a = float((c > 0.5).mean())
        assert 0.04 < a < 0.30, f"S2 面积异常 {a}"
        ok += 1
        print(f"  [ok] 剪影库 {len(lib)} 个，S2 面积 {a*100:.1f}%")
    except FileNotFoundError as e:
        print(f"  [skip] S2: {e}")
    print(f"selftest: {ok} 项通过")


def demo(out_path, pair=None, n_per_mode=2, size=256):
    """在真实 (T, R) 上演示 S1 / S2 注入效果；输出 2 行 × 4 列。"""
    rng = np.random.default_rng(7)
    lib = load_lib()
    if pair is None:
        cand = glob.glob("/DataDisk/zmy_code/TintedGlass_Enhance/data/train_sets/real_pairs_1026/I/*.png")
        cand = sorted(cand)[:1] or sorted(glob.glob("data/train_sets/real_pairs_1026/I/*.png"))[:1]
        if not cand:
            raise SystemExit("找不到真实配对用于演示")
        key = os.path.basename(cand[0])
        base = os.path.dirname(os.path.dirname(cand[0]))
        Ti = cv2.imread(os.path.join(base, "T", key))
        Ii = cv2.imread(os.path.join(base, "I", key))
        R_est = np.clip(Ii.astype(np.float32) - Ti.astype(np.float32), 0, 255)
        T = Ti
        R = R_est
    else:
        T, R = pair
    T = cv2.resize(T, (size, size)).astype(np.float32)
    R = cv2.resize(R.astype(np.float32), (size, size))

    rows, labels = [], []
    for mode in ("s1", "s2"):
        for _ in range(n_per_mode):
            c = coverage(rng, mode, size, size, 0.12, lib=lib)
            I, Tt, Reff, om = apply_occlusion(T, R, c, rng)
            st = _stats(Tt, I, om)
            ov = I.copy()
            cnts, _ = cv2.findContours(om.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
            res = np.clip(np.abs(I.astype(np.float32) - Tt.astype(np.float32)) * 4, 0, 255).astype(np.uint8)
            row = np.hstack([Tt.astype(np.uint8), I, ov, cv2.applyColorMap(res, cv2.COLORMAP_INFERNO)])
            rows.append(row)
            labels.append(f"{mode.upper()}  Om={st['area']*100:.1f}%  I/T in={st['ratio_in']:.3f} out={st['ratio_out']:.3f}")
            print("  " + labels[-1])
    canvas = np.vstack(rows)
    for i, lab in enumerate(labels):
        y = i * rows[0].shape[0] + 22
        cv2.putText(canvas, lab, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, lab, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)
    print("demo ->", out_path, canvas.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--demo-out", default="synth_occlusion_demo.png")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.demo:
        demo(a.demo_out)
    else:
        ap.print_help()
