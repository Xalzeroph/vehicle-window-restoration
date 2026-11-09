#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N2 核心算子：训练域字典约束的有符号双字典两步近端输出校正（2026-10-31）。

主张（审查后收窄，不得越界）
---------------------------
本模块**不宣称恢复物理唯一的反射层**。它是"训练域字典约束的输入条件输出校正"：
  P = F0(I)（冻结基线输出，RGB [0,1]），I = 输入，输出 T̂ = N2(I, P)。
DT/DR 由**原训练池**的 T 与 I−T 学习，不使用任何测试 GT / 人工 Ω / 场景编号 / 逐图自由参数。
DR 表示 I−T 中的**一切**有符号干扰（照度变化、配准误差、真实反射），两本字典张成空间
可以重叠 —— 层混淆仍然存在，单位列范数只消除尺度自由度。任何"分层得到保证"的说法撤回。

锁定口径（预登记；运行时不得扫超参）
------------------------------------
  patch        5×5×RGB = 75 维，stride 1，reflect padding 2
  字典         每本 16 原子 = 1 个均值/DC 原子 + 15 个中心化截断 SVD 原子，单位列范数
               确定性：eigh 降序；符号固定为"绝对值最大的分量为正"；均值原子 = mean/‖mean‖
  DC 处理      DC **显式**由一个均值原子承载，不另做去均值；τ 对该原子同样生效
  目标         E(t,a,b) = ½‖t−p‖² + ½‖i−t−DRb‖² + ½‖t−DTa‖² + τ(‖a‖₁+‖b‖₁)
               ⇒ ∂E/∂t = 3t − p − i + DRb − DTa ⇒ t* = (p + i − DRb + DTa)/3
  ISTA         a ← soft(a + h·Dᵀ(t−Da), hτ)，b 同理
               h = 1/λ_max(D Dᵀ) = 1/‖D‖₂²（**取等号的精确谱界**，满足审查的 h≤1/‖D‖₂²）
               ⚠️ 与审查建议的 h=1/16 的差别：1/16 是 trace 界（trace(DDᵀ)=16），比谱界小
               约 4–16 倍，会把有效步长压到收敛所需步数的零头。h **不改变 ISTA 不动点**，
               只改变收敛速度；unit 动作里有"h=1/16 跑 200 步 vs h=谱界跑 K 步"的不动点一致性
               检查。若坚持 h=1/16，则必须把 K 提到 ~20+（CPU 探针预算不够），否则是在
               数值未收敛处判方案死刑。
               内迭代 K = 4（固定，不早停），外迭代 N_OUTER = 2
  t 松弛       t ← t + η_k(t* − t)；η_k 直接可训、初始化 0、每步投影到 [0,1]
  η=0          输出**逐像素精确等于 p**（uint8 存盘也相同）——关闭即基线
  τ            随像素标度线性缩放：τ = ρ·median‖patch‖₂（训练池 T patch，[0,1] 像素单位）
               ρ = 0.10 预登记；若改用 [0,255] 像素单位，τ 必须 ×255
  聚合         转置卷积重建 / 覆盖计数（逐像素实际覆盖的 patch 数，边界同口径归一）
  数值         算子全程 float32（ISTA 对精度敏感，AMP 下强制关闭 autocast）

已知非恒等退化（**不是 bug，是结构性代价，必须量化**）
------------------------------------------------------
  1) i=p 且编码被杀光（|系数|≤τ）：t* = (p+i)/3 = 2p/3 ⇒ 单步压暗 ≈33%。
     稀疏惩罚本身会把"没有反射的平坦/低对比块"往暗里推，这正是本项目 Ω 内误压的同型风险。
  2) 只有 t 项没有 i 项的对照（MODE_TONLY）目标为 ½‖t−p‖²+½‖t−DTa‖² ⇒ t*=(p+DTa)/2，
     分母是 2 而不是 3（审查要求）；不能"删掉 DR 后沿用分母 3"。
  3) 分析用 reflect padding、合成用零 padding 的转置卷积，两者不是严格共轭 ⇒ ≤2 px 边框环
     近似；由 dense 参考实现 + 边框量化单独核对（unit 动作）。

依赖：numpy + torch（CPU 可跑）。本文件不 import RDNet。
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ------------------------- 预登记常量（不得在运行时扫） -------------------------
PATCH = 5
CH = 3
DIM = PATCH * PATCH * CH          # 75
PAD = PATCH // 2                  # 2
N_ATOM = 16
K_INNER = 4
N_OUTER = 2
RHO_TAU = 0.10
H_TRACE_BOUND = 1.0 / N_ATOM      # 1/16，审查建议的保守统一值（仅作参考）
MODE_DUAL = "dual"
MODE_TONLY = "tonly"
N_PARAM_DICT = 2 * N_ATOM * DIM   # 2400
N_PARAM_TOTAL = N_PARAM_DICT + N_OUTER   # + 2 个 η = 2402


def ista_step(dt, dr):
    """h = 1/λ_max(D Dᵀ) = 1/‖D‖₂²：精确谱界（两本字典取较小者，统一一个 h）。"""
    g = np.asarray(dt, np.float32) @ np.asarray(dt, np.float32).T
    r = np.asarray(dr, np.float32) @ np.asarray(dr, np.float32).T
    lam = max(float(np.linalg.eigvalsh(g).max()), float(np.linalg.eigvalsh(r).max()))
    lam = max(lam, 1e-8)
    return 1.0 / lam, dict(lam_max=lam, h_trace_bound=H_TRACE_BOUND)


class _NoAutocast:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def autocast_off():
    """强制 float32：算子内部关闭 autocast（CPU 上退化为 no-op）。"""
    try:
        return torch.autocast(device_type="cuda", enabled=False)
    except Exception:                                    # noqa: BLE001
        return _NoAutocast()


def soft_thresh(x, thr):
    return torch.sign(x) * torch.relu(x.abs() - thr)


# ------------------------------- 稠密参考实现 -------------------------------
def forward_dense_np(i, p, dt, dr, tau, eta, k_inner=K_INNER, n_outer=N_OUTER,
                     h=None, mode=MODE_DUAL):
    """逐像素**稠密**参考实现（numpy，只用于单元测试，H,W 取 32 量级）。

    与卷积实现刻意走两条不同的代码路径：
      * patch 提取：np.lib.stride_tricks.sliding_window_view（不用 conv2d）
      * 重建聚合：按 25 个偏移逐次累加 + 覆盖计数（不用 conv_transpose2d）
    输入/输出 [CH,H,W] float32；eta 为长度 n_outer 的序列。
    返回 (t, a, b, diag)。
    """
    i = np.asarray(i, np.float32)
    p = np.asarray(p, np.float32)
    dt = np.asarray(dt, np.float32)
    dr = np.asarray(dr, np.float32)
    if h is None:
        h = ista_step(dt, dr)[0]
    _, H, W = p.shape

    def bank(x):
        xp = np.pad(x, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect")
        v = np.lib.stride_tricks.sliding_window_view(xp, (PATCH, PATCH), axis=(1, 2))
        return np.ascontiguousarray(v.transpose(1, 2, 0, 3, 4).reshape(H * W, DIM))

    gp = dt @ dt.T
    gr = dr @ dr.T
    # 覆盖计数：每个原图像素被多少个 patch 位置覆盖（边界少于 25）
    cov = np.zeros((H + 2 * PAD, W + 2 * PAD), np.float32)
    ones = np.ones((H, W), np.float32)
    for dy in range(PATCH):
        for dx in range(PATCH):
            cov[dy:dy + H, dx:dx + W] += ones
    cov3 = cov[PAD:PAD + H, PAD:PAD + W][None]

    def aggregate(coef, D):
        rec = (coef @ D).reshape(H, W, CH, PATCH, PATCH)
        acc = np.zeros((CH, H + 2 * PAD, W + 2 * PAD), np.float32)
        for dy in range(PATCH):
            for dx in range(PATCH):
                acc[:, dy:dy + H, dx:dx + W] += rec[:, :, :, dy, dx].transpose(2, 0, 1)
        return acc[:, PAD:PAD + H, PAD:PAD + W] / cov3

    t = p.copy()
    a = np.zeros((H * W, N_ATOM), np.float32)
    b = np.zeros((H * W, N_ATOM), np.float32)
    use_r = (mode == MODE_DUAL)
    resid = None
    for k in range(n_outer):
        for _ in range(k_inner):
            pt = bank(t)
            a_prev = a.copy()
            a = soft_thresh_np(a + h * (pt @ dt.T - a @ gp.T), h * tau)
            if use_r:
                # b 的残差 = i − t（见 N2Dict.forward 的说明；用 t 会让两本字典互相抵消）
                pr = bank(i - t)
                b = soft_thresh_np(b + h * (pr @ dr.T - b @ gr.T), h * tau)
            resid = float(np.linalg.norm(a - a_prev) / max(np.linalg.norm(a), 1e-8))
        rec_t = aggregate(a, dt)
        if use_r:
            rec_r = aggregate(b, dr)
            t_star = (p + i - rec_r + rec_t) / 3.0
        else:
            t_star = (p + rec_t) / 2.0
        t = t + float(eta[k]) * (t_star - t)
    return (t.astype(np.float32), a.reshape(H, W, N_ATOM), b.reshape(H, W, N_ATOM),
            dict(ista_a_rel_change=resid))


def soft_thresh_np(x, thr):
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


# ------------------------------- 算子本体 -------------------------------
class N2Dict(nn.Module):
    """有符号双字典两步近端校正（部署算子，卷积实现）。"""

    def __init__(self, dt, dr, tau, eta_init=0.0, k_inner=K_INNER, n_outer=N_OUTER,
                 h=None, mode=MODE_DUAL, grad_ckpt=True):
        super().__init__()
        dt = torch.as_tensor(np.asarray(dt, np.float32))
        dr = torch.as_tensor(np.asarray(dr, np.float32))
        if tuple(dt.shape) != (N_ATOM, DIM) or tuple(dr.shape) != (N_ATOM, DIM):
            raise ValueError(f"字典形状应为 {(N_ATOM, DIM)}，收到 {tuple(dt.shape)}/{tuple(dr.shape)}")
        self.dt = nn.Parameter(dt / dt.norm(dim=1, keepdim=True).clamp(min=1e-8))
        self.dr = nn.Parameter(dr / dr.norm(dim=1, keepdim=True).clamp(min=1e-8))
        self.register_buffer("tau", torch.tensor(float(tau), dtype=torch.float32))
        self.eta = nn.Parameter(torch.full((int(n_outer),), float(eta_init)))
        self.k_inner = int(k_inner)
        self.n_outer = int(n_outer)
        if h is None:
            h = ista_step(dt.numpy(), dr.numpy())[0]
        self.h = float(h)
        self.mode = mode
        self.grad_ckpt = bool(grad_ckpt)
        self._cov = {}

    def _coverage(self, H, W, ref):
        c = self._cov.get((H, W))
        if c is None or c.device != ref.device:
            ones = torch.ones(1, 1, H, W, device=ref.device, dtype=torch.float32)
            k = torch.ones(1, 1, PATCH, PATCH, device=ref.device, dtype=torch.float32)
            c = F.conv_transpose2d(ones, k, padding=PAD)
            self._cov[(H, W)] = c
        return c

    def kernels(self):
        return (self.dt.reshape(N_ATOM, CH, PATCH, PATCH),
                self.dr.reshape(N_ATOM, CH, PATCH, PATCH))

    def _ista_step(self, t, a, b, i, use_r):
        """一步 ISTA（a 用残差 t−DTa；b 用残差 i−t−DRb）。"""
        kt, kr = self.kernels()
        gt = self.dt @ self.dt.t()
        gr = self.dr @ self.dr.t()
        tp = F.pad(t, (PAD, PAD, PAD, PAD), mode="reflect")
        at = F.conv2d(tp, kt)
        a2 = soft_thresh(a + self.h * (at - torch.einsum("ij,bjhw->bihw", gt, a)),
                         self.h * self.tau)
        b2 = b
        if use_r:
            ip = F.pad(i - t, (PAD, PAD, PAD, PAD), mode="reflect")
            ar = F.conv2d(ip, kr)
            b2 = soft_thresh(b + self.h * (ar - torch.einsum("ij,bjhw->bihw", gr, b)),
                             self.h * self.tau)
        return a2, b2

    def forward(self, i, p, eta=None, mode=None, return_diag=False):
        """i, p: [B,3,H,W]；返回 T̂（dtype 同 p）。"""
        out_dtype = p.dtype
        with autocast_off():
            i = i.float()
            p = p.float()
            kt, kr = self.kernels()
            B, _, H, W = p.shape
            cov = self._coverage(H, W, p)
            use_r = (mode or self.mode) == MODE_DUAL
            if eta is None:
                eta = self.eta
            eta = eta.float().reshape(-1)
            if eta.numel() < self.n_outer:
                raise ValueError(f"eta 需要 {self.n_outer} 个，收到 {eta.numel()}")
            t = p
            a = p.new_zeros(B, N_ATOM, H, W)
            b = p.new_zeros(B, N_ATOM, H, W)
            rec_t = p.new_zeros(B, CH, H, W)
            resid = 0.0
            # 梯度检查点：320²×bs4 下 ISTA 的 8 步激活约 +1.7 GB，10 GB 卡会 OOM；
            # checkpoint 只留每步输入（重算换显存，语义/前向完全不变）。
            use_ckpt = bool(self.grad_ckpt) and torch.is_grad_enabled() and p.requires_grad
            for k in range(self.n_outer):
                for _ in range(self.k_inner):
                    a_prev = a
                    if use_ckpt:
                        a, b = checkpoint(self._ista_step, t, a, b, i, use_r,
                                          use_reentrant=False)
                    else:
                        a, b = self._ista_step(t, a, b, i, use_r)
                    if return_diag and k == self.n_outer - 1:
                        resid = float((a - a_prev).norm() / a.norm().clamp(min=1e-8))
                rec_t = F.conv_transpose2d(a, kt, padding=PAD) / cov
                if use_r:
                    rec_r = F.conv_transpose2d(b, kr, padding=PAD) / cov
                    t_star = (p + i - rec_r + rec_t) / 3.0
                else:
                    t_star = (p + rec_t) / 2.0
                t = t + eta[k] * (t_star - t)
            out = t
            if return_diag:
                with torch.no_grad():
                    d = out - p
                    zero_a = (a == 0).all(dim=1)
                    zero_b = (b == 0).all(dim=1) if use_r else torch.ones_like(zero_a)
                    diag = {
                        "delta_abs_mean": float(d.abs().mean()),
                        "delta_signed_mean": float(d.mean()),
                        "frac_both_zero": float((zero_a & zero_b).float().mean()),
                        "eta": [float(x) for x in eta.detach().cpu()],
                        "a_l1": float(a.abs().mean()),
                        "b_l1": float(b.abs().mean()),
                        "ista_a_rel_change": float(resid),
                    }
                return out.to(out_dtype), diag
        return out.to(out_dtype)

    def extra_repr(self):
        return (f"atoms={N_ATOM} patch={PATCH} tau={float(self.tau):.5f} "
                f"h={self.h:.5f} K={self.k_inner} outer={self.n_outer} mode={self.mode} "
                f"grad_ckpt={self.grad_ckpt}")


# --------------------------- 字典构造（训练域） ---------------------------
def patch_bank(img, step=1):
    """img [3,H,W] float32 [0,1] -> [N,75]（同一网格位置，用于对齐 T 与 I−T）。"""
    v = np.lib.stride_tricks.sliding_window_view(np.asarray(img, np.float32),
                                                 (PATCH, PATCH), axis=(1, 2))
    v = v[:, ::step, ::step]
    return np.ascontiguousarray(v.transpose(1, 2, 0, 3, 4).reshape(-1, DIM))


def build_dictionary(patches, n_atom=N_ATOM):
    """均值原子 + 中心化截断 SVD 原子；确定性符号。返回 (atoms[16,75], info)。"""
    patches = np.asarray(patches, np.float32)
    mu = patches.mean(0)
    nrm = float(np.linalg.norm(mu))
    if nrm > 1e-8:
        dc = mu / nrm
    else:
        dc = np.ones(DIM, np.float32) / math.sqrt(DIM)
    X = (patches - mu).astype(np.float64)
    cov = (X.T @ X) / max(len(X), 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][: n_atom - 1]
    atoms = [np.asarray(dc, np.float64)]
    for j in order:
        v = V[:, j].copy()
        k = int(np.argmax(np.abs(v)))
        if v[k] < 0:
            v = -v
        atoms.append(v / max(float(np.linalg.norm(v)), 1e-12))
    A = np.stack(atoms).astype(np.float32)
    nn = np.linalg.norm(A, axis=1)
    assert np.allclose(nn, 1.0, atol=1e-5), nn
    return A, dict(mean_norm=nrm, eig=w[::-1][: n_atom - 1].tolist(),
                   n_patch=int(len(patches)))


def tau_from_bank(t_patches, rho=RHO_TAU):
    """τ = ρ·median‖patch‖₂（[0,1] 像素单位，随像素标度线性）。"""
    nrm = np.linalg.norm(np.asarray(t_patches, np.float32), axis=1)
    med = float(np.median(nrm))
    return rho * med, dict(rho=rho, median_patch_norm=med)


def save_config(path, dt, dr, tau, meta=None):
    h, hinfo = ista_step(dt, dr)
    obj = dict(patch=PATCH, ch=CH, dim=DIM, n_atom=N_ATOM, h=h,
               k_inner=K_INNER, n_outer=N_OUTER, rho_tau=RHO_TAU,
               mode=MODE_DUAL, tau=float(tau),
               dt=np.asarray(dt, np.float32).tolist(),
               dr=np.asarray(dr, np.float32).tolist(),
               n_param=N_PARAM_TOTAL, step_info=hinfo, meta=meta or {})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    return path


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    for k, want in (("patch", PATCH), ("n_atom", N_ATOM), ("ch", CH)):
        if int(obj.get(k, -1)) != want:
            raise ValueError(f"字典配置 {k}={obj.get(k)} 与锁定值 {want} 不符（{path}）")
    dt = np.asarray(obj["dt"], np.float32)
    dr = np.asarray(obj["dr"], np.float32)
    return dt, dr, float(obj["tau"]), obj


def n2_from_config(path, eta_init=0.0, mode=None, k_inner=None, grad_ckpt=True):
    dt, dr, tau, obj = load_config(path)
    return N2Dict(dt, dr, tau, eta_init=eta_init,
                  k_inner=int(obj.get("k_inner", K_INNER) if k_inner is None else k_inner),
                  n_outer=int(obj.get("n_outer", N_OUTER)),
                  h=float(obj.get("h", ista_step(dt, dr)[0])),
                  mode=mode or obj.get("mode", MODE_DUAL),
                  grad_ckpt=grad_ckpt)
