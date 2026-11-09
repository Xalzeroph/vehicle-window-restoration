#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GCF：梯度方向余弦损失与同口径指标（2026-10-20）。

为什么必须是**方向余弦**而不是 L1 梯度
------------------------------------
项目里已有 `finetune_rdnet_s2.py::grad_fidelity = ‖∇T̂−∇T‖₁`（9 个臂用过）。它的权重
被**大幅值像素**主导，而那些大幅值往往是**反射层的边缘** —— 于是它同时"想保住反射边缘"，
与去反射目标打架。实测：带该损失的 `mix_b2` 在 1019 上 GC=0.579，比 identity 的 0.599 更差。

余弦形式先**归一化掉幅值**，只惩罚方向：
    L_gc = 1 − mean_{p∈S} cos( ∇T̂(p), ∇T(p) )
它才是评测指标 GC 的直接对齐目标。

S 的选择
--------
取 **GT 的** top-k% 强梯度像素（默认 15%），原因：
· 用 GT 而非预测选定，避免"模型越糊、S 越小、损失越容易"的可游戏性；
· 弱梯度处方向本就不稳定（JPEG 噪声），纳入只会加噪。

注意：训练与评测**必须共用同一份实现**，否则口径漂移（本项目已因此踩过坑）。
本文件同时提供 `grad_cosine_loss`（可微）与 `grad_cosine_metric`（no-grad）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-6


def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=device, dtype=dtype)
    ky = kx.t()
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def sobel_xy(x: torch.Tensor):
    """逐通道 Sobel。x: (B,C,H,W) -> gx, gy 同形状。"""
    c = x.shape[1]
    kx, ky = _sobel_kernels(x.device, x.dtype)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return gx, gy


def _cos_map(a_gx, a_gy, b_gx, b_gy):
    """逐像素方向余弦，返回 (B,C,H,W)；分母 clamp 防零向量。"""
    num = a_gx * b_gx + a_gy * b_gy
    den = torch.sqrt(a_gx ** 2 + a_gy ** 2 + _EPS) * torch.sqrt(b_gx ** 2 + b_gy ** 2 + _EPS)
    return num / den


def grad_cosine_loss(T_hat: torch.Tensor, T: torch.Tensor,
                     topk: float = 0.15) -> torch.Tensor:
    """1 − 方向余弦均值。T_hat/T: (B,3,H,W)，建议 [0,1] 或 [0,255] 同尺度。

    用 GT 选像素集；AMP 下强制 float32（cosine 反传在 fp16 易溢出）。
    """
    Th = T_hat.float()
    Tg = T.float()
    hgx, hgy = sobel_xy(Th)
    tgx, tgy = sobel_xy(Tg)

    mag_t = torch.sqrt(tgx ** 2 + tgy ** 2 + _EPS)          # (B,C,H,W)
    # 逐图取 top-k 阈值（对每张图分别选，避免批内亮度差导致偏置）
    B, C = mag_t.shape[0], mag_t.shape[1]
    flat = mag_t.reshape(B, C, -1)
    k = max(1, int(round(flat.shape[-1] * topk)))
    thr = flat.kthvalue(flat.shape[-1] - k + 1, dim=-1).values.view(B, C, 1, 1)
    mask = (mag_t >= thr).float()

    cos = _cos_map(hgx, hgy, tgx, tgy)                      # (B,C,H,W)
    denom = mask.sum().clamp(min=1.0)
    return 1.0 - (cos * mask).sum() / denom


@torch.no_grad()
def grad_cosine_metric(T_hat: torch.Tensor, T: torch.Tensor,
                       topk: float = 0.15) -> float:
    """评测用：与 `grad_cosine_loss` 完全同口径，返回 cos 均值（=1−loss）。"""
    return float(1.0 - grad_cosine_loss(T_hat, T, topk=topk))


def grad_cosine_metric_bgr255(pred_bgr: "object", gt_bgr: "object",
                              topk: float = 0.15) -> float:
    """numpy (H,W,3) uint8/float，BGR 或 RGB 任意但两侧须一致；内部转 tensor 计算。

    评测脚本（table_final_1019.py 的 grad_cos）已有一份 SQL 实现，本函数供新臂
    自检用，两者在 test_gcf_loss.py 里做过一致性核对。
    """
    import numpy as np
    a = torch.from_numpy(np.ascontiguousarray(pred_bgr)).permute(2, 0, 1)[None].float()
    b = torch.from_numpy(np.ascontiguousarray(gt_bgr)).permute(2, 0, 1)[None].float()
    return grad_cosine_metric(a, b, topk=topk)


if __name__ == "__main__":
    # 自检：自身 cos=1；反向 cos=−1；随机≈0
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)
    print("self   1-L =", float(grad_cosine_loss(x, x)))           # ≈0
    print("neg    1-L =", float(grad_cosine_loss(-x, x)))          # ≈2
    y = torch.rand(2, 3, 64, 64)
    print("rand   1-L =", float(grad_cosine_loss(y, x)))           # ≈1
    xs = (x * 50).requires_grad_(True)
    loss = grad_cosine_loss(xs, x * 50)
    loss.backward()
    print("grad finite:", bool(torch.isfinite(xs.grad).all()))
