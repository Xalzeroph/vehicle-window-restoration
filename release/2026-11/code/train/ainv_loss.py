"""尺度不变（affine-invariant）重建损失 —— 独立模块，不改原训练脚本。

动机
----
单帧无法确定"这张图该多亮"：I = τT + R 中 (λT, R + (1−λ)T) 给出完全相同的 I。
于是逐像素损失里有一大块误差**根本不该由模型承担**：
  实测 35 个臂系统性偏亮 6.66 灰阶，事后逐图 affine 校正可捞回 +0.63（最大 +1.89）dB。
本模块把这一自由度**解析消掉**，让梯度只去学结构。

两种可选形式
------------
1) `charbonnier_affine`：先求每个样本（逐通道）的最优 (a,b)，再用 charbonnier 算残差
       min_{a,b} Σ charbonnier(a·x + b, y)
   闭式解（最小二乘）：
       a = Cov(x,y)/Var(x),  b = ȳ − a·x̄
   归一化：除以目标方差 ⇒ 等价于 1−R²，范围固定在 [0,1]，常数输出得最差。
   a 下限 clamp(0.2) 防止反相解。

2) `si_snr_loss`：按目标能量归一化的负 SI-SNR（语音里的成熟形式）
       −10·log10( ‖α·x_t‖² / ‖x_t − α·x_t‖² ),  α = ⟨x,y⟩/‖y‖²
   与 (1) 的区别：不消除偏移 b，只消除增益。

为什么 detach(a,b) 是对的（包络定理）
------------------------------------
在最优点 ∂L/∂a = ∂L/∂b = 0，因此 a,b 随 x 变化的间接项恰好消失，
总导数 = 冻结 a,b 时的偏导。所以 a,b 由 x 现算、但反传时视作常数即可。

用法（训练脚本内）
------------------
    from ainv_loss import charbonnier_affine, si_snr_loss
    loss = charbonnier_affine(T_hat, T)            # [0,1] 范围
    loss = si_snr_loss(T_hat, T)                   # 负 SI-SNR

自检
----
    python ainv_loss.py --selftest
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

EPS = 1e-8
A_MIN = 0.2


def _fit_ab(x: torch.Tensor, y: torch.Tensor):
    """逐样本、逐通道最小二乘拟合 y ≈ a·x + b。返回 a,b，形状 [B,1,C,1,1]。"""
    dims = tuple(range(2, x.dim()))          # 空间维
    mx = x.mean(dim=dims, keepdim=True)
    my = y.mean(dim=dims, keepdim=True)
    var = ((x - mx) ** 2).mean(dim=dims, keepdim=True)
    cov = ((x - mx) * (y - my)).mean(dim=dims, keepdim=True)
    a = cov / var.clamp_min(EPS)
    a = a.clamp_min(A_MIN).detach()
    b = (my - a * mx).detach()
    return a, b


def charbonnier_affine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3,
                       normalize: bool = True) -> torch.Tensor:
    """min_{a,b} mean charbonnier(a·x+b, y)，按目标方差归一化。

    x,y: [B,C,H,W] 或 [B,C,...]，取值在 [0,1]。
    返回值在 [0,1] 量级；x 为常数时逼近 1.0（最差）。
    """
    a, b = _fit_ab(x, y)
    r = (a * x + b) - y
    loss = (r * r + eps * eps).sqrt().mean()
    if normalize:
        dims = tuple(range(2, y.dim()))
        vy = y.var(dim=dims, unbiased=False).mean().clamp_min(EPS)
        loss = loss / vy.sqrt()
    return loss


def si_snr_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """负 SI-SNR：只消除增益，不消除偏移。返回 -dB 的均值（越小越好）。"""
    dims = tuple(range(1, x.dim()))          # 全部通道+空间
    x0 = x - x.mean(dim=dims, keepdim=True)
    y0 = y - y.mean(dim=dims, keepdim=True)
    alpha = ((x0 * y0).sum(dim=dims, keepdim=True) /
             (y0 * y0).sum(dim=dims, keepdim=True).clamp_min(eps)).detach()
    target = alpha * y0
    noise = x0 - target
    ratio = (target ** 2).sum(dim=dims) / (noise ** 2).sum(dim=dims).clamp_min(eps)
    return -(10.0 * torch.log10(ratio.clamp_min(eps))).mean()


def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """原版（线性）charbonnier，留作对照。"""
    r = x - y
    return (r * r + eps * eps).sqrt()


def _selftest() -> int:
    torch.manual_seed(0)
    t = torch.rand(2, 3, 32, 32)
    checks = []
    # 1) 完美预测 -> 0
    v = charbonnier_affine(t, t).item()
    checks.append(("完美预测 ≈ 0", v < 1e-2, v))
    # 2) 全局增益/偏移不应被惩罚
    v2 = charbonnier_affine(0.4 * t + 0.3, t).item()
    checks.append(("增益+偏移后仍 ≈ 0", v2 < 1e-2, v2))
    # 3) 常数输出最差（≈1）
    v3 = charbonnier_affine(torch.full_like(t, 0.5), t).item()
    checks.append(("常数输出 ≈ 1（最差）", 0.5 < v3 < 2.0, v3))
    # 4) 结构错但亮度对 -> 明显大于 0
    v4 = charbonnier_affine(torch.flip(t, dims=[3]), t).item()
    checks.append(("结构错 ⇒ 明显 > 0", v4 > 0.1, v4))
    # 5) SI-SNR 对增益不敏感：完美（增益差）应给出极大 SNR ⇒ 损失很负
    v5 = si_snr_loss(0.5 * t, t).item()
    checks.append(("SI-SNR 增益不变 ⇒ 损失很负(<-40)", v5 < -40.0, v5))
    # 6) 梯度能反传（在非最优处测，最优点梯度理应≈0）
    x = torch.flip(t, dims=[3]).clone().requires_grad_(True)
    charbonnier_affine(x, t).backward()
    checks.append(("非最优点梯度非零", x.grad is not None and float(x.grad.abs().sum()) > 0,
                   float(x.grad.abs().sum())))
    ok = True
    for name, passed, val in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<26} value={val:.4f}")
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
