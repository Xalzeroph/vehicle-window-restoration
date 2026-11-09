# -*- coding: utf-8 -*-
"""I1（2026-10-31）：全局均值与空间缓变残差分离的微型光度适配器。

出处/借形：AdaptIR（arXiv:2312.08881）的 global-spatial / channel 表征分工。
本文件是**本次提出的改造**，不是论文原式：只借"全局支路 + 空间支路分工"这一个结构想法，
不复制完整 MoE、不替换 Mona、不接扩散先验。

纪律（项目硬约束）
------------------
只**新增**本文件，不改 `RDnet_.py` / `rdnet_peft.py` / `reverse_function.py` /
`finetune_rdnet_s2.py`。`FullNet_NLP_Photo` 是 `FullNet_NLP` 的子类，只重写 `forward`。

数学（与审核通过的设计一致）
----------------------------
设某列 decoder 拼出的六通道输出为 `out = cat([Ř, Ŷ], 1)`（`Ŷ = out[:, 3:]` 即透射层，
口径以训练器 `finetune_rdnet_s2.py` 的 `R_hat, T_hat = out[:, :3], out[:, 3:]` 为准）。
记 F0 = 该列的 c0（64 通道）、F3 = 该列的 c3（512 通道）、I = 输入：

    g        = GAP(F3)                              ∈ R^512
    μ, σ     = mean_xy(I), sqrt(var_xy(I) + ε)      ∈ R^3   （FP32 累积，ε = 1e-6）
    h        = GELU(W1 [g, μ, σ] + b1)              ∈ R^32            （W1: 518→32）
    b_c      = W2 h + b2                            ∈ R^3             （W2: 32→3，零初始化）
    P        = adaptive_avg_pool2d(F0, 8)           ∈ R^{64×8×8}
    z        = Conv1x1_{16→3}( GELU( DWConv3x3_{16}( Conv1x1_{96→16}([P, broadcast(h)]) ) ) )
    S        = bilinear_resize(z → (H, W); align_corners=False)
    Ŷ'       = Ŷ + b_c + (S − mean_xy(S))                             （去均值在 H×W 上做，FP32）

均值与零均值空间场分离，避免两条支路重复承担 DC。所有监督列**共享同一适配器实例**
（同一份权重，逐列各算一次），因此参数量不随列数增长。

量纲：I、Ŷ、b_c、S 均为归一化 RGB 强度（0..1 连续值），池化/插值/求均值不改变量纲，
不存在跨量纲相加。损失仍作用在原始连续输出上，本模块**不做 clip、不做量化**。

初始化与退化
------------
`fc2`（W2,b2）与 `out_conv`（空间末层）全零 ⇒ 初始化时 b_c ≡ 0、S ≡ 0 ⇒ **Ŷ' ≡ Ŷ**
（FP16 下加 0 也是精确的），即整网输出与基线逐元素相同。`photo_enabled=False` 时
`forward` 连适配器都不调用（真正旁路，不消耗 RNG、不产生额外计算），严格退化为基线。

首步梯度（数学审结论，已由 `scripts/selfcheck_photo1031.py` 实测断言）
--------------------------------------------------------------------
记 g_out = ∂L/∂Ŷ'：
  * ∂L/∂b2 = Σ_xy g_out，一般非零 ⇒ W2/b2 首步有梯度；
  * ∂L/∂W_out ∝ z ⊗ g_out，z 在初始化时非零（上游随机初始化）⇒ 空间末层首步有梯度；
  * ∂L/∂h 经过 W2（=0）与空间末层（=0）两条路 ⇒ **恒为 0** ⇒ W1/b1、in_conv、dw
    首步梯度为零，**末层更新之后才打开**。这是预期现象，不是死支路；
    不要为了"修死梯度"改成非零输出初始化（那会破坏"初始化严格等价基线"）。

审核更正（已落实）
------------------
1. 删除空间末层的 3 维 bias：它经插值仍是常数场，被 `S − mean_xy(S)` 完全消掉，
   梯度恒为零（死参数）。参数由 18,470 → **18,467**。
2. 插值固定 `mode='bilinear', align_corners=False`；DWConv 固定 `padding=1`；
   输入统计量固定 `sqrt(var + 1e-6)`；AdaptIR 的 `correction` 输入固定为 0（不接额外输入）。
3. 去均值在**输出实际尺寸**上做、FP32 累积（FP16 累加 320² 会掉精度）。

边界（不可宣称的东西）
----------------------
* 确定性特征与输入统计量不会创造额外观测：相同输入对应不同真实透射层时，网络只能给
  一个结果；若基线已是给定输入的条件均值，确定性校正不保证降低总体期望 MSE
  （也不等价于逐图 PSNR 平均的最优性）。
* `S` 只是低容量缓变场（8×8 上采样），不是严格带限信号，也没有对 Ŷ 做任何滤波；
  不能凭 b/S 的可视化声称"结构/光度解耦"——新支路与原 decoder 之间的分解不唯一。
* 学到的只是训练分布下**可预测的条件校正**，不宣称恢复物理 ρ 或识别 Ω。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from models.arch.RDnet_ import FullNet_NLP

__all__ = ["GlobalSpatialPhotometricAdapter", "FullNet_NLP_Photo", "PHOTO_N_PARAMS"]

#: 预登记参数规模（模块级断言用；审核更正后的值：18,470 − 3）
PHOTO_N_PARAMS = 18467


class GlobalSpatialPhotometricAdapter(nn.Module):
    """预测 RGB 全局均值（DC）+ 空间缓变零均值残差（AC）的微型支路。

    参数（共 18,467，四列共享）::

        fc1       Linear(518→32)   weight[32,518] + bias[32] = 16,608
        fc2       Linear(32→3)     weight[3,32]   + bias[3]  =     99   （零初始化）
        in_conv   Conv1×1(96→16)   weight[16,96,1,1] + bias[16] = 1,552
        dw        Conv3×3 dw(16)   weight[16,1,3,3]  + bias[16] =   160
        out_conv  Conv1×1(16→3)    weight[3,16,1,1]  **无 bias** =    48   （零初始化）

    518 = GAP(F3) 512 + μ(I) 3 + σ(I) 3；96 = P(64) + broadcast(h)(32)。
    """

    def __init__(self, f0_dim: int = 64, f3_dim: int = 512, hidden: int = 32,
                 grid: int = 8, spatial: int = 16, stats_eps: float = 1e-6):
        super().__init__()
        self.f0_dim = int(f0_dim)
        self.f3_dim = int(f3_dim)
        self.hidden = int(hidden)
        self.grid = int(grid)
        self.spatial = int(spatial)
        self.stats_eps = float(stats_eps)

        self.fc1 = nn.Linear(self.f3_dim + 6, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 3)
        self.in_conv = nn.Conv2d(self.f0_dim + self.hidden, self.spatial, kernel_size=1)
        self.dw = nn.Conv2d(self.spatial, self.spatial, kernel_size=3, padding=1,
                            groups=self.spatial)
        self.act = nn.GELU()
        # 审核更正 1：空间末层不带 bias（常数场会被 mean_xy 减掉 ⇒ 梯度恒为 0）
        self.out_conv = nn.Conv2d(self.spatial, 3, kernel_size=1, bias=False)

        # 与 RDNet 内部 `_init_weights` 同风格（trunc_normal 0.02 + bias 0）
        for m in (self.fc1, self.in_conv, self.dw):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        # 末两层零初始化 ⇒ 初始化时 b_c ≡ 0、S ≡ 0 ⇒ Ŷ' ≡ Ŷ
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.out_conv.weight)

        self.last_stats_t = None    # 诊断用（4 元张量 [abs_b, std_s, max_s, abs_h]，未同步）

    # ------------------------------------------------------------------ 前向
    def forward(self, x_in, f0, f3, y_t):
        """返回校正后的透射层（dtype 与 `y_t` 一致，便于保持原 forward 的数值口径）。

        x_in : (B, 3, H, W)      输入 I（归一化强度）
        f0   : (B, 64, ·, ·)     该列 c0（高分辨率小通道；本配置 64 通道）
        f3   : (B, 512, ·, ·)    该列 c3（低分辨率大通道；本配置 512 通道）
        y_t  : (B, 3, H, W)      原 decoder 给出的透射层 Ŷ = out[:, 3:]
        """
        if f0.shape[1] != self.f0_dim:
            raise RuntimeError(f"c0 通道 {f0.shape[1]} ≠ 适配器 f0_dim {self.f0_dim}")
        if f3.shape[1] != self.f3_dim:
            raise RuntimeError(f"c3 通道 {f3.shape[1]} ≠ 适配器 f3_dim {self.f3_dim}")

        b = y_t.shape[0]
        feat_dtype = f0.dtype

        # ---- 全局支路：均值（DC）
        gap = F.adaptive_avg_pool2d(f3, 1).flatten(1)                 # (B, 512)
        xf = x_in.float()                                             # 统计量 FP32 累积
        mu = xf.mean(dim=(2, 3))                                      # (B, 3)
        sd = (xf.var(dim=(2, 3), unbiased=False) + self.stats_eps).sqrt()
        stats = torch.cat([gap.to(feat_dtype), mu.to(feat_dtype), sd.to(feat_dtype)], dim=1)
        h = self.act(self.fc1(stats))                                 # (B, 32)
        b_c = self.fc2(h)                                             # (B, 3)，零初始化

        # ---- 空间支路：缓变零均值场（AC）
        pooled = F.adaptive_avg_pool2d(f0, self.grid)                 # (B, 64, 8, 8)
        hmap = h[:, :, None, None].expand(-1, -1, self.grid, self.grid)
        z = self.out_conv(self.act(self.dw(self.in_conv(
            torch.cat([pooled, hmap], dim=1)))))                      # (B, 3, 8, 8)
        s = F.interpolate(z, size=y_t.shape[-2:], mode="bilinear", align_corners=False)

        sf = s.float()
        sf = sf - sf.mean(dim=(2, 3), keepdim=True)                   # DC 归零（FP32 累积）
        s = sf.to(y_t.dtype)

        out = y_t + b_c.to(y_t.dtype).view(b, 3, 1, 1) + s

        # 诊断（不触发 GPU 同步：只建一个 4 元张量，训练脚本在打日志时才 .tolist()）
        with torch.no_grad():
            self.last_stats_t = torch.stack([
                b_c.detach().float().abs().mean(),
                s.detach().float().std(dim=(2, 3), unbiased=False).mean(),
                s.detach().float().abs().max(),
                h.detach().float().abs().mean(),
            ])          # [abs_b, std_s, max_s, abs_h]
        return out


# ------------------------------------------------------------------ 网络子类
class FullNet_NLP_Photo(FullNet_NLP):
    """`FullNet_NLP` + 每列输出的光度适配器（只动后 3 通道，前 3 通道同 storage 视图）。

    `__init__` 完全复用父类（含 focal.pth 的 strict 加载、`_init_weights`、Mona 插入点），
    只额外挂 `self.photo`；`forward` 是父类 forward 的逐行照抄，唯一改动见 `# >>> I1` 标注处。
    """

    def __init__(self, *args, photo_hidden: int = 32, photo_grid: int = 8,
                 photo_spatial: int = 16, photo_f0_dim: int = 64, photo_f3_dim: int = 512,
                 photo_enabled: bool = True, photo_seed: int = 20261031, **kwargs):
        # 父类 `FullNet_NLP.__init__` 会对传入的 channels 列表做**原地 reverse**
        # （channels.reverse() 后再交给 Decoder）。同一 list 复用两次会造出通道序反了的
        # 第二张网（ARCH_H1 已踩过）。这里显式拷贝一份，把坑封在子类里。
        if kwargs.get("channels") is not None:
            kwargs["channels"] = list(kwargs["channels"])
        super().__init__(*args, **kwargs)

        # 适配器初始化**隔离 RNG**：不得改变基线后续的数据/dropout 随机流。
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            torch.manual_seed(int(photo_seed))
            self.photo = GlobalSpatialPhotometricAdapter(
                f0_dim=photo_f0_dim, f3_dim=photo_f3_dim, hidden=photo_hidden,
                grid=photo_grid, spatial=photo_spatial)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

        self.photo_enabled = bool(photo_enabled)
        self.photo_diag = None          # dict(cols=本次列数, stats=[逐列 4 元张量])，诊断用

    # ------------------------------------------------------------------ 校正
    def _photo_correct(self, out6, x_in, c0, c3):
        """把六通道输出 `out6` 的后 3 通道（透射层）替换为校正版；前 3 通道原样（视图）。"""
        if out6.shape[1] != 6:
            raise RuntimeError(f"预期六通道输出，实际 {out6.shape[1]}")
        y_t = out6[:, 3:]
        y_new = self.photo(x_in, c0, c3, y_t)
        self._photo_acc.append(None if self.photo.last_stats_t is None
                               else self.photo.last_stats_t)
        return torch.cat([out6[:, :3], y_new], dim=1)

    def forward(self, x_in, alpha, prompt=True):
        x_cls_out = []
        x_img_out = []
        c0, c1, c2, c3 = 0, 0, 0, 0
        interval = self.num_subnet // 4                      # noqa: F841 (父类同款，保持逐行一致)

        x_base, x_stem = self.baseball(x_in)
        c0, c1, c2, c3 = x_base
        x_stem = self.baseball_adapter[0](x_stem)
        c0, c1, c2, c3 = self.baseball_adapter[1](c0),\
                         self.baseball_adapter[2](c1),\
                         self.baseball_adapter[3](c2),\
                         self.baseball_adapter[4](c3)
        if prompt is True:
            prompt_alpha = self.prompt(alpha)
            prompt_alpha = prompt_alpha.unsqueeze(-1).unsqueeze(-1)
            x = prompt_alpha * x_stem
        else:
            x = x_stem
        self._photo_acc = []            # >>> I1：本列诊断累积（仅在启用时被填）
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)
            if i > (self.num_subnet - self.Loss_col):
                out6 = torch.cat([x_in, x_in], dim=-3) - self.decoder_blocks[-1](c3, c2, c1, c0)
                # >>> I1：仅校正后 3 通道（透射层），前 3 通道（反射层）逐元素不变
                if self.photo_enabled:
                    out6 = self._photo_correct(out6, x_in, c0, c3)
                x_img_out.append(out6)
        # >>> I1：记录本次 forward 的监督列数与逐列诊断张量（[abs_b,std_s,max_s,abs_h]）
        #     张量不在这里 .tolist()，避免每 iter 多次 GPU 同步；训练脚本打日志时才取。
        if self.photo_enabled and self._photo_acc:
            self.photo_diag = dict(cols=len(self._photo_acc),
                                   stats=[d for d in self._photo_acc if d is not None])
        else:
            self.photo_diag = None
        return x_cls_out, x_img_out
