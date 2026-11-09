# -*- coding: utf-8 -*-
"""ARCH_H1（2026-10-30）：完美重构 Haar 子带上的有符号残差调制。

只新增文件，**不改** `RDnet_.py` / `modules_sig.py` / `reverse_function.py`：
  * `haar2` / `ihaar2`            —— 正交 2D Haar 分析/合成（逐 RGB 通道，无参数）
  * `ResidualSubbandModulator`   —— 新算子本体（2,969 参数）
  * `FullNet_NLP_Subband`        —— `FullNet_NLP` 的子类，**只重写 `forward`**：
        在最后一列（`i == num_subnet - 1`）decoder 输出之后、拼进 `x_img_out` 之前，
        对 transmission 残差 `d = dec[:, 3:6]` 的 Haar 细节系数做有符号调制。
        其余列、SubNet、ReverseFunction、Mona、Decoder 全部原样。

数学（与预登记一致；W = 逐通道三层正交 Haar，U = 满量程 1）
-------------------------------------------------------------
    d        = I - T0 = dec[:, 3:6]                （单位 U，与 I/T 同量纲）
    (ll1,{d1}) = W d ;  (ll2,{d2}) = W ll1 ;  (ll3,{d3}) = W ll2
    q_ll     = ll3                                 （最粗 LL 不调制 ⇒ 对齐 8×8 块均值不变）
    q_ℓ      = (1 + tanh z_ℓ) ⊙ d_ℓ ,  ℓ = 1,2,3   （逐 (band, rgb) 标量门，值域 (0,2)）
    T̂        = I - Wᵀ q

初始化：`out_conv` 权重与偏置全零 ⇒ z ≡ 0 ⇒ g ≡ 1 ⇒ **T̂ ≡ T0**（末层零初始化）。
此时隐藏层首步梯度为零、末层通常有梯度（见 `scripts/selfcheck_arch1030.py` 的实测断言）。

边界（显式 E / C，与数学审一致）
--------------------------------
H、W 不是 8 的倍数时，先用 `replicate` 边距补到 8 的倍数（**E**，补边），做完
分析/调制/合成后裁回原尺寸（**C**，裁剪）。此时：
  * 在补边后的网格上，`WᵀW = Id`（完美重构）严格成立；
  * 原图边缘不完整的 8×8 块"块均值不变"**不受保证**；
  * 原图误差范数一般**不等于**补边域系数误差范数（不把完美重构当原图能量守恒）。
训练裁剪 224/256/320 与测试面板 1280×832 均可被 8 整除 ⇒ 实际不触发 E/C。

Ω 与 identity
-------------
前向**不含** Ω（无掩码输入，也就没有"空掩码"代码路径）。identity 不是输入性质而是
参数状态性质：D=0（残差为零）时 T̂ = I 精确成立；`out_conv=0` 时 T̂ ≡ T0 精确成立。
反过来，"无反射输入时自动不动"仍需验收，不能由函数包含关系冒充识别保证。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from models.arch.RDnet_ import FullNet_NLP

__all__ = ["haar2", "ihaar2", "ResidualSubbandModulator", "FullNet_NLP_Subband"]


# --------------------------------------------------------------------------- Haar
def haar2(x):
    """正交 2D Haar 分析（逐通道，单级）。x:(B,C,H,W)，H、W 为偶数。

    返回 (LL, LH, HL, HH)，各 (B,C,H/2,W/2)。四个 2×2 系数的 L2 范数守恒：
    ||LL||²+||LH||²+||HL||²+||HH||² = ||x||²，即 W 正交；`ihaar2` 为其逆。
    """
    u = F.pixel_unshuffle(x, 2)                       # 通道块序 (00,01,10,11)
    a, b = u[:, 0::4], u[:, 1::4]
    c, d = u[:, 2::4], u[:, 3::4]
    return ((a + b + c + d) * 0.5,
            (a - b + c - d) * 0.5,
            (a + b - c - d) * 0.5,
            (a - b - c + d) * 0.5)


def ihaar2(ll, lh, hl, hh):
    """`haar2` 的逆：`ihaar2(*haar2(x)) == x`（FP32 下误差 ~1e-7）。"""
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    n, ch, h, w = ll.shape
    q = torch.stack([a, b, c, d], dim=2).reshape(n, 4 * ch, h, w)
    return F.pixel_shuffle(q, 2)


# --------------------------------------------------------------------- 调制算子
class ResidualSubbandModulator(nn.Module):
    """在 Haar 子带上调制有符号残差 d 的**细节**系数（LL 不动）。

    新增参数共 2,969（三尺度共享同一套头）：
      feat_proj  Conv1×1 (64→16)      weight[16,64,1,1] + bias[16] = 1,040
      in_conv    Conv1×1 (40→32)      weight[32,40,1,1] + bias[32] = 1,312
      dw         Conv3×3 depthwise(32) weight[32,1,3,3] + bias[32] =   320
      out_conv   Conv1×1 (32→9)       weight[9,32,1,1]  + bias[9]  =   297
    头输入 40 = Haar(I) 12 + Haar(d) 12 + 投影特征 16（三尺度共享）。
    """

    def __init__(self, feat_dim=64, feat_hidden=16, width=32, levels=3):
        super().__init__()
        if levels != 3:
            raise ValueError("本实现固定三层（对齐 8×8 块边界），收到 levels=%r" % (levels,))
        self.levels = int(levels)
        self.feat_dim = int(feat_dim)
        self.feat_proj = nn.Conv2d(feat_dim, feat_hidden, kernel_size=1)
        in_ch = 3 * 4 + 3 * 4 + feat_hidden           # Haar(I) 12 + Haar(d) 12 + 特征
        self.in_conv = nn.Conv2d(in_ch, width, kernel_size=1)
        self.dw = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width)
        self.act = nn.GELU()
        self.out_conv = nn.Conv2d(width, 3 * 3, kernel_size=1)   # (band, rgb) = (3, 3)
        # 与 RDNet 内部 `_init_weights` 同风格；末层再零初始化 -> 初值 g≡1
        for m in (self.feat_proj, self.in_conv, self.dw):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.last_aux = None                         # 诊断用：每级 mean|g-1|（detached 标量）

    def forward(self, x_in, d_pred, feat, gate_identity=False):
        """返回被调制的残差 d_mod（dtype 与 `d_pred` 一致，便于保持原 forward 的数值口径）。

        x_in  : (B,3,H,W) 输入 I
        d_pred: (B,3,H,W) 原 decoder 给出的 transmission 残差 d = I - T0
        feat  : (B,C,·,·) 最终 c0 特征（本配置 C=64，空间 1/2）
        gate_identity: True 时 g≡1（诊断用：算子在校但不起作用，必须数值上回到 T0）
        """
        n, _, h, w = d_pred.shape
        pad_h, pad_w = (-h) % 8, (-w) % 8            # E：补到 8 的倍数
        x = x_in.float()
        d = d_pred.float()
        if pad_h or pad_w:
            pad = (0, pad_w, 0, pad_h)
            x = F.pad(x, pad, mode="replicate")
            d = F.pad(d, pad, mode="replicate")
        hg, wg = h + pad_h, w + pad_w

        # ---- 分析：三级金字塔（I 与 d 同步分解）
        pyr_i, pyr_d = [], []
        cur_i, cur_d = x, d
        for _ in range(self.levels):
            li = haar2(cur_i)
            ld = haar2(cur_d)
            pyr_i.append(li)
            pyr_d.append(ld)
            cur_i, cur_d = li[0], ld[0]

        f = self.feat_proj(feat.float())             # (B,16,~H/2,~W/2)
        gates = []
        for lev in range(self.levels):
            hl, wl = hg >> (lev + 1), wg >> (lev + 1)
            if f.shape[-2] != hl or f.shape[-1] != wl:
                fl = F.interpolate(f, size=(hl, wl), mode="area")
            else:
                fl = f
            inp = torch.cat([pyr_i[lev][0], pyr_i[lev][1], pyr_i[lev][2], pyr_i[lev][3],
                             pyr_d[lev][0], pyr_d[lev][1], pyr_d[lev][2], pyr_d[lev][3],
                             fl], dim=1)                     # (B,40,hl,wl)
            z = self.out_conv(self.act(self.dw(self.in_conv(inp))))   # (B,9,hl,wl)
            if gate_identity:
                gates.append(torch.ones_like(z))
            else:
                gates.append(1.0 + torch.tanh(z))                # (0,2)
        if not gate_identity:
            with torch.no_grad():
                self.last_aux = tuple(g.detach().sub(1.0).abs().mean() for g in gates)
        else:
            self.last_aux = tuple(torch.zeros((), device=d.device) for _ in gates)

        # ---- 合成：最粗 LL 不动，逐级把 (g⊙d) 贴回去
        cur = pyr_d[self.levels - 1][0]
        for lev in range(self.levels - 1, -1, -1):
            _, lh, hl_, hh = pyr_d[lev]
            det = torch.stack([lh, hl_, hh], dim=2)          # (B,Crgb,3band,hl,wl)
            g = gates[lev].view(n, 3, 3, det.shape[-2], det.shape[-1]).permute(0, 2, 1, 3, 4)
            gated = det * g                                  # (B,Crgb,3band,hl,wl)
            cur = ihaar2(cur, gated[:, :, 0], gated[:, :, 1], gated[:, :, 2])
        d_mod = cur[..., :h, :w]                             # C：裁回原尺寸
        return d_mod.to(d_pred.dtype)


# ------------------------------------------------------------------ 网络子类
class FullNet_NLP_Subband(FullNet_NLP):
    """`FullNet_NLP` + 最后一列的 Haar 子带残差调制。

    `__init__` 完全复用父类（含 focal.pth 的 strict 加载、`_init_weights`、Mona 插入点），
    只额外挂 `self.subband`；`forward` 是父类 forward 的逐行照抄，唯一改动见
    `# >>> ARCH_H1` 标注处。
    """

    def __init__(self, *args, subband_feat_dim=64, subband_feat_hidden=16,
                 subband_width=32, **kwargs):
        # 父类 `FullNet_NLP.__init__` 会对传入的 channels 列表做**原地 reverse**
        # （`channels.reverse()`，随后再交给 Decoder）。同一个 list 对象复用两次会造出
        # 通道序反了的第二张网（实测会让 decoder_blocks.* 尺寸对不上 ckpt）。
        # 这里显式拷贝一份，把这个坑封在子类里。
        if kwargs.get("channels") is not None:
            kwargs["channels"] = list(kwargs["channels"])
        super().__init__(*args, **kwargs)
        self.subband = ResidualSubbandModulator(feat_dim=subband_feat_dim,
                                                feat_hidden=subband_feat_hidden,
                                                width=subband_width, levels=3)
        # 诊断开关（推理用）：True -> g≡1，应当数值上回到原网络输出
        self.subband_gate_identity = False

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
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)
            if i > (self.num_subnet - self.Loss_col):
                dec = self.decoder_blocks[-1](c3, c2, c1, c0)
                # >>> ARCH_H1：仅最后一列的 transmission 残差做子带调制；
                #     dec = cat([x_clean, x_ref])，训练器把 out[:, 3:] 当 T̂，故改 x_ref = dec[:, 3:6]
                if i == self.num_subnet - 1:
                    d_ref = dec[:, 3:6]
                    d_mod = self.subband(x_in, d_ref, c0,
                                         gate_identity=getattr(self, 'subband_gate_identity', False))
                    dec = torch.cat([dec[:, :3], d_mod], dim=1)
                x_img_out.append(torch.cat([x_in, x_in], dim=-3) - dec)

        return x_cls_out, x_img_out
