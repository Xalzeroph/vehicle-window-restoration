#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1 输出端有界残差倍率门（bounded residual multiplier gate）—— 模块本体。

定位（审查后的准确说法，勿升级）
--------------------------------
本模块**不是**特征 prompt、**不是**物理可辨识的反射分离、**不是**"有符号去反射方向"。
它是对基线输出 T0 的一个受限输出残差校正：

    D   = I - T0                       # 基线已有的一维修正方向（逐像素 RGB）
    T^  = T0 - a(I,T0) * D             # a 由 12 通道输入预测，逐像素、跨 RGB 共享
        = I - g * D,  g = 1 + a in [0, 2]   (默认 clip=1, sym 模式)

- g >= 0 恒成立：本门**只能减弱或增强**原有去除，不能反转方向；D=0 处无控制力。
- a=0 -> T^=T0（基线）；a=-1 -> I；a=+1 -> 2T0-I（可能越界，最终仍由部署端 clamp 到 [0,1]）。
- 输入 u = [I, T0, D, AvgPool3x3(D)] 全部由 I 与基线输出确定，**没有引入新观测**。
- 逐图 oracle 上的可修空间（0.21-0.48 dB）不能推出单帧可预测性；L_quiet 的剂量反应也可能
  只是"整体少动"，不证明"哪里不该动"可跨场景预测（审查意见原文）。本模块只提供一个可执行、
  可关闭、可被探针否决的低容量假设。

结构（NAFNet 元件迁移；元件本身不是新机制）
-------------------------------------------
    u (12) -> Conv1x1 12->32 -> DW3x3 32 -> SimpleGate 32->16 -> SCA 16->16 -> Conv1x1 16->1
    w = W_out h + b_out;  a = clamp(w, -clip, clip)
末层 Conv1x1 权重与偏置**全零**，SCA 初始化为恒等乘子（W_s=0, b_s=1），其余随机非零。
因此初始化时 a≡0，逐位等于基线输出（float 最大偏差应为 0.0；uint8 逐像素相同）。

参数（width=32, hid=16）
------------------------
    Conv1x1 12->32 : 12*32+32        = 416
    DW3x3  32      : 32*3*3+32       = 320
    SCA    16->16  : 16*16+16        = 272
    Conv1x1 16->1  : 16+1            =  17
    合计                                1025
（SCA 的 1x1 卷积作用在 GAP 后的 1x1 特征上；它只调制通道，不做空间路由。）

必须随结果一起报的已知限制
--------------------------
1. clip 为 hard clamp：饱和区（|w|>=clip）对 w 的梯度**严格为 0**。必须报饱和率，
   不得事后换激活函数续命。训练脚本 `--a1-gate clip` 与探针都会报 sat_frac。
2. 有界范围只约束"相对 D 的倍率"，不约束输出值域/细节损伤；最终 clamp/量化顺序由部署端
   `predict_with_gate` 固定为：门 -> clamp(0,1) -> *255（与表口径一致）。
3. 本门不预测 Ω、不输出掩码；训练期不使用任何真 Ω 监督（本项目训练域本来也没有真 Ω 掩码）。
4. 关闭（gate=None）时不构造模块、不消耗任何 RNG、不增加参数 —— A/B 的 B 臂就是原脚本。
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DEFAULT_CFG",
    "ResidualGateA1",
    "make_gate",
    "apply_gate",
    "predict_with_gate",
    "param_report",
    "isolated_init",
]

DEFAULT_CFG = dict(
    enabled=True,
    width=32,
    hid=16,
    clip=1.0,
    seed=20261101,
    mode="sym",          # sym: a in [-clip, clip]; cons: a in [-clip, 0]（仅诊断用）
    scope="last_col",    # 只作用于最后一列 decoder 的 transmission 三通道
    in_ch=12,
)


# --------------------------------------------------------------------------------------
# RNG 隔离：构造本模块绝不能改变全局 RNG 状态，否则同 seed 的 A/B 数据顺序会漂移
# --------------------------------------------------------------------------------------
class isolated_init:
    """with isolated_init(seed): 用独立 seed 初始化，退出时恢复 CPU/全部可见 CUDA RNG。"""

    def __init__(self, seed: int):
        self.seed = int(seed)

    def __enter__(self):
        self._cpu = torch.get_rng_state()
        self._cuda = None
        if torch.cuda.is_available():
            try:
                self._cuda = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
            except Exception:  # noqa: BLE001
                self._cuda = None
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self._cpu)
        if self._cuda is not None:
            for i, st in enumerate(self._cuda):
                try:
                    torch.cuda.set_rng_state(st, i)
                except Exception:  # noqa: BLE001
                    pass
        return False


class ResidualGateA1(nn.Module):
    """12 -> 32 -> DW3x3 -> SG(16) -> SCA(16) -> 1，输出 clip 到 [-clip, clip] 的逐像素 a。"""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        c = dict(DEFAULT_CFG)
        if cfg:
            c.update(cfg)
        self.cfg = c
        self.width = int(c["width"])
        self.hid = int(c["hid"])
        self.clip = float(c["clip"])
        self.mode = str(c["mode"])
        if self.width % 2 != 0:
            raise ValueError("width 必须为偶数（SimpleGate 通道对乘）")
        if self.width // 2 != self.hid:
            raise ValueError("本实现要求 hid == width//2（与方案参数账一致）")
        if self.mode not in ("sym", "cons"):
            raise ValueError(f"未知 mode {self.mode}")

        # 注意：Conv2d 的构造本身也会消耗全局 RNG，因此**整段构造**必须在隔离上下文内，
        # 否则同 seed 的 A/B 会因 DataLoader 洗牌序列漂移（探针 unit 会断言这一点）。
        with isolated_init(self.seed_from_cfg(c)):
            self.conv_in = nn.Conv2d(int(c["in_ch"]), self.width, 1)
            self.dw = nn.Conv2d(self.width, self.width, 3, padding=1, groups=self.width)
            self.sca = nn.Conv2d(self.hid, self.hid, 1)
            self.conv_out = nn.Conv2d(self.hid, 1, 1)
            for m in (self.conv_in, self.dw):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)
            # SCA 初始为恒等乘子：W_s=0, b_s=1
            nn.init.zeros_(self.sca.weight)
            nn.init.ones_(self.sca.bias)
            # 末层零初始化 -> a 恒为 0 -> T^ == T0（逐位）
            nn.init.zeros_(self.conv_out.weight)
            nn.init.zeros_(self.conv_out.bias)

        self.last: dict = {}

    @staticmethod
    def seed_from_cfg(c: dict) -> int:
        return int(c.get("seed", DEFAULT_CFG["seed"]))

    # ---------------- 特征（探针与训练共用同一条路径） ----------------
    def features(self, I: torch.Tensor, T0: torch.Tensor):
        """返回 (h, D)：h 为 16 通道 SCA 后特征；D=I-T0。"""
        D = I - T0
        u = torch.cat([I, T0, D, F.avg_pool2d(D, 3, stride=1, padding=1)], dim=1)
        h = self.conv_in(u)
        h = self.dw(h)
        h1, h2 = h.chunk(2, dim=1)
        h = h1 * h2                                   # SimpleGate
        h = h * self.sca(F.adaptive_avg_pool2d(h, 1))  # Simplified Channel Attention
        return h, D

    def gate_from_features(self, h: torch.Tensor) -> torch.Tensor:
        w = self.conv_out(h)
        if self.mode == "cons":      # 诊断用：只允许 g in [0,1]（a<=0）
            return w.clamp(min=-self.clip, max=0.0)
        return w.clamp(min=-self.clip, max=self.clip)

    def forward(self, I: torch.Tensor, T0: torch.Tensor) -> torch.Tensor:
        h, D = self.features(I, T0)
        a = self.gate_from_features(h)
        T = T0 - a * D
        with torch.no_grad():
            ad = a.detach()
            self.last = dict(
                a_mean=float(ad.mean()),
                a_abs_mean=float(ad.abs().mean()),
                a_min=float(ad.amin()),
                a_max=float(ad.amax()),
                sat_frac=float((ad.abs() >= self.clip).float().mean()),
                amp_frac=float((ad > 0).float().mean()),
                neg_frac=float((ad < 0).float().mean()),
                d_abs_mean=float(D.detach().abs().mean()),
            )
        return T


def make_gate(cfg: dict | None = None, device=None) -> ResidualGateA1:
    g = ResidualGateA1(cfg)
    if device is not None:
        g = g.to(device)
    return g


def apply_gate(outs, I: torch.Tensor, gate: ResidualGateA1 | None):
    """把门作用到最后一列输出的 transmission 三通道；R 分支与前三列完全不动。

    outs: list[tensor(B,6,H,W)]，第 3..6 通道 = T^（口径以训练器为准）。
    返回新的 list；gate is None 时原样返回（不复制、不消耗 RNG）。

    注意：门**强制在 fp32 下计算**（内部关 autocast）。1025 个参数在 fp16 下没有任何收益，
    而 conv_out.bias 的梯度是"逐像素 dL/da 之和"，在 fp16 下会溢出成 inf，导致
    GradScaler 跳过整个 optimizer.step()（实测 it1 的 bias 梯度 = inf）。输出再 cast 回
    原 dtype，保证 a==0 时与 T0 逐位相同。
    """
    if gate is None:
        return outs
    out = list(outs)
    last = out[-1]
    R_hat = last[:, :3]
    T0 = last[:, 3:]
    dt = T0.dtype
    with torch.cuda.amp.autocast(enabled=False):
        T_new = gate(I.float(), T0.float())
    out[-1] = torch.cat([R_hat, T_new.to(dt)], dim=1)
    return out


@torch.no_grad()
def predict_with_gate(net, net_c, img, dev, gate: ResidualGateA1 | None = None):
    """与 scripts/eval_1019_multi.predict_rdnet 同口径（pad 到 32 的倍数、取最后一列 T、裁回原图），
    唯一差别 = 在 clamp 之前插入门。返回 float32 0..255 的 HxWx3（RGB）。"""
    x = torch.from_numpy(img.transpose(2, 0, 1) / 255.0)[None].to(dev)
    pad = (0, (-x.shape[-1]) % 32, 0, (-x.shape[-2]) % 32)
    xp = F.pad(x, pad, mode="reflect") if any(pad) else x
    code = net_c(xp)
    with torch.cuda.amp.autocast():
        _, out = net(xp, code, prompt=True)
    T = out[-1][:, 3:]
    if gate is not None:
        with torch.cuda.amp.autocast(enabled=False):
            T = gate(xp.float(), T.float())
    t = T[0].float().clamp(0, 1)
    H, W = img.shape[:2]
    t = t[:, :H, :W].permute(1, 2, 0).cpu().numpy() * 255.0
    return t


def gate_cfg_of(gate: ResidualGateA1 | None) -> dict | None:
    if gate is None:
        return None
    return dict(gate.cfg)


def param_report(cfg: dict | None = None) -> dict:
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    w, hid = int(c["width"]), int(c["hid"])
    n_in = int(c["in_ch"]) * w + w
    n_dw = w * 9 + w
    n_sca = hid * hid + hid
    n_out = hid + 1
    total = n_in + n_dw + n_sca + n_out
    mac_per_px = int(c["in_ch"]) * w + w * 9 + hid  # conv_in + dw + out（SCA 在 1x1 上，忽略）
    return dict(cfg=c, n_conv_in=n_in, n_dw=n_dw, n_sca=n_sca, n_out=n_out,
                n_total=total, mac_per_px=mac_per_px,
                gmac_at_320=mac_per_px * 320 * 320 / 1e9)


def save_gate_into_ckpt_obj(obj: dict, gate: ResidualGateA1 | None, cfg: dict | None = None):
    """给 ckpt 字典补 a1_gate_cfg / a1_gate_sd（不改变已有 key）。"""
    if gate is None:
        return obj
    obj["a1_gate_cfg"] = dict(cfg or gate.cfg)
    obj["a1_gate_sd"] = gate.state_dict()
    return obj


def load_gate_from_ckpt(ck: dict, device=None) -> ResidualGateA1 | None:
    if not isinstance(ck, dict) or "a1_gate_sd" not in ck:
        return None
    cfg = dict(ck.get("a1_gate_cfg") or DEFAULT_CFG)
    g = make_gate(cfg, device=device)
    g.load_state_dict(ck["a1_gate_sd"])
    g.eval()
    return g


if __name__ == "__main__":  # 简易自检：python3 rdnet_a1gate_1101.py
    rep = param_report()
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    g = make_gate()
    n = sum(p.numel() for p in g.parameters())
    assert n == 1025, n
    I = torch.rand(2, 3, 64, 64)
    T0 = torch.rand(2, 3, 64, 64)
    T = g(I, T0)
    print("init exact max|T-T0| =", float((T - T0).abs().max().detach()))
