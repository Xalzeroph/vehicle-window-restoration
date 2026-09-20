#!/usr/bin/env python3
"""Mona-PEFT 适配器 → RDNet 的 FocalNet 主干（训练/评测共享）。

出处: Mona: Not All Weights Are Created Equal (CVPR 2025) 官方实现
      /tmp/mona_ref/Swin-Transformer-Classification/mmcls/models/backbones/swin_transformer_mona.py
      `MonaOp` / `Mona` 两个类逐行照搬语义（仅去掉 mmcv.BaseModule 依赖，BaseModule→nn.Module）。

插入点（对齐官方 SwinTransformerBlock.forward 里 my_module_1 / my_module_2 的位置：
都挂在**残差之后**，不在 norm 之前）:
    FocalModulationBlock.forward:
        x = shortcut + drop_path(gamma_1 * modulation(x))     ← mona1（调制残差之后）
        x = x + drop_path(gamma_2 * norm2(mlp(x)))            ← mona2（FFN 残差之后）
    mode="both" 两个都插；mode="ffn" 只插 mona2；mode="none" 不插。

硬约束（血泪坑）: patch 只用 `add_module("mona1"/"mona2")` 追加新子模块，**不改动、不重命名任何
已有子模块 key**，否则 rdnet.pth / focal.pth 里的 `baseball.*` 权重会静默加载不上。
-> `patch_focalnet` 必须在 `FullNet_NLP(...)` 构造**之后**调用（构造函数内部 strict 加载 focal.pth）。

用法:
    from rdnet_peft import patch_focalnet, mona_params, is_mona_name
    net = FullNet_NLP(...)          # 内部 strict 加载 ./pretrained/focal.pth
    patch_focalnet(net, "both")     # 再 patch
"""
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
if RDNET not in sys.path:
    sys.path.insert(0, RDNET)
from models.arch.focalnet import FocalModulationBlock   # noqa: E402

MONA_MODULE_NAMES = ("mona1", "mona2")
MONA_MODES = ("both", "ffn", "none")


# ---------------------------------------------------------------- 官方实现照搬
class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x


class Mona(nn.Module):
    """官方 `Mona`（BaseModule→nn.Module，语义一致；factor 参数官方未使用，保留签名）。"""

    def __init__(self, in_dim, factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(64)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2


def _init_mona(m):
    """对齐官方 `SwinTransformer_mona.init_weights()` 的 `_init_weights`：Linear→trunc_normal(0.02)+bias 0。
    （nn.LayerNorm 默认 weight=1/bias=0，与官方一致；MonaOp 的卷积官方也未额外初始化，保持 PyTorch 默认。）"""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------- FocalNet 侧包装
class MonaFocalModulationBlock(FocalModulationBlock):
    """`FocalModulationBlock` 的 mona 版 forward（子类，故 state_dict key 布局与原类完全相同）。

    不重写 `__init__`：mona1/mona2 由 `patch_block` 用 `add_module` 动态挂上，
    forward 里按"挂没挂"决定插不插（= mode 语义）。

    显存：mona 可训 = 冻结的 FocalNet 前向也进 autograd 图（梯度要穿过它回到上游 mona）→
    focal modulation 的 unfold 中间量（B,C,H,W,k²）必须留到反向。实测 224²bs8 直接 OOM，
    bs4 就要 8.7G。故默认对每个 block 做 gradient checkpointing（重算换显存，数学等价；
    use_reentrant=False 自持 RNG，Mona 的 dropout 掩码在重算时一致）。eval（no_grad）不触发。
    """

    def _forward_mona(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature (由 BasicLayer 写入 self.H/self.W)。
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        if not self.use_postln:
            x = self.norm1(x)
        x = x.view(B, H, W, C)

        # FM
        x = self.modulation(x).view(B, H * W, C)
        if self.use_postln:
            x = self.norm1(x)

        # 调制残差
        x = shortcut + self.drop_path(self.gamma_1 * x)
        mona1 = getattr(self, "mona1", None)
        if mona1 is not None:
            x = mona1(x, (H, W))

        # FFN
        if self.use_postln:
            x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        mona2 = getattr(self, "mona2", None)
        if mona2 is not None:
            x = mona2(x, (H, W))

        return x

    def forward(self, x):
        if getattr(self, "_mona_ckpt", False) and torch.is_grad_enabled():
            return cp.checkpoint(self._forward_mona, x, use_reentrant=False)
        return self._forward_mona(x)


def is_mona_name(name):
    """name（参数名或模块名）是否属于 mona 适配器。"""
    return any(seg in MONA_MODULE_NAMES for seg in name.split("."))


def patch_block(blk, mode="both", grad_ckpt=True):
    """给单个 FocalModulationBlock 挂 mona1/mona2（mode: both/ffn/none）。返回挂上的模块数。"""
    assert mode in MONA_MODES, f"未知 mode: {mode}"
    n = 0
    if mode in ("both",):
        blk.add_module("mona1", Mona(blk.dim))
        n += 1
    if mode in ("both", "ffn"):
        blk.add_module("mona2", Mona(blk.dim))
        n += 1
    if n == 0:
        return 0
    for name in MONA_MODULE_NAMES:
        m = getattr(blk, name, None)
        if m is not None:
            m.apply(_init_mona)
    blk._mona_mode = mode
    blk._mona_ckpt = bool(grad_ckpt)
    # 类换成 mona 版 forward（同一实例布局，不动 _modules/_parameters -> 已有 key 不变）
    try:
        blk.__class__ = MonaFocalModulationBlock
    except TypeError:      # pragma: no cover - 仅当类布局不允许换类时兜底
        blk.forward = types.MethodType(MonaFocalModulationBlock.forward, blk)
    return n


def patch_focalnet(net, mode="both", grad_ckpt=True, verbose=False):
    """把 net 里所有 FocalModulationBlock 换成 mona 版。返回被 patch 的 block 数。

    必须先 `FullNet_NLP(...)` 构造完（内部 strict 加载 focal.pth）再调用。
    grad_ckpt=True 时训练（grad 开启）对每个 block 重算换显存（数学等价，见类 docstring）。
    """
    if mode == "none":
        return 0
    blocks = [m for m in net.modules() if isinstance(m, FocalModulationBlock)]
    n_mod = 0
    for blk in blocks:
        n_mod += patch_block(blk, mode, grad_ckpt)
    if verbose:
        print(f"[peft] patch_focalnet mode={mode} grad_ckpt={grad_ckpt}: "
              f"{len(blocks)} 个 block, 挂上 {n_mod} 个 Mona")
    return len(blocks)


# ---------------------------------------------------------------- 查询 / 冻结工具
def mona_params(net):
    """net 中所有 mona 参数 (name, param)，按名字序（供冻结/解冻、断言的定序来源）。"""
    return [(n, p) for n, p in net.named_parameters() if is_mona_name(n)]


def mona_param_names(net):
    """mona 参数名列表。"""
    return [n for n, _ in mona_params(net)]


def mona_state_dict_keys(sd):
    """state_dict 里的 mona key 列表（评测端用它确认 ckpt 带没带 mona 权重）。"""
    return [k for k in sd if is_mona_name(k)]


def set_mona_trainable(net, flag=True):
    """显式把 mona 参数设为 requires_grad=flag（冻结 baseball.* 后必须再调一次 True）。返回参数个数。"""
    ps = mona_params(net)
    for _, p in ps:
        p.requires_grad_(flag)
    return len(ps)
