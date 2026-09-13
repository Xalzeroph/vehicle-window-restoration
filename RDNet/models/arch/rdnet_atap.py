"""αTAP-RDNet: 空间 α 头 + P3 硬减法 + 除法出口 (pilot 版)。

按 docs/2026.9.13_αTAP方案评审与数学落地_opus.md 修正清单实现:
  - α 头: x_stem(h/4, 64ch) + FocalNet c3(h/32) 融合 → h/4 logits → 以 I 为导引的
    全分辨率细化(残差式); α̂ = α_min + (1−α_min)·σ(z)
  - P3 硬减法: T0̂ = I − R̂ + Δ̂ (Δ̂ 零初始化, h/4 → 全分辨率)
  - ramp 出口(必须在 forward): T̂ = T0̂·(1 − β + β/α̂)
  - prompt 拼接: 原 6 维 classifier code + α̂ 统计 3 维; 新列零初始化
  - inject=False 时为方案 B 配置: α 头/Δ̂ 不注入网络, 输出保持官方参数化,
    α̂ 仅供 L_α 监督与 MAE 报告(即"独立小 α 头"对照)

载入官方 icnn 权重: load_official_atap(model, ckpt) 处理 prompt 首层加宽
(取官方 6 列 + 新列置零) 与新增 key (strict=False)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arch.RDnet_ import FullNet_NLP
from models.arch.modules_sig import LayerNorm

ALPHA_MIN = 0.05
PROMPT_EXTRA = 3


class FullNet_NLP_ATap(FullNet_NLP):
    def __init__(self, channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
                 num_subnet=4, loss_col=4, inject=True, alpha_min=ALPHA_MIN, **kw):
        # base __init__ 会原地 reverse channels, 先取出原始序的通道数
        stem_ch, c3_ch = channels[0], channels[3]
        ch = list(channels)
        super().__init__(ch, layers, num_subnet, loss_col, **kw)
        self.inject = inject
        self.alpha_min = alpha_min
        self.alpha_c3 = nn.Conv2d(c3_ch, stem_ch, 1)
        self.alpha_head = nn.Sequential(
            nn.Conv2d(2 * stem_ch, stem_ch, 3, padding=1),
            LayerNorm(stem_ch, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(stem_ch, 1, 1))
        self.alpha_refine = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        if inject:
            self.delta_c3 = nn.Conv2d(c3_ch, stem_ch, 1)
            self.delta_head = nn.Sequential(
                nn.Conv2d(2 * stem_ch, stem_ch, 3, padding=1), nn.GELU(),
                nn.Conv2d(stem_ch, 3, 1))
            new = nn.Linear(6 + PROMPT_EXTRA, 512)
            with torch.no_grad():
                new.weight.zero_()
                new.bias.zero_()
            self.prompt[0] = new
        for m in [self.alpha_head[-1], self.alpha_refine[-1]] + (
                [self.delta_head[-1]] if inject else []):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _fuse(self, proj, x_stem, c3b):
        """x_stem(h/2, 64ch) + proj(c3)(h/16 → 上采样 h/2) 融合。"""
        c3u = F.interpolate(proj(c3b), size=x_stem.shape[-2:], mode="bilinear",
                            align_corners=False)
        return torch.cat([x_stem, c3u], dim=1)

    def _alpha_hat(self, x_stem, c3b, x_in):
        logits = self.alpha_head(self._fuse(self.alpha_c3, x_stem, c3b))  # h/2
        up = F.interpolate(logits, size=x_in.shape[-2:], mode="bilinear",
                           align_corners=False)
        z = up + self.alpha_refine(torch.cat([up, x_in], dim=1))
        alpha = self.alpha_min + (1.0 - self.alpha_min) * torch.sigmoid(z)
        return alpha

    def _alpha_stats(self, alpha):
        a = alpha.float()
        return torch.stack([a.mean((1, 2, 3)), a.std((1, 2, 3)),
                            (a < 0.5).float().mean((1, 2, 3))], dim=-1)

    def forward_alpha_only(self, x_in):
        """阶段 1 用: 只跑到 α 头 (跳过 4 个 subnet 列), 供 L_α 单独训练。"""
        x_base, x_stem = self.baseball(x_in)
        x_stem = self.baseball_adapter[0](x_stem)
        c3b = self.baseball_adapter[4](x_base[3])
        return self._alpha_hat(x_stem, c3b, x_in)

    def _delta(self, x_stem, c3b, x_in):
        d = self.delta_head(self._fuse(self.delta_c3, x_stem, c3b))  # h/2
        return F.interpolate(d, size=x_in.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, x_in, alpha_code, beta=0.0, prompt=True):
        x_img_out = []
        c0, c1, c2, c3 = 0, 0, 0, 0

        x_base, x_stem = self.baseball(x_in)
        x_stem = self.baseball_adapter[0](x_stem)
        c0b, c1b, c2b, c3b = (self.baseball_adapter[1](x_base[0]),
                              self.baseball_adapter[2](x_base[1]),
                              self.baseball_adapter[3](x_base[2]),
                              self.baseball_adapter[4](x_base[3]))
        c0, c1, c2, c3 = c0b, c1b, c2b, c3b

        alpha = self._alpha_hat(x_stem, c3b, x_in)
        stats = self._alpha_stats(alpha)
        delta = self._delta(x_stem, c3b, x_in) if self.inject else None

        if prompt:
            inp = torch.cat([alpha_code, stats], dim=-1) if self.inject else alpha_code
            prompt_alpha = self.prompt(inp).unsqueeze(-1).unsqueeze(-1)
            x = prompt_alpha * x_stem
        else:
            x = x_stem

        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f"subnet{str(i)}")(x, c0, c1, c2, c3)
            if i > (self.num_subnet - self.Loss_col):
                out = torch.cat([x_in, x_in], dim=-3) - \
                    self.decoder_blocks[-1](c3, c2, c1, c0)
                if self.inject:
                    r_hat = out[:, :3]
                    t0 = x_in - r_hat + delta
                    t_hat = t0 * (1.0 - beta + beta / alpha)
                    out = torch.cat([r_hat, t_hat], dim=-3)
                x_img_out.append(out)

        return x_img_out, {"alpha": alpha, "delta": delta, "stats": stats}


def load_official_atap(model, ckpt_path, verbose=True):
    """官方 rdnet.pth['icnn'] → αTAP 模型。prompt.0 加宽: 官方 6 列 + 新列零初始化。"""
    sd = torch.load(ckpt_path, map_location="cpu")["icnn"]
    if model.inject:
        old_w = sd.pop("prompt.0.weight")
        old_b = sd.pop("prompt.0.bias")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if model.inject:
        with torch.no_grad():
            p = model.prompt[0]
            p.weight[:, :6].copy_(old_w)
            p.bias.copy_(old_b)
    if verbose:
        print(f"[atap] load {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
        for k in missing[:8]:
            print("  missing:", k)
        for k in unexpected[:8]:
            print("  unexpected:", k)
    return missing, unexpected
