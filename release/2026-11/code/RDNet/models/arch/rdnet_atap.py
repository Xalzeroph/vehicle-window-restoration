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

    def _alpha_hat(self, x_stem, c3b, x_in, code=None):
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

    def forward_alpha_only(self, x_in, code=None):
        """阶段 1 用: 只跑到 α 头 (跳过 4 个 subnet 列), 供 L_α 单独训练。"""
        x_base, x_stem = self.baseball(x_in)
        x_stem = self.baseball_adapter[0](x_stem)
        c3b = self.baseball_adapter[4](x_base[3])
        return self._alpha_hat(x_stem, c3b, x_in, code)

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

        alpha = self._alpha_hat(x_stem, c3b, x_in, alpha_code)
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


class FullNet_NLP_ATap_Code(FullNet_NLP_ATap):
    """α 头 v2: classifier code 注入全局档位先验 + 低容量空间 offset。

    动机 (2026-09-13 饱和测试): 原 α 头在直接监督下饱和于 naked≈0.7、结构 corr≈0,
    诊断为其局部特征无法区分"裸区 α=1"与"膜区 α=0.3"的全局档位。code (6 维,
    KDNet 的 transmission-rate 预测) 正是全局档位信息, 故:
      z = 0.5·tanh(z_global(code)) + Δz_spatial(features)
      α̂ = α_min + (1−α_min)·σ(z)
    Δz 零初始化 → 起点即纯 code 全局先验; 空间头从零学起。
    """
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        fuse_ch = self.alpha_head[0].in_channels  # x_stem(64) + proj(c3)(512)
        self.alpha_global_head = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 1))
        self.alpha_film = nn.Linear(6, fuse_ch)

    def _alpha_hat(self, x_stem, c3b, x_in, code=None):
        if code is None:
            return super()._alpha_hat(x_stem, c3b, x_in)
        feat = self._fuse(self.alpha_c3, x_stem, c3b)
        dz = self.alpha_head(feat + self.alpha_film(code)[:, :, None, None])  # h/2
        up = F.interpolate(dz, size=x_in.shape[-2:], mode="bilinear", align_corners=False)
        dz = up + self.alpha_refine(torch.cat([up, x_in], dim=1))
        z = 0.5 * torch.tanh(self.alpha_global_head(code))[:, :, None, None] + dz
        return self.alpha_min + (1.0 - self.alpha_min) * torch.sigmoid(z)


def box_mean_channels(x, scales=(0.03, 0.10, 0.25)):
    """x (B,3,H,W) → (B,9,H,W): 3 尺度 box-mean (stride 1, 反射 padding)。"""
    outs = []
    H = x.shape[-2]
    for s in scales:
        k = max(3, int(round(H * s)) | 1)
        outs.append(F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, 1))
    return torch.cat(outs, dim=1)


class AlphaRawEncoder(nn.Module):
    """组件① raw 特征源: 独立轻量编码器 (不依赖 FocalNet)。

    输入 12ch (RGB + 3 尺度 box-mean), 4 级 stride-2 conv (base 32), 各级上采样到
    h/2 后拼接 → 128@h/2, 与 taps 配置输出同形状可直接换接。
    """
    def __init__(self, in_ch=12, base=32, out_ch=128):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, c[0], 3, 2, 1), nn.GELU(),
                                nn.Conv2d(c[0], c[0], 3, 1, 1), nn.GELU())
        self.b2 = nn.Sequential(nn.Conv2d(c[0], c[1], 3, 2, 1), nn.GELU(),
                                nn.Conv2d(c[1], c[1], 3, 1, 1), nn.GELU())
        self.b3 = nn.Sequential(nn.Conv2d(c[1], c[2], 3, 2, 1), nn.GELU(),
                                nn.Conv2d(c[2], c[2], 3, 1, 1), nn.GELU())
        self.b4 = nn.Sequential(nn.Conv2d(c[2], c[3], 3, 2, 1), nn.GELU(),
                                nn.Conv2d(c[3], c[3], 3, 1, 1), nn.GELU())
        self.out = nn.Conv2d(sum(c), out_ch, 3, 1, 1)

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        up = lambda t: F.interpolate(t, size=f1.shape[-2:], mode="bilinear",
                                     align_corners=False)
        return self.out(torch.cat([f1, up(f2), up(f3), up(f4)], dim=1))


class FullNet_NLP_ATap_Fac(FullNet_NLP_ATap):
    """组件① 分解式 α 头 (docs/2026.9.13_架构创新线实施方案 §2)。

    与生成器结构同构: α = 1 − m·(1 − c·g)
      z_m → m̂ = σ(z_m)  覆盖率掩膜, 可辨识部分, 复用 alpha_refine 边界精修
      z_g → ĝ = 1 + g_scale·tanh(z_g)  膜内缓变增益 (软界 ±g_scale)
      level: GAP+GMP ⊕ 21 维全局统计 → p∈Δ³ → ĉ = Σ p_k·ᾱ_k (档位后验)
    naked 锚定由构造保证 (m̂→0 ⇒ α̂→1); 参数前缀 alpha_fac.*。

    raw_encoder=True 时 α 相关特征完全来自 AlphaRawEncoder (不读 FocalNet),
    同一头 + 同一监督下与 taps 对比即"特征源假设"判决。
    """
    LEVELS = (1.0, 0.6, 0.3)

    # g_scale=1.0: 生成器 g = 1 + 0.3·f 中 f 是逐图单位方差场, 故 ±0.3 只是 ±1σ 带,
    # 实测支撑 [0.06, 1.96] (36% 像素出带); 软界取 ±1.0 才覆盖得住。
    def __init__(self, *args, g_scale=1.0, m_bias_init=1.8, raw_encoder=False,
                 region_level=False, level_reg=False, levels=None, **kw):
        super().__init__(*args, **kw)
        if levels is not None:
            self.LEVELS = tuple(float(x) for x in levels)
        self.region_level = region_level
        self.level_reg = level_reg and region_level   # 序数回归仅在区域池化头下启用
        fuse_ch = self.alpha_head[0].in_channels          # x_stem(64) + proj(c3)(64)
        self.alpha_fac = nn.Module()                      # 参数前缀统一 alpha_fac.*
        self.alpha_fac.m = nn.Sequential(
            nn.Conv2d(fuse_ch, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1))
        self.alpha_fac.g = nn.Sequential(
            nn.Conv2d(fuse_ch, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1))
        if self.region_level:
            # 档位头输入 = 标准化[stats21 ⊕ Δ(f_in−f_out) ⊕ 比值 ⊕ 物理低维(13)]
            # 用 BatchNorm1d(运行统计) 对齐探针的 per-feature StandardScaler 口径
            in_dim_rl = 21 + 2 * fuse_ch + 13
            self.alpha_fac.input_norm = nn.BatchNorm1d(in_dim_rl)
            out_dim = len(self.LEVELS) + (1 if self.level_reg else 0)
            self.alpha_fac.level = nn.Sequential(
                nn.Linear(in_dim_rl, 96), nn.GELU(), nn.Linear(96, out_dim))
        else:
            self.alpha_fac.level = nn.Sequential(
                nn.Linear(fuse_ch * 2 + 21, 64), nn.GELU(), nn.Linear(64, len(self.LEVELS)))
        self.alpha_raw_enc = AlphaRawEncoder() if raw_encoder else None
        self.g_scale = g_scale
        with torch.no_grad():       # 起点: m̂≈σ(1.8)=0.86, ĝ≡1, p 均匀 (ĉ=0.633)
            nn.init.zeros_(self.alpha_fac.m[-1].weight)
            self.alpha_fac.m[-1].bias.fill_(m_bias_init)
            nn.init.zeros_(self.alpha_fac.g[-1].weight)
            nn.init.zeros_(self.alpha_fac.g[-1].bias)
            nn.init.zeros_(self.alpha_fac.level[-1].weight)
            nn.init.zeros_(self.alpha_fac.level[-1].bias)
            if getattr(self, "level_reg", False):
                # 回归支起点 σ(z)=0.6（典型膜内 ᾱ）；softmax 支保持均匀 (ĉ=0.633)
                self.alpha_fac.level[-1].bias[len(self.LEVELS)] = 0.405
        self.register_buffer("alpha_levels", torch.tensor(self.LEVELS))

    @staticmethod
    def _global_stats(x):
        """21 维全局统计 (同探针 5): 每通道 mean/std/p10/p50/p90 + 色差均值×2
        + lum std / p95−p5 / <0.1 占比 / >0.9 占比。"""
        B = x.shape[0]
        v = x.float().reshape(B, 3, -1)                                  # B×3×N
        qs = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9], device=x.device), dim=2)
        lum = v.mean(1)                                                  # B×N
        lq = torch.quantile(lum, torch.tensor([0.05, 0.95], device=x.device), dim=1)
        return torch.cat([v.mean(2), v.std(2), *qs.unbind(0),
                          (v[:, 0] - v[:, 1]).mean(1, keepdim=True),
                          (v[:, 1] - v[:, 2]).mean(1, keepdim=True),
                          lum.std(1, keepdim=True), (lq[1] - lq[0]).unsqueeze(1),
                          (lum < 0.1).float().mean(1, keepdim=True),
                          (lum > 0.9).float().mean(1, keepdim=True)], dim=1)

    def _fac_feat(self, x_stem, c3b, x_in):
        if self.alpha_raw_enc is not None:
            return self.alpha_raw_enc(torch.cat([x_in, box_mean_channels(x_in)], dim=1))
        return self._fuse(self.alpha_c3, x_stem, c3b)

    def _fac_heads(self, feat, x_in):
        zm = self.alpha_fac.m(feat)                                      # h/2 logits
        up = F.interpolate(zm, size=x_in.shape[-2:], mode="bilinear", align_corners=False)
        m = torch.sigmoid(up + self.alpha_refine(torch.cat([up, x_in], dim=1)))
        g = 1.0 + self.g_scale * torch.tanh(F.interpolate(
            self.alpha_fac.g(feat), size=x_in.shape[-2:], mode="bilinear",
            align_corners=False))
        stats = self._global_stats(x_in)
        if not self.region_level:
            gap = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            gmp = F.adaptive_max_pool2d(feat, 1).flatten(1)
            out = self.alpha_fac.level(torch.cat([gap, gmp, stats], dim=1))
            p = torch.softmax(out, dim=1)
            c = (p * self.alpha_levels).sum(1).view(-1, 1, 1, 1)
        else:
            # m̂ 条件化双区域池化: 用 (detach 的) refine 后软掩膜把特征切成膜内/膜外两堆,
            # 显式给出相对量 (差/比值) + 亮度相对量与覆盖率 —— 档位的可迁移线索。
            with torch.no_grad():
                wm = F.adaptive_avg_pool2d(m, feat.shape[-2:])           # h/2 软权重(精修后 m̂)
            s_in = wm.sum(dim=(2, 3), keepdim=True) + 1e-6
            s_out = (1.0 - wm).sum(dim=(2, 3), keepdim=True) + 1e-6
            f_in = (feat * wm).sum(dim=(2, 3)) / s_in.flatten(1)
            f_out = (feat * (1.0 - wm)).sum(dim=(2, 3)) / s_out.flatten(1)
            delta = f_in - f_out
            ratio = torch.clamp(f_in / (f_out.abs() + 0.1), -8.0, 8.0)
            xh = F.adaptive_avg_pool2d(x_in, feat.shape[-2:])           # B×3×h/2×w/2
            b_in = (xh * wm).sum(dim=(2, 3)) / s_in.flatten(1)
            b_out = (xh * (1.0 - wm)).sum(dim=(2, 3)) / s_out.flatten(1)
            b_dif = b_in - b_out
            b_rat = torch.clamp(b_in / (b_out.abs() + 0.05), 0.0, 10.0)
            cov = wm.mean(dim=(2, 3))                                    # B×1 膜覆盖率
            phys = torch.cat([b_in, b_out, b_dif, b_rat, cov], dim=1)    # B×13
            out = self.alpha_fac.level(self.alpha_fac.input_norm(
                torch.cat([stats, delta, ratio, phys], dim=1)))
            if self.level_reg:
                p = torch.softmax(out[:, :len(self.LEVELS)], dim=1)
                c = torch.sigmoid(out[:, len(self.LEVELS):]).view(-1, 1, 1, 1)
            else:
                p = torch.softmax(out, dim=1)
                c = (p * self.alpha_levels).sum(1).view(-1, 1, 1, 1)
        self.alpha_fac.last_c = c        # 供训练损失读取实际使用的幅度标量
        alpha = (1.0 - m * (1.0 - c * g)).clamp(self.alpha_min, 1.0)
        return alpha, m, g, p

    def _alpha_hat(self, x_stem, c3b, x_in, code=None):
        return self._fac_heads(self._fac_feat(x_stem, c3b, x_in), x_in)[0]

    def forward_alpha_only(self, x_in, code=None):
        """返回 (α̂, m̂, ĝ, p); raw 配置下完全不走 backbone。"""
        if self.alpha_raw_enc is not None:
            return self._fac_heads(self._fac_feat(None, None, x_in), x_in)
        x_base, x_stem = self.baseball(x_in)
        x_stem = self.baseball_adapter[0](x_stem)
        c3b = self.baseball_adapter[4](x_base[3])
        return self._fac_heads(self._fac_feat(x_stem, c3b, x_in), x_in)


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
