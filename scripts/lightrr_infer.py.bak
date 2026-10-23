#!/usr/bin/env python3
"""LightRR 推理封装 + 尺寸鲁棒性判决（2026-10-12）

发现的问题
----------
`AFM_LowFrequencySSM.forward` 结尾有
    y = self.idwt(torch.cat([y_ll, x_lh, x_hl, x_hh], dim=1))
    assert yh == H ; assert yw == W
而 `IDWT_2D` 在任一维为**奇数**时会**静默丢掉一行/一列**：
    idwt(1,256,48,67) -> (1,256,96,134)      # 67 -> 134，丢了一列
于是断言失败。**H、W 必须都是偶数**（且逐级下采样后仍需偶数 → 实际要求 **32 的倍数**）。

对本课题的直接后果
------------------
公共基准与自采数据的尺寸**大多不是 32 的倍数**：
    real20 1114x717 / 1126x740 / 678x517 …
    SIR2   540x400
直接喂进去会崩。LightRR 自己的评测用 `--if_align` 把图 resize 到 `h//32*32, w//32*32`
来规避——**这等于承认了该约束，但用"裁掉几条像素"来绕过**（会改变图片内容，
且与其它基线方法的口径不一致：别人不裁）。

本模块的选择
------------
**pad 到 32 的倍数 → 推理 → 裁回原尺寸**。理由：
  - 不改变原图内容（只加边），与其它基线的输入口径一致；
  - 输出尺寸恒等于输入尺寸，可与 RDNet 等逐像素对齐比较；
  - 代价是边缘多算了一圈，对 224 的倍数情形完全等价（无 pad）。
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

REPO = "/home/zmy/mycode/LightRR-CVPR26"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MULT = 32


def pad_to_multiple(x: torch.Tensor, mult: int = MULT, mode: str = "reflect"):
    """把 (B,C,H,W) 的 H、W 补到 mult 的倍数，返回 (padded, (H,W))。"""
    B, C, H, W = x.shape
    ph = (mult - H % mult) % mult
    pw = (mult - W % mult) % mult
    if ph == 0 and pw == 0:
        return x, (H, W)
    # reflect 需要 pad < 边长，否则退回 replicate
    if ph >= H or pw >= W:
        mode = "replicate"
    xp = F.pad(x, (0, pw, 0, ph), mode=mode)
    return xp, (H, W)


class LightRRInfer:
    """LightRR_9M 的推理封装：自动 pad/裁剪，输出与输入同尺寸。"""

    def __init__(self, ckpt: str, device="cuda", repo: str = REPO):
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from models.arch import LightRR_9M
        self.device = torch.device(device)
        self.net = LightRR_9M().to(self.device).eval()
        ck = torch.load(ckpt, map_location="cpu")
        sd = ck["weight"] if isinstance(ck, dict) and "weight" in ck else ck
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        self.ckpt_info = {
            "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
            "missing": len(missing), "unexpected": len(unexpected),
        }
        self.n_params = sum(p.numel() for p in self.net.parameters())

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) in [0,1]. 返回 (B,3,H,W)。"""
        x = x.to(self.device).float()
        H, W = x.shape[-2:]
        xp, _ = pad_to_multiple(x)
        outs, _ = self.net(xp)
        y = outs[-1]
        return y[..., :H, :W].clamp(0, 1)

    @torch.no_grad()
    def forward_multiscale(self, x: torch.Tensor):
        """返回 4 个尺度的输出（已裁回各自尺度），用于分析。"""
        x = x.to(self.device).float()
        H, W = x.shape[-2:]
        xp, _ = pad_to_multiple(x)
        outs, _ = self.net(xp)
        res = []
        for o in outs:
            h, w = o.shape[-2:]
            # 各尺度相对输入的缩放
            sh, sw = int(round(H * h / xp.shape[-2])), int(round(W * w / xp.shape[-1]))
            res.append(o[..., :sh, :sw].clamp(0, 1))
        return res


def main() -> int:
    ckpt = f"{REPO}/weights/lightrr/lightrr_woNature.pt"
    m = LightRRInfer(ckpt, device="cuda")
    print(f"[ckpt] {m.ckpt_info}  params={m.n_params/1e6:.2f} M")

    sizes = [(224, 224), (384, 384), (540, 400), (400, 540), (717, 1114),
             (740, 1126), (517, 678), (135, 96), (100, 100), (1080, 1920),
             (255, 257), (1, 33)]
    print(f"\n{'H':>6}{'W':>6}{'pad to':>12}  result")
    print("-" * 74)
    n_ok = 0
    for (H, W) in sizes:
        x = torch.rand(1, 3, H, W, device="cuda")
        ph = (MULT - H % MULT) % MULT
        pw = (MULT - W % MULT) % MULT
        try:
            y = m(x)
            assert y.shape == x.shape, f"shape {tuple(y.shape)} != {tuple(x.shape)}"
            print(f"{H:>6}{W:>6}{f'+{ph},+{pw}':>12}  OK  out={tuple(y.shape)}")
            n_ok += 1
        except torch.cuda.OutOfMemoryError:
            print(f"{H:>6}{W:>6}{f'+{ph},+{pw}':>12}  OOM (skip)")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{H:>6}{W:>6}{f'+{ph},+{pw}':>12}  {type(e).__name__}: {str(e)[:44]}")
    print(f"\n{n_ok}/{len(sizes)} sizes OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
