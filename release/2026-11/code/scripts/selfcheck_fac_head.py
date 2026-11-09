#!/usr/bin/env python3
"""S1 前置自检 (方案文档 §2.3): 分解式 α 头形状 / 参数量 / 初始值 / 三连图组装。

用法:
  python3 scripts/selfcheck_fac_head.py                      # taps + raw 两配置
  python3 scripts/selfcheck_fac_head.py --config taps
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, RDNET)
os.chdir(RDNET)

from models.arch.rdnet_atap import FullNet_NLP_ATap, FullNet_NLP_ATap_Fac  # noqa: E402

TRIPLETS = os.path.join(PROJ, "data/train_sets/train_triplets_0913")
LEVEL_TAGS = ("l100", "l060", "l030")


def _img(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0


def load_scene(scene_id):
    I = torch.from_numpy(np.stack([_img(os.path.join(TRIPLETS, "I", f"{scene_id}_{t}.png"))
                                   for t in LEVEL_TAGS]))                       # 3×3×H×W
    A = torch.from_numpy(np.stack([
        np.load(os.path.join(TRIPLETS, "alpha", f"{scene_id}_{t}.npy")).astype(np.float32)[None]
        for t in LEVEL_TAGS]))                                                 # 3×1×H×W
    T = torch.from_numpy(_img(os.path.join(TRIPLETS, "T", f"{scene_id}.png")))
    R = torch.from_numpy(_img(os.path.join(TRIPLETS, "R", f"{scene_id}.png")))
    m = torch.from_numpy(np.load(os.path.join(TRIPLETS, "m", f"{scene_id}.npy")).astype(np.float32))
    g = torch.from_numpy(np.load(os.path.join(TRIPLETS, "g", f"{scene_id}.npy")).astype(np.float32))
    return dict(I=I, A=A, T=T, R=R, m=m, g=g)


def build(cfg):
    kw = dict(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
              num_subnet=4, loss_col=4, inject=True)
    net = FullNet_NLP_ATap_Fac(raw_encoder=(cfg == "raw"), **kw)
    base = FullNet_NLP_ATap(**kw)
    return net.eval(), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["taps", "raw", "both"], default="both")
    ap.add_argument("--scenes", default="s0000,s0001")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    cfgs = ["taps", "raw"] if args.config == "both" else [args.config]
    meta = json.load(open(os.path.join(TRIPLETS, "meta.json")))
    ae_by = {(s["scene_id"], s["level"]): s["alpha_eff"] for s in meta["samples"]}
    sids = args.scenes.split(",")

    for cfg in cfgs:
        net, base = build(cfg)
        net = net.to(device)
        n_fac = sum(p.numel() for n, p in net.named_parameters()
                    if n.startswith("alpha_fac.") or n.startswith("alpha_raw_enc."))
        n_base = sum(p.numel() for p in base.parameters())
        n_all = sum(p.numel() for p in net.parameters())
        print(f"\n=== config={cfg} ===")
        print(f"[a] 参数量: 新增(alpha_fac⁺+raw_enc) {n_fac/1e3:.1f}K  "
              f"v1 基线 {n_base/1e6:.3f}M  fac 全模型 {n_all/1e6:.3f}M")

        scenes = [load_scene(s) for s in sids]
        for s, sc in zip(sids, scenes):                 # [b] 档序核验 (tag ↔ level)
            for j, tag in enumerate(LEVEL_TAGS):
                lv = float(FullNet_NLP_ATap_Fac.LEVELS[j])
                assert abs(sc["A"][j].mean().item() - ae_by[(s, lv)]) < 2e-3, \
                    f"{s} {tag} 档序错: {sc['A'][j].mean():.4f} vs meta {ae_by[(s, lv)]:.4f}"
        print(f"[b] 档序核验 ✓: 同场景 3 档 = {list(FullNet_NLP_ATap_Fac.LEVELS)} "
              f"(scene-major × level-minor, batch = k 场景 × 3 档); "
              f"三档共享 T/R/m/g (场景级落盘)")

        I = torch.cat([sc["I"] for sc in scenes], 0).to(device)     # (k*3,3,H,W)
        A = torch.cat([sc["A"] for sc in scenes], 0).to(device)

        with torch.no_grad():
            alpha, m, g, p = net.forward_alpha_only(I)
        assert alpha.shape == A.shape and alpha.shape[-2:] == I.shape[-2:], \
            (alpha.shape, A.shape, I.shape)
        assert abs(p.sum(1).mean().item() - 1.0) < 1e-4
        print(f"[c] 形状: α̂/m̂/ĝ {tuple(m.shape)} (full-res ✓)  p {tuple(p.shape)} "
              f"行和∈[{p.sum(1).min():.4f},{p.sum(1).max():.4f}]  "
              f"α̂∈[{alpha.min():.3f},{alpha.max():.3f}]")

        a_mean = alpha.mean().item()
        naked = (A >= 0.95)
        naked_mean = alpha[naked].mean().item() if naked.any() else float("nan")
        c_hat = (p * net.alpha_levels).sum(1)
        consist = (alpha - (1 - m * (1 - c_hat[:, None, None, None] * g))).abs().max().item()
        print(f"[d] 初始 α̂ 均值 {a_mean:.3f} (目标 0.5-0.7); "
              f"naked(α_GT≥0.95)区 {naked_mean:.3f}")
        print(f"    初始 m̂ {m.mean():.3f}  ĝ {g.mean():.3f}  ĉ {c_hat.mean():.3f}; "
              f"构造一致性 max|α̂−(1−m̂(1−ĉĝ))|={consist:.1e}")
        gt_m = (A < 0.95).float()
        print(f"[e] m_lab=1[α_GT<0.95] 覆盖率 {gt_m.mean():.3f}; "
              f"m̂ 初始 BCE {torch.nn.functional.binary_cross_entropy(m.clamp(1e-6, 1-1e-6), gt_m).item():.3f}")


if __name__ == "__main__":
    main()
