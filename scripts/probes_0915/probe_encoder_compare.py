#!/usr/bin/env python3
"""raw 失败归因探针: 同口径区域池化特征在 raw vs taps 编码器上的跨域可分性。

对每个 ckpt: 提取 (逐图) [f_in, f_out, Δ, 比值, stats21]（用该模型自己的精修 m̂ 池化），
在训练场景上拟合线性探针（logreg），测 v2.1 / indomain 的档位 acc。
另报告 m̂ 池化覆盖率与 mask IoU，判定失败在"特征无信息"还是"头部没学会用"。

用法: python3 scripts/probes_0915/probe_encoder_compare.py --ckpt <path> --gpu 5 --n-train 200
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/DataDisk/zmy_code/TintedGlass_Enhance"
RDNET = os.path.join(ROOT, "RDNet")
sys.path.insert(0, RDNET)
os.chdir(RDNET)
from models.arch.rdnet_atap import FullNet_NLP_ATap_Fac  # noqa

LEVELS = ["1", "0.6", "0.3"]
TS = os.path.join(ROOT, "data/test_sets/synth_rrw_v21_0913")
TS_IND = os.path.join(ROOT, "data/test_sets/synth_rrw_v21_indomain")
TRAIN = os.path.join(ROOT, "data/train_sets/train_triplets_0913")


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32).transpose(2, 0, 1)[None] / 255.0


def build(ckpt, gpu):
    ck = torch.load(ckpt, map_location="cpu")
    a = ck.get("args", {}) or {}
    net = FullNet_NLP_ATap_Fac(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
                               num_subnet=4, loss_col=4, inject=True,
                               raw_encoder=a.get("encoder", "taps") == "raw",
                               region_level=a.get("region_level", False),
                               level_reg=a.get("level_reg", False))
    net.load_state_dict(ck["icnn"], strict=False)
    return net.to(f"cuda:{gpu}").eval(), a


@torch.no_grad()
def extract(net, x):
    if net.alpha_raw_enc is not None:
        feat = net._fac_feat(None, None, x)
    else:
        x_base, x_stem = net.baseball(x)
        x_stem = net.baseball_adapter[0](x_stem)
        c3b = net.baseball_adapter[4](x_base[3])
        feat = net._fac_feat(x_stem, c3b, x)
    zm = net.alpha_fac.m(feat)
    up = F.interpolate(zm, size=x.shape[-2:], mode="bilinear", align_corners=False)
    m = torch.sigmoid(up + net.alpha_refine(torch.cat([up, x], dim=1)))
    wm = F.adaptive_avg_pool2d(m, feat.shape[-2:])
    s_in = wm.sum((2, 3), keepdim=True) + 1e-6
    s_out = (1 - wm).sum((2, 3), keepdim=True) + 1e-6
    f_in = (feat * wm).sum((2, 3)) / s_in.flatten(1)
    f_out = (feat * (1 - wm)).sum((2, 3)) / s_out.flatten(1)
    stats = net._global_stats(x)
    return torch.cat([f_in, f_out, f_in - f_out,
                      torch.clamp(f_in / (f_out.abs() + 0.1), -8, 8), stats], 1)[0].cpu().numpy(), \
        float(wm.mean()), float(m.mean())


def collect(net, files, lvl):
    rows, covs = [], []
    for f in files:
        x = torch.from_numpy(load(f)).to(next(net.parameters()).device)
        v, cov, _ = extract(net, x)
        rows.append(v); covs.append(cov)
    return np.array(rows), np.array(covs), np.array([lvl] * len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=200)
    args = ap.parse_args()
    net, ck_a = build(args.ckpt, args.gpu)
    print(f"ckpt={os.path.basename(os.path.dirname(args.ckpt))} encoder={ck_a.get('encoder')} "
          f"region_level={ck_a.get('region_level')} level_reg={ck_a.get('level_reg')}")

    tr = sorted(glob.glob(os.path.join(TRAIN, "I", "*_l100.png")))
    tr = [tr[i] for i in np.random.default_rng(0).choice(len(tr), args.n_train, replace=False)]
    Xtr, _, ytr = [], [], []
    for lv, tag in zip(LEVELS, ("l100", "l060", "l030")):
        fs = sorted(glob.glob(os.path.join(TRAIN, "I", f"*_{tag}.png")))
        fs = [fs[i] for i in np.random.default_rng(1).choice(len(fs), args.n_train, replace=False)]
        X, C, y = collect(net, fs, LEVELS.index(lv))
        Xtr.append(X); ytr.append(y)
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=0.3).fit(Xtr, ytr)
    for name, base in (("v2.1", TS), ("indomain", TS_IND)):
        Xte, yte = [], []
        for lv in LEVELS:
            fs = sorted(glob.glob(os.path.join(base, "levels", f"alpha_{lv}", "*.png")))
            X, C, y = collect(net, fs, LEVELS.index(lv))
            Xte.append(X); yte.append(y)
        Xte, yte = np.concatenate(Xte), np.concatenate(yte)
        acc = float((clf.predict(Xte) == yte).mean())
        # 模型自身 p 的 acc（对照）
        print(f"  →{name:9s} 线性探针 acc={acc:.3f}   (n={len(yte)})")


if __name__ == "__main__":
    main()
