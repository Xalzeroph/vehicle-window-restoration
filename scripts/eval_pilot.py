#!/usr/bin/env python3
"""Pilot 评测: 方案 A / 方案 B / M0(zero-shot) 在基准 v2.1 测试集上对照。

- arm A: αTAP ckpt, 推理 β=1 (T̂ = (I−R̂+Δ̂)/α̂)
- arm B: inject=False ckpt (官方参数化)
- M0:   官方 rdnet.pth (无 α 头)
指标: raw PSNR/SSIM + 增益匹配(gm) PSNR/SSIM (口径同 eval_e1.py) + α̂ MAE 分桶
(边界带 |∇α|>0.02 / 膜内 / 裸区 α≥0.95)。

用法:
  python3 scripts/eval_pilot.py --arm A --ckpt pilot_ckpt/A/ckpt_final.pth --gpu 0
  python3 scripts/eval_pilot.py --arm M0 --gpu 0
输出: pilot_pred/{arm}/alpha_X/{stem}{,_r}.png + {stem}_a.npy; pilot_eval_{arm}.csv
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, RDNET)
os.chdir(RDNET)

from models.arch.RDnet_ import FullNet_NLP  # noqa: E402
from models.arch.rdnet_atap import FullNet_NLP_ATap, load_official_atap  # noqa: E402
from models.arch.classifier import PretrainedConvNext  # noqa: E402
from skimage.metrics import structural_similarity as ssim_fn  # noqa: E402

RDNET_CKPT = os.path.join(PROJ, "pretrained", "rdnet.pth")


def load(f):
    return np.asarray(Image.open(f).convert("RGB"), np.float32)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def gain_match(q, g):
    q = np.asarray(q, np.float64)
    g = np.asarray(g, np.float64)
    scale = float((q * g).sum() / (q * q).sum())
    return np.clip(q * scale, 0, 255)


def grad_mag(a):
    gx = np.abs(np.diff(a, axis=1, prepend=a[:, :1]))
    gy = np.abs(np.diff(a, axis=0, prepend=a[:1, :]))
    return gx + gy


def build(arm, ckpt, device):
    if arm == "M0":
        net = FullNet_NLP([64, 128, 256, 512], [2, 2, 4, 2], 4, 4)
        net.load_state_dict(torch.load(RDNET_CKPT, map_location="cpu")["icnn"])
    else:
        net = FullNet_NLP_ATap([64, 128, 256, 512], [2, 2, 4, 2], 4, 4,
                               inject=(arm == "A"))
        if ckpt:
            ck = torch.load(ckpt, map_location="cpu")["icnn"]
            miss, unex = net.load_state_dict(ck, strict=False)
            print(f"[{arm}] load {ckpt}: missing={len(miss)} unexpected={len(unex)}")
        else:
            load_official_atap(net, RDNET_CKPT, verbose=False)
    net.eval().to(device)
    net_c = PretrainedConvNext("convnext_small_in22k").to(device)
    net_c.load_state_dict(torch.load(os.path.join(RDNET, "pretrained/cls_model.pth"),
                                     map_location="cpu")["icnn"])
    net_c.eval()
    return net, net_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "M0"], required=True)
    ap.add_argument("--ckpt", default=None, help="默认 pilot_ckpt/{arm}/ckpt_final.pth (M0 不需要)")
    ap.add_argument("--ts", default=os.path.join(PROJ, "synth_rrw_v21_0913"))
    ap.add_argument("--levels", default="1.0,0.6,0.3")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-root", default=os.path.join(PROJ, "pilot_pred"))
    ap.add_argument("--tag", default="", help="输出后缀: {out_root}{tag}/ 与 pilot_eval_{arm}{tag}.csv")
    args = ap.parse_args()
    if args.ckpt is None and args.arm != "M0":
        args.ckpt = os.path.join(PROJ, "pilot_ckpt", args.arm, "ckpt_final.pth")

    device = torch.device(f"cuda:{args.gpu}")
    net, net_c = build(args.arm, args.ckpt, device)
    levels = [float(x) for x in args.levels.split(",")]
    scenes = sorted(glob.glob(os.path.join(args.ts, "gt", "*.png")))

    rows = []
    for lv in levels:
        outd = os.path.join(args.out_root + args.tag, args.arm, f"alpha_{lv:g}")
        os.makedirs(outd, exist_ok=True)
        for gt_f in scenes:
            stem = os.path.splitext(os.path.basename(gt_f))[0]
            i_f = os.path.join(args.ts, "levels", f"alpha_{lv:g}", stem + ".png")
            i = load(i_f)
            x = torch.from_numpy(i.transpose(2, 0, 1) / 255.0)[None].to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                code = net_c(x)
                if isinstance(net, FullNet_NLP_ATap):
                    cols, aux = net(x, code, beta=1.0)
                else:
                    _, cols = net(x, code, prompt=True)
                    aux = {}
            t_hat = cols[-1][:, 3:].float().clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255
            r_hat = cols[-1][:, :3].float().clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255
            Image.fromarray(t_hat.round().astype(np.uint8)).save(os.path.join(outd, stem + ".png"))
            Image.fromarray(r_hat.round().astype(np.uint8)).save(os.path.join(outd, stem + "_r.png"))
            g = load(gt_f)
            rec = dict(level=lv, scene=stem,
                       psnr=psnr(t_hat, g), ssim=ssim_fn(t_hat, g, channel_axis=2, data_range=255),
                       t_mean=t_hat.mean(), gt_mean=g.mean())
            t_gm = gain_match(t_hat, g)
            rec["psnr_gm"] = psnr(t_gm, g)
            rec["ssim_gm"] = ssim_fn(t_gm, g, channel_axis=2, data_range=255)
            inp_gm = gain_match(i, g)
            rec["psnr_in_gm"] = psnr(inp_gm, g)
            rec["ssim_in_gm"] = ssim_fn(inp_gm, g, channel_axis=2, data_range=255)
            if "alpha" in aux and aux["alpha"] is not None:
                ah = aux["alpha"].float()[0, 0].cpu().numpy()
                np.save(os.path.join(outd, stem + "_a.npy"), ah.astype(np.float16))
                ag = np.load(os.path.join(args.ts, "alpha", f"{stem}_a{lv:g}.npy")).astype(np.float32)
                ae = np.abs(ah - ag)
                band = grad_mag(ag) > 0.02
                naked = ag >= 0.95
                film = ~(band | naked)
                rec.update(a_mae=float(ae.mean()),
                           a_mae_band=float(ae[band].mean()) if band.any() else np.nan,
                           a_mae_film=float(ae[film].mean()) if film.any() else np.nan,
                           a_mae_naked=float(ae[naked].mean()) if naked.any() else np.nan,
                           a_frac_band=float(band.mean()))
            rows.append(rec)
            print(f"[{args.arm}] lv={lv:g} {stem}: PSNR {rec['psnr']:.2f} (gm {rec['psnr_gm']:.2f}) "
                  f"SSIM {rec['ssim']:.3f} T̄={rec['t_mean']:.0f} GT̄={rec['gt_mean']:.0f}"
                  + (f" aMAE={rec.get('a_mae', np.nan):.3f}" if "a_mae" in rec else ""))

    csv = os.path.join(PROJ, f"pilot_eval_{args.arm}{args.tag}.csv")
    keys = ["level", "scene", "psnr", "ssim", "psnr_gm", "ssim_gm", "psnr_in_gm", "ssim_in_gm",
            "t_mean", "gt_mean", "a_mae", "a_mae_band", "a_mae_film", "a_mae_naked", "a_frac_band"]
    with open(csv, "w") as fp:
        fp.write(",".join(keys) + "\n")
        for r in rows:
            fp.write(",".join(("" if r.get(k) is None else
                               (f"{r[k]:.5f}" if isinstance(r.get(k), float) else str(r.get(k, ""))))
                              for k in keys) + "\n")
    print(f"\n== {args.arm} 汇总 (12 场景均值) ==")
    for lv in levels:
        sel = [r for r in rows if r["level"] == lv]
        m = lambda k: np.mean([r[k] for r in sel if k in r and r[k] is not None])
        print(f"  a={lv:g}: raw PSNR {m('psnr'):6.2f} SSIM {m('ssim'):.4f} | "
              f"gm {m('psnr_gm'):6.2f}/{m('ssim_gm'):.4f} | in_gm {m('psnr_in_gm'):6.2f} | "
              f"T̄ {m('t_mean'):5.1f} vs GT {m('gt_mean'):5.1f}"
              + (f" | αMAE {m('a_mae'):.3f} (band {m('a_mae_band'):.3f}/film {m('a_mae_film'):.3f}/naked {m('a_mae_naked'):.3f})"
                 if any('a_mae' in r for r in sel) else ""))
    print("csv ->", csv)


if __name__ == "__main__":
    main()
