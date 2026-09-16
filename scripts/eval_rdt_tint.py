#!/usr/bin/env python3
"""S1 分离器评测：rdt ckpt 输出 T̂ → 三测试集 × raw/gm PSNR/SSIM，含输入基线（ingm）。

用法:
  python3 scripts/eval_rdt_tint.py --ckpt runs/ckpt/rdt_freeze/ckpt_final.pth --tag rdt_freeze
  python3 scripts/eval_rdt_tint.py --ckpt pretrained/rdnet.pth --tag zeroshot   # 官方权重对照
输出:
  eval/eval_rdt_<tag>_<set>.csv（逐图） + _summary.json（档位均值）
  runs/pred/rdt_<tag>/<set>/<level>/<scene>/{t_hat,r_hat}.png
口径: PSNR/SSIM vs 全亮度 GT；gm = 对预测做最优全局乘性匹配后再算（eval_e1 同款）。
      R̂ 诊断列（r_raw/r_gm）: vs 测试集 r/（= I − α·T 的训练目标 R_src），仅作归因诊断。
"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, RDNET)
os.chdir(RDNET)
from models.arch.RDnet_ import FullNet_NLP              # noqa: E402
from models.arch.classifier import PretrainedConvNext  # noqa: E402

SETS = {
    "synth_rrw_v21_0913": "data/test_sets/synth_rrw_v21_0913",
    "synth_rrw_v21_indomain": "data/test_sets/synth_rrw_v21_indomain",
    "synth_rrw_v22_scan": "data/test_sets/synth_rrw_v22_scan",
}


def load_img(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def gain_match(q, g):
    """最优乘性匹配 q ← s·q（最小二乘），与 eval_e1.py 一致。"""
    q = np.asarray(q, np.float64)
    g = np.asarray(g, np.float64)
    s = float((q * g).sum() / (q * q).sum().clip(1e-8))
    return np.clip(q * s, 0, 255)


def build(ckpt_path, save_memory=True):
    net = FullNet_NLP(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2], num_subnet=4,
                      loss_col=4, num_classes=1000, drop_path=0, save_memory=save_memory,
                      inter_supv=True, head_init_scale=None, kernel_size=3)
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("icnn", ck)
    miss, unex = net.load_state_dict(sd, strict=False)
    print(f"[eval-rdt] ckpt={ckpt_path} missing={len(miss)} unexpected={len(unex)}"
          f" iter={ck.get('iter', '-')}")
    net_c = PretrainedConvNext("convnext_small_in22k")
    ck2 = torch.load(os.path.join(PROJ, "pretrained/cls_model.pth"), map_location="cpu")
    sd2 = ck2.get("icnn", ck2.get("model", ck2))
    net_c.load_state_dict(sd2, strict=False)
    net_c.eval().requires_grad_(False)
    return net, net_c


@torch.no_grad()
def predict(net, net_c, img, dev):
    """img: (H,W,3) float32 0..255 → (T̂, R̂) 同形状（通道序已核: [:, :3]=R̂, [:, 3:]=T̂）。"""
    x = torch.from_numpy(img.transpose(2, 0, 1) / 255.0)[None].to(dev)
    pad = tuple([0, (-x.shape[-1]) % 32, 0, (-x.shape[-2]) % 32])
    xp = F.pad(x, pad, mode="reflect") if any(pad) else x
    code = net_c(xp)          # fp32（训练脚本同款；autocast 下 ConvNeXt 会 Half/float 不匹配）
    with torch.cuda.amp.autocast():
        _, out = net(xp, code, prompt=True)
    t = out[-1][0, 3:].float().clamp(0, 1)
    r = out[-1][0, :3].float().clamp(0, 1)
    H, W = img.shape[:2]
    t = t[:, :H, :W].permute(1, 2, 0).cpu().numpy() * 255.0
    r = r[:, :H, :W].permute(1, 2, 0).cpu().numpy() * 255.0
    return t, r


def eval_set(net, net_c, set_name, root, dev, pred_root, save_pred):
    gt_dir = os.path.join(root, "gt")
    lv_dir = os.path.join(root, "levels")
    levels = sorted(os.listdir(lv_dir), key=lambda s: float(s.split("_")[-1]))
    scenes = sorted(os.listdir(gt_dir))
    rows, per_level = [], []
    print(f"[eval-rdt] {set_name}: {len(scenes)} 场景 × {len(levels)} 档")
    for lv in levels:
        acc = {k: [] for k in ("raw", "gm", "in_raw", "in_gm", "raw_s", "gm_s",
                               "r_raw", "r_gm")}
        for sc in scenes:
            gt = load_img(os.path.join(gt_dir, sc))
            r_gt = load_img(os.path.join(root, "r", sc))
            inp = load_img(os.path.join(lv_dir, lv, sc))
            t_hat, r_hat = predict(net, net_c, inp, dev)
            for tag, im, ref in (("raw", t_hat, gt), ("gm", gain_match(t_hat, gt), gt),
                                 ("in_raw", inp, gt), ("in_gm", gain_match(inp, gt), gt),
                                 ("r_raw", r_hat, r_gt), ("r_gm", gain_match(r_hat, r_gt), r_gt)):
                acc[tag].append(psnr(im, ref))
                if tag in ("raw", "gm"):
                    acc[tag + "_s"].append(ssim_fn(im, ref, channel_axis=2, data_range=255))
            rows.append(dict(level=lv, scene=sc, raw=round(acc["raw"][-1], 3),
                             gm=round(acc["gm"][-1], 3),
                             raw_ssim=round(acc["raw_s"][-1], 4),
                             gm_ssim=round(acc["gm_s"][-1], 4),
                             in_raw=round(acc["in_raw"][-1], 3),
                             in_gm=round(acc["in_gm"][-1], 3),
                             r_raw=round(acc["r_raw"][-1], 3),
                             r_gm=round(acc["r_gm"][-1], 3)))
            if save_pred:
                d = os.path.join(pred_root, set_name, lv, os.path.splitext(sc)[0])
                os.makedirs(d, exist_ok=True)
                Image.fromarray(t_hat.round().astype(np.uint8)).save(os.path.join(d, "t_hat.png"))
                Image.fromarray(r_hat.round().astype(np.uint8)).save(os.path.join(d, "r_hat.png"))
        rec = dict(level=lv, n=len(scenes),
                   raw=round(float(np.mean(acc["raw"])), 3),
                   gm=round(float(np.mean(acc["gm"])), 3),
                   raw_ssim=round(float(np.mean(acc["raw_s"])), 4),
                   gm_ssim=round(float(np.mean(acc["gm_s"])), 4),
                   in_raw=round(float(np.mean(acc["in_raw"])), 3),
                   in_gm=round(float(np.mean(acc["in_gm"])), 3),
                   r_raw=round(float(np.mean(acc["r_raw"])), 3),
                   r_gm=round(float(np.mean(acc["r_gm"])), 3))
        rec["gm_minus_in_gm"] = round(rec["gm"] - rec["in_gm"], 3)
        per_level.append(rec)
        print(f"  {lv:>12} n={rec['n']:2d}  raw {rec['raw']:6.2f}  gm {rec['gm']:6.2f}"
              f"  in_raw {rec['in_raw']:6.2f}  in_gm {rec['in_gm']:6.2f}"
              f"  gm−ingm {rec['gm_minus_in_gm']:+.2f}  ssim(raw/gm) {rec['raw_ssim']:.3f}/{rec['gm_ssim']:.3f}"
              f"  R̂ {rec['r_raw']:6.2f}/{rec['r_gm']:6.2f}")
    return rows, per_level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="synth_rrw_v21_0913,synth_rrw_v21_indomain,synth_rrw_v22_scan")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-save-pred", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(PROJ, "eval"))
    args = ap.parse_args()

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJ, args.ckpt)
    pred_root = os.path.join(PROJ, "runs/pred", f"rdt_{args.tag}")
    dev = torch.device(f"cuda:{args.gpu}")
    net, net_c = build(ckpt)
    net.to(dev).eval()
    net_c.to(dev)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    for s in args.sets.split(","):
        s = s.strip()
        rows, per_level = eval_set(net, net_c, s, os.path.join(PROJ, SETS[s]), dev,
                                   pred_root, not args.no_save_pred)
        csvp = os.path.join(args.out_dir, f"eval_rdt_{args.tag}_{s}.csv")
        with open(csvp, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with open(csvp.replace(".csv", "_summary.json"), "w") as fp:
            json.dump(dict(tag=args.tag, ckpt=os.path.relpath(ckpt, PROJ), set=s,
                           per_level=per_level), fp, ensure_ascii=False, indent=1)
        print(f"  -> {os.path.relpath(csvp, PROJ)}")
    print(f"[eval-rdt] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
