#!/usr/bin/env python3
"""S1 组件① 分解式 α 头训练 (三连图, 只训头)。

协议 (docs/2026.9.13_架构创新线实施方案_可辨识性对齐恢复.md §2.5 + 附录 A):
  - 数据: train_triplets_0913 (800 场景 × 3 档, 同场景三档共享 m/f/R_eff/噪声)
  - batch = k 场景 × 3 档 (L_trip/L_rank 必须三连图同 batch); 场景序 = level 序 [1.0,0.6,0.3]
  - 只训头: 单列前向 forward_alpha_only, 不走 subnet/解码器
  - 损失: L_mask + L_g + L_c + L_trip + L_rank + L_α (无 κ_naked)
  - 两配置: --encoder taps (FocalNet 特征) / raw (独立编码器), 同头同监督 → G1 特征源判决

用法:
  python3 scripts/finetune_fac_head.py --encoder taps --gpu 0 --iters 4000 --batch 2
  python3 scripts/finetune_fac_head.py --encoder raw  --gpu 1 --iters 4000 --batch 2
输出: pilot_ckpt/fac_{encoder}/ckpt_final.pth + train_log.csv (每 500 iter 中间 ckpt)
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, RDNET)
os.chdir(RDNET)

from finetune_alpha_tap import grad_mag, load_ckpt_into  # noqa: E402
from models.arch.rdnet_atap import (FullNet_NLP_ATap_Fac,  # noqa: E402
                                    load_official_atap)

RDNET_CKPT = os.path.join(PROJ, "pretrained", "rdnet.pth")
TAGS = ("l100", "l060", "l030")
BAND_TAU = 0.02
NAKED_TAU = 0.95
OBS_TAU = 0.95          # m_lab = 1[α_GT < OBS_TAU]: 可观测膜指标


class TripletDataset(Dataset):
    """一场景 → 三档连图 (I/A 按 TAGS 序, T/R 场景级)。"""

    def __init__(self, root):
        self.root = root
        self.ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "T")))
        assert self.ids, f"空数据集: {root}"

    def __len__(self):
        return len(self.ids)

    def _img(self, sub, name):
        return np.asarray(Image.open(os.path.join(self.root, sub, name)).convert("RGB"),
                          np.float32).transpose(2, 0, 1) / 255.0

    def __getitem__(self, i):
        sid = self.ids[i]
        I = torch.from_numpy(np.stack([self._img("I", f"{sid}_{t}.png") for t in TAGS]))
        A = torch.from_numpy(np.stack([
            np.load(os.path.join(self.root, "alpha", f"{sid}_{t}.npy")).astype(np.float32)[None]
            for t in TAGS]))
        T = torch.from_numpy(self._img("T", sid + ".png").copy())
        g = torch.from_numpy(np.load(os.path.join(self.root, "g", sid + ".npy")).astype(np.float32))
        m = torch.from_numpy(np.load(os.path.join(self.root, "m", sid + ".npy")).astype(np.float32))
        return I, T, A, g, m


def edge_band(m, px=3):
    """形态学梯度 + 膨胀 px 像素 → 边界带 (float 0/1)。"""
    mx, mn = F.max_pool2d(m, 3, 1, 1), -F.max_pool2d(-m, 3, 1, 1)
    return F.max_pool2d((mx - mn).clamp(0, 1), 2 * px + 1, 1, px)


def fac_losses(net, I, T, A, g_gt, m_gt, lvl_idx, args):
    """附录 A 六项损失。batch = k 场景 × 3 档 (scene-major)。"""
    k3 = I.shape[0]
    alpha, m, g, p = net.forward_alpha_only(I)
    k = k3 // 3
    m_lab = (A < OBS_TAU).float()                                    # 可观测膜指标
    obs = torch.ones_like(m_lab)
    if getattr(args, "ignore_clipped", False):
        # 膜区但 α 被 clip ≥0.95 → 与裸区物理不可分, 标签矛盾(教幻觉), 掩膜损失不监督
        obs = 1.0 - (m_gt > 0.5).float() * (1.0 - m_lab)
    w_b = (1.0 + 5.0 * edge_band(m_lab)) * obs

    bce = -(m_lab * (m + 1e-6).log() + (1.0 - m_lab) * (1.0 - m + 1e-6).log())   # BCE, autocast 安全
    L_mask = (bce * w_b).sum() / w_b.sum().clamp(min=1.0)

    L_g = ((g - g_gt).abs() * m_lab).sum() / m_lab.sum().clamp(min=1.0)

    c_hat = (p * net.alpha_levels).sum(1)
    a_lvl = net.alpha_levels[lvl_idx]
    L_c = (-torch.log(p.gather(1, lvl_idx[:, None]) + 1e-8)).mean() \
        + 0.5 * ((c_hat - a_lvl).abs() / a_lvl).mean()

    band = (grad_mag(A) > BAND_TAU).float()
    w_a = T.mean(1, keepdim=True) * (1.0 + 10.0 * band)
    L_alpha = (((alpha.clamp(min=1e-3).log() - A.clamp(min=1e-3).log()).abs()
                * w_a).sum() / w_a.sum().clamp(min=1.0))

    # --- 三连图项: reshape 成 (k,3,...) 后按档位配对 ---
    M = m.view(k, 3, 1, *m.shape[-2:])
    G = g.view(k, 3, 1, *g.shape[-2:])
    C = c_hat.view(k, 3)
    obs_ok = A.view(k, 3, 1, *A.shape[-2:]) < OBS_TAU                 # 双侧可观测
    L_trip = torch.zeros((), device=I.device)
    for lo, hi in ((0, 1), (1, 2), (0, 2)):
        o = (obs_ok[:, lo] & obs_ok[:, hi]).float()
        den = o.sum().clamp(min=1.0)
        L_trip = L_trip + (((M[:, lo] - M[:, hi]).abs() + (G[:, lo] - G[:, hi]).abs()) * o).sum() / den
    L_trip = L_trip / 3.0

    L_rank = torch.zeros((), device=I.device)
    for lo, hi in ((0, 1), (1, 2), (0, 2)):
        margin = 0.1 * (net.alpha_levels[lo] - net.alpha_levels[hi])
        L_rank = L_rank + F.relu(margin - (C[:, lo] - C[:, hi])).mean()
    L_rank = L_rank / 3.0

    total = (args.lam_mask * L_mask + args.lam_g * L_g + args.lam_c * L_c
             + args.lam_trip * L_trip + args.lam_rank * L_rank + args.lam_alpha * L_alpha)
    with torch.no_grad():
        stats = dict(L_mask=float(L_mask), L_g=float(L_g), L_c=float(L_c),
                     L_trip=float(L_trip), L_rank=float(L_rank), L_alpha=float(L_alpha),
                     m_hat_mean=float(m.mean()), c_hat_mean=float(c_hat.mean()),
                     alpha_hat_mean=float(alpha.float().mean()),
                     alpha_mae=float((alpha.float() - A).abs().mean()))
    return total, stats


def build(encoder, device, init=None, lr=1e-4):
    net = FullNet_NLP_ATap_Fac(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
                               num_subnet=4, loss_col=4, inject=True,
                               raw_encoder=(encoder == "raw"))
    load_official_atap(net, RDNET_CKPT, verbose=False)
    if init:
        load_ckpt_into(net, init)
    keep = ("alpha_fac.", "alpha_refine.")
    keep += ("alpha_raw_enc.",) if encoder == "raw" else ("alpha_c3.", "baseball_adapter.")
    for n, p in net.named_parameters():
        p.requires_grad_(n.startswith(keep))
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[fac-{encoder}] trainable {n_tr/1e3:.1f}K "
          f"({'独立编码器, 不读 FocalNet' if encoder == 'raw' else 'FocalNet 特征 + adapter taps'})")
    return net.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["taps", "raw"], required=True)
    ap.add_argument("--triplets", default=os.path.join(PROJ, "train_triplets_0913"))
    ap.add_argument("--out", default=None, help="默认 pilot_ckpt/fac_{encoder}")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2, help="场景数; 实际 batch = 3×")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-mask", type=float, default=1.0)
    ap.add_argument("--lam-g", type=float, default=1.0)
    ap.add_argument("--lam-c", type=float, default=0.3)
    ap.add_argument("--lam-trip", type=float, default=0.5)
    ap.add_argument("--lam-rank", type=float, default=0.3)
    ap.add_argument("--lam-alpha", type=float, default=1.0)
    ap.add_argument("--ignore-clipped", action="store_true",
                    help="掩膜损失忽略'膜区但 α 被 clip≥0.95'的不可观测像素(消伪标签矛盾)")
    ap.add_argument("--init", default=None, help="初始 ckpt (默认官方 rdnet.pth)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()
    out = args.out or os.path.join(PROJ, "pilot_ckpt", f"fac_{args.encoder}")
    if not os.path.isabs(out):
        out = os.path.join(PROJ, out)
    os.makedirs(out, exist_ok=True)
    for k in ("init", "triplets"):
        v = getattr(args, k)
        if v and not os.path.isabs(v):
            setattr(args, k, os.path.join(PROJ, v))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    ds = TripletDataset(args.triplets)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    print(f"[fac-{args.encoder}] 场景 {len(ds)} × 3 档, iters={args.iters} "
          f"batch={args.batch}×3={args.batch*3} -> {out}")

    net = build(args.encoder, device, args.init)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    logf = open(os.path.join(out, "train_log.csv"), "w")
    logf.write("iter,L_mask,L_g,L_c,L_trip,L_rank,L_alpha,total,m_hat,c_hat,alpha_hat,alpha_mae,sec\n")
    t0 = time.time()
    it = 0
    run = {}
    while it < args.iters:
        for I3, T, A3, g3, m3 in dl:
            if it >= args.iters:
                break
            k = I3.shape[0]
            I = I3.flatten(0, 1).to(device, non_blocking=True)      # (k*3,3,H,W)
            A = A3.flatten(0, 1).to(device, non_blocking=True)
            T = T.repeat_interleave(3, 0).to(device, non_blocking=True)
            g_gt = g3.repeat_interleave(3, 0).unsqueeze(1).to(device, non_blocking=True)
            m_gt = m3.repeat_interleave(3, 0).unsqueeze(1).to(device, non_blocking=True)
            lvl_idx = torch.arange(3, device=device).repeat(k)

            with torch.cuda.amp.autocast():
                loss, run = fac_losses(net, I, T, A, g_gt, m_gt, lvl_idx, args)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            it += 1
            if it % args.log_every == 0 or it == 1:
                sec = time.time() - t0
                print(f"[fac-{args.encoder}] it {it:4d} L_mask={run['L_mask']:.4f} L_g={run['L_g']:.4f} "
                      f"L_c={run['L_c']:.4f} L_trip={run['L_trip']:.4f} L_rank={run['L_rank']:.4f} "
                      f"L_a={run['L_alpha']:.4f} | m̂={run['m_hat_mean']:.3f} ĉ={run['c_hat_mean']:.3f} "
                      f"α̂={run['alpha_hat_mean']:.3f} MAE={run['alpha_mae']:.3f} {sec:.0f}s")
                logf.write(",".join([str(it)] + [f"{run[k]:.5f}" for k in
                                                 ("L_mask", "L_g", "L_c", "L_trip", "L_rank",
                                                  "L_alpha")] + [f"{float(loss):.5f}",
                                                 f"{run['m_hat_mean']:.4f}", f"{run['c_hat_mean']:.4f}",
                                                 f"{run['alpha_hat_mean']:.4f}",
                                                 f"{run['alpha_mae']:.4f}", f"{sec:.1f}"]) + "\n")
                logf.flush()
            if it % args.save_every == 0:
                torch.save({"icnn": net.state_dict(), "encoder": args.encoder,
                            "args": vars(args), "iter": it},
                           os.path.join(out, f"ckpt_{it}.pth"))
    logf.close()
    torch.save({"icnn": net.state_dict(), "encoder": args.encoder, "args": vars(args)},
               os.path.join(out, "ckpt_final.pth"))
    print(f"done [fac-{args.encoder}] -> {out}/ckpt_final.pth  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
