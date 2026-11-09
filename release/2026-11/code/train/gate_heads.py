"""门控头预训练：α̂（照度门，Ω 监督）+ M̂（反射门，|R| 监督），共用冻结 RDNet-ATap 主干。

设计要点（对应预登记 H1/H2）：
- 载入官方 rdnet.pth 的 **ATap** 变体（`models/arch/rdnet_atap.FullNet_NLP_ATap`），
  主干全程冻结，**只训 `alpha_*` 与 `refl_*` 两组新参数**（约 0.15 M + 0.11 M）。
- α 标签：α_gt = 1 − δ_i·Ω_s，δ_i 由该对图自身洞内外 (I/T) 中位比估出（实测 apr 中位 0.121 / jun 0.177）。
- 反射标签：E = clip(|R|/q90(|R|), 0, 1)（软）+ B = 1[|∇R| > q90]（硬边）。
- 场景级 8:1 划分（默认每 9 个场景留 1 个），防同场景泄漏。
- 报告：α̂ MAE / IoU(α̂<0.95, Ω) / α̂ std（塌缩检查）；M̂ IoU / 精确率 / 召回；
  并与 **MaxRF 伪标签同口径**（面积 48.1%、IoU 0.063）并列。

用法：
  python gate_heads.py --pool real_pairs_1027 --epochs 30 --out runs/gate_head/gate_v1.pth
  python gate_heads.py --smoke            # CPU 冒烟：2 iter
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
sys.path.insert(0, os.path.join(P, "scripts"))


# ---------------- 标签 ----------------
def to_luma(x):
    return x.mean(axis=2)


def alpha_label(I, T, om, dilate=3):
    """α_gt = 1 − δ_i·Ω_s；δ_i = 1 − med(I/T|Ω)/med(I/T|~Ω)。"""
    lI, lT = to_luma(I.astype(np.float32)), np.maximum(to_luma(T.astype(np.float32)), 1e-6)
    m = om > 0.5
    if m.sum() < 200 or (~m).sum() < 200:
        return np.ones_like(om, np.float32), 0.0
    r = lI / lT
    delta = 1.0 - float(np.median(r[m])) / max(float(np.median(r[~m])), 1e-6)
    delta = float(np.clip(delta, -0.5, 0.6))
    om_s = cv2.dilate((m.astype(np.uint8)), np.ones((dilate, dilate), np.uint8)).astype(np.float32)
    alpha = 1.0 - delta * om_s
    return np.clip(alpha, 0.0, 1.0).astype(np.float32), delta


def refl_label(R, q=0.90):
    """E（软，归一化 |R|）与 B（硬边）。"""
    a = np.abs(R.astype(np.float32))
    tau = max(float(np.quantile(a, q)), 2.0)
    E = np.clip(a / tau, 0.0, 1.0).astype(np.float32)
    gm = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))
    B = (gm > np.quantile(gm, q)).astype(np.float32)
    return E, B


# ---------------- 反射门头 ----------------
class ReflHead(nn.Module):
    """ATap 风格：stem+c3 融合 → h/2 logits → 以 I 为导引的全分辨率细化；输出 2 通道（E, B）。"""

    def __init__(self, stem_ch, c3_ch, ci_ch=1):
        super().__init__()
        self.c3 = nn.Conv2d(c3_ch, stem_ch, 1)
        self.head = nn.Sequential(
            nn.Conv2d(2 * stem_ch, stem_ch, 3, padding=1),
            nn.GroupNorm(8, stem_ch), nn.GELU(),
            nn.Conv2d(stem_ch, 2, 1))
        self.refine = nn.Sequential(nn.Conv2d(2 + ci_ch, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 2, 1))
        for m in (self.head[-1], self.refine[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_stem, c3b, x_in):
        c3u = F.interpolate(self.c3(c3b), size=x_stem.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(torch.cat([x_stem, c3u], 1))
        up = F.interpolate(logits, size=x_in.shape[-2:], mode="bilinear", align_corners=False)
        z = up + self.refine(torch.cat([up, x_in.mean(1, keepdim=True)], 1))
        return z[:, :1], z[:, 1:2]


def build_atap(inject=False, mona="none"):
    cwd = os.getcwd()
    rd = os.path.join(P, "RDNet")
    if rd not in sys.path:
        sys.path.insert(0, rd)
    os.chdir(rd)
    try:
        from models.arch.rdnet_atap import FullNet_NLP_ATap, load_official_atap
        net = FullNet_NLP_ATap(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2], num_subnet=4,
                               loss_col=4, num_classes=1000, drop_path=0, save_memory=True,
                               inter_supv=True, head_init_scale=None, kernel_size=3, inject=inject)
        load_official_atap(net, os.path.join(P, "pretrained/rdnet.pth"), verbose=False)
        if mona != "none":
            from rdnet_peft import patch_focalnet
            patch_focalnet(net, mona, verbose=False)
        stem_ch, c3_ch = 64, 512
        net.refl_head = ReflHead(stem_ch, c3_ch)
    finally:
        os.chdir(cwd)
    return net


def head_forward(net, x):
    """一次 stem 前向，同时得到 α̂ 与 M̂（复用 ATap 的内部结构）。"""
    with torch.no_grad():
        x_base, x_stem = net.baseball(x)
        x_stem = net.baseball_adapter[0](x_stem)
        c3b = net.baseball_adapter[4](x_base[3])
    alpha = net._alpha_hat(x_stem, c3b, x)
    m_logit, b_logit = net.refl_head(x_stem, c3b, x)
    return alpha, torch.sigmoid(m_logit), b_logit


# ---------------- 数据 ----------------
class PairGate(torch.utils.data.Dataset):
    def __init__(self, pool, scenes, crop=256, reps=1):
        root = os.path.join(P, "data/train_sets", pool)
        self.root, self.crop, self.reps = root, crop, reps
        ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(root, "I", "*.png")))
        self.ids = [k for k in ids if k.split("__")[1] in scenes]

    def __len__(self):
        return len(self.ids) * self.reps

    def __getitem__(self, i):
        k = self.ids[i % len(self.ids)]
        I = cv2.imread(os.path.join(self.root, "I", k + ".png")).astype(np.float32)
        T = cv2.imread(os.path.join(self.root, "T", k + ".png")).astype(np.float32)
        R = cv2.imread(os.path.join(self.root, "R", k + ".png"), 0).astype(np.float32) - 128.0
        om = cv2.imread(os.path.join(self.root, "omega", k + ".png"), 0).astype(np.float32) / 255.0
        h, w = I.shape[:2]
        c = self.crop
        y0 = np.random.randint(0, max(h - c, 1)) if h > c else 0
        x0 = np.random.randint(0, max(w - c, 1)) if w > c else 0
        sl = (slice(y0, y0 + c), slice(x0, x0 + c))
        I, T, R, om = I[sl], T[sl], R[sl], om[sl]
        a_gt, delta = alpha_label(I, T, om)
        E, B = refl_label(R)
        pack = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1) if x.ndim == 3 else torch.from_numpy(np.ascontiguousarray(x))[None]
        return dict(I=pack(I / 255.0), a=a_gt[None], E=E[None], B=B[None], delta=delta, id=k)


def split_scenes(pool, hold_every=9):
    root = os.path.join(P, "data/train_sets", pool)
    ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(root, "I", "*.png")))
    scenes = sorted({k.split("__")[1] for k in ids})
    hold = {s for i, s in enumerate(scenes) if i % hold_every == 0}
    return scenes, [s for s in scenes if s not in hold], sorted(hold)


# ---------------- 训练 / 评估 ----------------
def set_trainable(net):
    for p in net.parameters():
        p.requires_grad_(False)
    for k, p in net.named_parameters():
        if k.startswith(("alpha_", "refl_head.")):
            p.requires_grad_(True)
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(net, ds, dev, maxn=60):
    net.eval()
    maes, ious, stds, ious_m, precs, recs = [], [], [], [], [], []
    for i in range(min(len(ds), maxn)):
        b = ds[i]
        I = b["I"][None].to(dev)
        a_hat, m_hat, _ = head_forward(net, I)
        a = a_hat[0, 0].float().cpu().numpy()
        a_gt = np.asarray(b["a"])[0]
        om = (a_gt < 0.999).astype(np.float32)
        maes.append(float(np.abs(a - a_gt).mean()))
        stds.append(float(a.std()))
        pr = (a < 0.95).astype(np.float32)
        u = ((pr > 0) | (om > 0)).sum()
        ious.append(float(((pr > 0) & (om > 0)).sum()) / max(u, 1))
        m = m_hat[0, 0].float().cpu().numpy()
        E = np.asarray(b["E"])[0]
        pm = (m > 0.5).astype(np.float32)
        tm = (E > 0.5).astype(np.float32)
        u2 = ((pm > 0) | (tm > 0)).sum()
        ious_m.append(float(((pm > 0) & (tm > 0)).sum()) / max(u2, 1))
        precs.append(float(((pm > 0) & (tm > 0)).sum()) / max(pm.sum(), 1))
        recs.append(float(((pm > 0) & (tm > 0)).sum()) / max(tm.sum(), 1))
    net.train()
    med = lambda v: round(float(np.median(v)), 4) if v else None
    return dict(n=len(maes), alpha_mae=med(maes), alpha_iou=med(ious), alpha_std=med(stds),
                refl_iou=med(ious_m), refl_prec=med(precs), refl_rec=med(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="real_pairs_1027")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-alpha", type=float, default=1.0)
    ap.add_argument("--lam-ref", type=float, default=1.0)
    ap.add_argument("--out", default="runs/gate_head/gate_v1.pth")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_atap(inject=False, mona="none").to(dev)
    n_tr = set_trainable(net)
    print(f"[gate] 可训 {n_tr/1e6:.3f}M (alpha_* + refl_head.*)", flush=True)

    scenes, tr, ho = split_scenes(args.pool)
    print(f"[gate] 场景 {len(scenes)}：训练 {len(tr)} / 留出 {len(ho)} -> {ho}", flush=True)
    ds_tr = PairGate(args.pool, set(tr), crop=args.crop, reps=1)
    ds_ho = PairGate(args.pool, set(ho), crop=args.crop, reps=1)
    if args.smoke:
        ds_tr.ids, ds_ho.ids = ds_tr.ids[:2], ds_ho.ids[:2]
        args.epochs = 1
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))
    hist = []
    for ep in range(args.epochs):
        tot_a = tot_r = 0.0
        for it, b in enumerate(dl):
            I = b["I"].to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=(dev.type == "cuda")):
                a_hat, m_hat, b_logit = head_forward(net, I)
                La = args.lam_alpha * F.l1_loss(a_hat[:, 0], b["a"].to(dev)[:, 0])
                Lr = args.lam_ref * (F.l1_loss(m_hat[:, 0], b["E"].to(dev)[:, 0]) +
                                     0.5 * F.binary_cross_entropy_with_logits(b_logit[:, 0], b["B"].to(dev)[:, 0]))
                loss = La + Lr
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_a += float(La); tot_r += float(Lr)
            if it % 20 == 0:
                print(f"  ep{ep} it{it} La={float(La):.4f} Lr={float(Lr):.4f}", flush=True)
            if args.smoke and it >= 1:
                break
        ev = evaluate(net, ds_ho, dev, maxn=(4 if args.smoke else 60))
        hist.append(dict(epoch=ep, La=tot_a / max(it + 1, 1), Lr=tot_r / max(it + 1, 1), **ev))
        print(f"[ep{ep}] {ev}", flush=True)
        if args.smoke:
            break

    out = args.out if os.path.isabs(args.out) else os.path.join(P, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(dict(alpha=net.alpha_head.state_dict(), alpha_c3=net.alpha_c3.state_dict(),
                    alpha_refine=net.alpha_refine.state_dict(),
                    refl_head=net.refl_head.state_dict(),
                    cfg=dict(pool=args.pool, crop=args.crop, epochs=args.epochs,
                             holdout=ho, lr=args.lr)),
               out)
    json.dump(dict(out=out, cfg=vars(args), split=dict(train=tr, holdout=ho), history=hist),
              open(os.path.splitext(out)[0] + "_report.json", "w"), ensure_ascii=False, indent=1)
    print("[gate] saved", out)


if __name__ == "__main__":
    main()
