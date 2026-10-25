#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PGC arm test: can we make the model OBEY a photometric scalar, and is counterfactual training needed?

ASCII-only. Two arms, one seed, short budget.

Key simplification (exact):
  s := rho_tgt.  Counterfactual target is  T' = T * (s * mu(I) / mu(T)).
  With s = u * rho_true and rho_true = mu(T)/mu(I)  =>  T' = u * T.
  So the target is just the GT rescaled by u, and the conditioning scalar is s = u * rho_true.

Arms
  A  counterfactual : u ~ U[1-d, 1+d] resampled every iteration, s = u * rho_true
  B  fixed          : u == 1 (s = rho_true), i.e. ordinary training, no counterfactual

Intervention test (both arms): feed s = u * rho_true for u in a grid, measure mean luma of output.
  ControlGain = (d mu(T_hat) / d s) / mu(I)          gate >= 0.50
  P1  arm A obeys the knob            ControlGain >= 0.50
  P2  counterfactual is necessary     arm B ControlGain <= 0.20
  P3  no damage at s = rho_true       PSNR(A) >= PSNR(frozen RDNet) - 0.5 dB
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
sys.path.insert(0, os.path.join(P, "scripts"))
sys.path.insert(0, os.path.join(P, "RDNet"))
os.chdir(os.path.join(P, "scripts"))

from finetune_rdnet_peft import build, charbonnier  # noqa: E402
from rdnet_peft import set_mona_trainable  # noqa: E402

RES = 128
CACHE = "/tmp/pgc_cache.npz"


def luma(x):
    return 0.299 * x[:, 2:3] + 0.587 * x[:, 1:2] + 0.114 * x[:, 0:3]


def build_cache():
    base = os.path.join(P, "data/train_sets/real_pairs_1020")
    meta = json.load(open(os.path.join(base, "meta.json"), encoding="utf-8"))
    Is, Ts, rhos, scn = [], [], [], []
    for c in sorted(meta):
        I = cv2.imread(os.path.join(base, "I", c + ".png"))
        T = cv2.imread(os.path.join(base, "T", c + ".png"))
        if I is None or T is None:
            continue
        I = cv2.resize(I, (RES, RES), interpolation=cv2.INTER_AREA)
        T = cv2.resize(T, (RES, RES), interpolation=cv2.INTER_AREA)
        i = torch.from_numpy(I.astype(np.float32) / 255.).permute(2, 0, 1)
        t = torch.from_numpy(T.astype(np.float32) / 255.).permute(2, 0, 1)
        mu_i = float(luma(i[None]).mean())
        mu_t = float(luma(t[None]).mean())
        Is.append(i); Ts.append(t); rhos.append(mu_t / max(mu_i, 1e-6))
        scn.append(meta[c]["scene"])
    np.savez(CACHE, I=np.stack([x.numpy() for x in Is]), T=np.stack([x.numpy() for x in Ts]),
             rho=np.array(rhos), scene=np.array(scn))
    print("[cache] saved", CACHE, "n=", len(Is))


class ScalarFiLM(nn.Module):
    """Modulate the 6-dim degradation code with a scalar. Zero-init -> identity at start."""

    def __init__(self, dim=6):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True), nn.Linear(32, 2 * dim))
        nn.init.zeros_(self.fc[-1].weight); nn.init.zeros_(self.fc[-1].bias)

    def forward(self, code, s):
        g, b = self.fc(s).chunk(2, dim=1)
        return code * (1 + g) + b


def train_arm(d, tr, va, mode, dev, iters=900, bs=8, seed=0, delta=0.15):
    torch.manual_seed(seed)
    kwargs = dict(save_memory=True, grad_ckpt=True)
    net, net_c = build(**kwargs)
    # mirror finetune_rdnet_s2.py EXACTLY: freeze only 'baseball.*' (reversible net needs the rest live)
    for n, p in net.named_parameters():
        if n.startswith("baseball."):
            p.requires_grad_(False)
    set_mona_trainable(net, True)
    net.to(dev); net_c.to(dev).eval()
    film = ScalarFiLM(6).to(dev)
    params = [p for p in net.parameters() if p.requires_grad] + list(film.parameters())
    opt = torch.optim.Adam(params, lr=5e-5)
    ntr = sum(p.numel() for p in params)
    print("   trainable %.3fM" % (ntr / 1e6), flush=True)

    I = torch.from_numpy(d["I"]); T = torch.from_numpy(d["T"])
    rho = torch.from_numpy(d["rho"]).float()
    idx_pool = torch.from_numpy(np.nonzero(tr)[0]).long()
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for it in range(iters):
        pick = idx_pool[torch.randint(0, idx_pool.numel(), (bs,), generator=g)]
        xb = I[pick].to(dev); tb = T[pick].to(dev); rb = rho[pick].to(dev)[:, None]
        if mode == "cf":
            u = (1.0 + delta * (2 * torch.rand(bs, 1, generator=g) - 1)).to(dev)
        else:
            u = torch.ones(bs, 1, device=dev)
        s = (u * rb).to(dev)
        tb = (tb * u[:, :, None, None]).clamp(0, 1)
        with torch.no_grad():
            code = net_c(xb)
        with torch.cuda.amp.autocast():
            _, outs = net(xb, film(code, s), prompt=True)
            loss = torch.zeros((), device=dev)
            wsum = 0.0
            for ci, oc in enumerate(outs):
                wc = float(ci + 1); wsum += wc
                loss = loss + wc * charbonnier(oc[:, 3:], tb).mean()
            loss = loss / wsum
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            print("   [%s] it %d non-finite loss, skipped" % (mode, it + 1), flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (it + 1) % 150 == 0:
            print("   [%s] it %d loss %.5f  %.0fs" % (mode, it + 1, float(loss.detach()), time.time() - t0), flush=True)
    torch.save({"film": film.state_dict(),
                "mona": {k: v for k, v in net.state_dict().items() if "mona" in k.lower()}},
               "/tmp/pgc_%s.pt" % mode)
    print("   [saved] /tmp/pgc_%s.pt" % mode, flush=True)
    return net, net_c, film


@torch.no_grad()
def evaluate(net, net_c, film, d, va, dev, ugrid=(0.85, 0.90, 0.95, 1.00, 1.05, 1.10)):
    Iall = torch.from_numpy(d["I"][va])
    Tall = torch.from_numpy(d["T"][va])
    rall = torch.from_numpy(d["rho"][va]).float()[:, None]
    muI = float(luma(Iall).mean())
    rows = []
    for u in ugrid:
        se = 0.0; n = 0; pil = []
        for a in range(0, Iall.shape[0], 6):
            I = Iall[a:a + 6].to(dev); T = Tall[a:a + 6].to(dev); r = rall[a:a + 6].to(dev)
            _, outs = net(I, film(net_c(I), r * u), prompt=True)
            th = outs[-1][:, 3:].clamp(0, 1)
            mu = float(luma(th).mean()); pil.append((float(luma(th).sum()), th.numel() // 3))
        mus = sum(x[0] for x in pil) / sum(x[1] for x in pil)
        rows.append((u, mus, 0.0))
    # PSNR at u=1 (batched) + frozen reference
    def psnr_at(use_film, u=1.0):
        se = 0.0; n = 0
        for a in range(0, Iall.shape[0], 6):
            I = Iall[a:a + 6].to(dev); T = Tall[a:a + 6].to(dev); r = rall[a:a + 6].to(dev)
            code = net_c(I)
            _, outs = net(I, film(code, r * u) if use_film else code, prompt=True)
            th = outs[-1][:, 3:].clamp(0, 1)
            se += float(((th - T) ** 2).sum()); n += th.numel()
        return float(-10 * np.log10(se / max(n, 1)))
    psA = psnr_at(True, 1.0)
    psnr0 = psnr_at(False)
    rows = [(u, m, psA if abs(u - 1.0) < 1e-9 else 0.0) for (u, m, _) in rows]
    mus = np.array([r[1] for r in rows])
    us = np.array([r[0] for r in rows])
    slope = float(np.polyfit(us, mus, 1)[0])         # d mu / d u
    # s = u*rho, so d mu/d s = slope / mean(rho)
    rbar = float(rall.mean())
    cg = (slope / max(rbar, 1e-6)) / max(muI, 1e-6)
    return rows, psnr0, cg


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CACHE):
        build_cache()
    d = np.load(CACHE, allow_pickle=True)
    scenes = d["scene"].astype(str)
    ks = sorted(set(scenes))
    va_scenes = set(ks[-2:])
    va = np.array([s in va_scenes for s in scenes])
    tr = ~va
    print("scenes=%d train=%d val=%d (%s)" % (len(ks), int(tr.sum()), int(va.sum()), sorted(va_scenes)))
    print("rho: mean %.4f  std %.4f  range [%.3f, %.3f]"
          % (d["rho"].mean(), d["rho"].std(), d["rho"].min(), d["rho"].max()))

    res = {}
    for mode in ("cf", "fixed"):
        print("\n===== arm %s =====" % mode, flush=True)
        net, net_c, film = train_arm(d, tr, va, mode, dev)
        try:
            rows, psnr0, cg = evaluate(net, net_c, film, d, va, dev)
        except Exception as e:
            print("   [eval ERROR] %s" % e, flush=True)
            import traceback; traceback.print_exc()
            res[mode] = (float("nan"), float("nan"), [(1.0, float("nan"), float("nan"))])
            del net, net_c, film
            torch.cuda.empty_cache()
            continue
        print("   frozen RDNet (no film) val PSNR = %.3f dB" % psnr0)
        print("   %5s %10s %9s" % ("u", "mu(out)", "PSNR"))
        for u, mu, ps in rows:
            print("   %5.2f %10.4f %9.3f" % (u, mu, ps))
        res[mode] = (cg, psnr0, rows)
        print("   ControlGain = %.3f" % cg)
        del net, net_c, film
        torch.cuda.empty_cache()

    print("\n================ VERDICT ================")
    cgA, baseA, rowsA = res["cf"]
    cgB, baseB, rowsB = res["fixed"]
    psA = [r[2] for r in rowsA if abs(r[0] - 1.0) < 1e-9][0]
    p1 = cgA >= 0.50
    p2 = abs(cgB) <= 0.20
    p3 = psA >= baseA - 0.5
    print("  P1 counterfactual arm obeys the knob : ControlGain=%.3f (gate>=0.50) -> %s" % (cgA, "PASS" if p1 else "FAIL"))
    print("  P2 fixed arm does NOT obey           : ControlGain=%.3f (gate<=0.20) -> %s" % (cgB, "PASS" if p2 else "FAIL"))
    print("  P3 no damage at u=1                  : PSNR=%.3f vs frozen %.3f (gate >= -0.5) -> %s"
          % (psA, baseA, "PASS" if p3 else "FAIL"))
    if p1 and p2:
        print("  => METHOD WORKS: counterfactual training is what makes the knob obey.")
    elif not p1:
        print("  => CLOSED: even with counterfactual training the output does not track s.")
    else:
        print("  => AMBIGUOUS: A obeys but B also moves; knob works but counterfactual is not the cause.")


if __name__ == "__main__":
    main()
