#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kill test for RCM (Relative Confidence Modulation) -- 2026-10-26, ASCII-only.

Question: does a conditioned modulation module actually USE the confidence map it is given?
(The failure mode we fear: the network routes around the conditioning channel and the map is decorative.)

Design
  frozen RDNet -> T0 (cached). Small trainable ModNet:  [I, T0, m] -> residual  ->  T_hat = T0 + residual.
  m = oracle relative reflection strength  |R| / mean(|R|)   (per image normalised)
  Arms
    A  real m
    B  shuffled m  (m taken from another sample)   -> does the *correct* map matter?
    C  no-map ablation (zero channel)
  Override test (on arm A, val scenes): replace m by {zeros, ones, noise, shuffled}
    K1  the map is USED  : mean |d mu| >= 1.0 gray or RMS >= 0.01
    K2  the map MATTERS  : PSNR(A) - PSNR(B) >= 0.3 dB (val scenes; low power, reported not gated)

Judgement
  K1 fails  -> architecture ignores the conditioning path -> RCM direction CLOSED
  K1 passes -> mechanism exists -> proceed to learned map (and only then to PSNR arms)
"""
from __future__ import annotations

import inspect
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

RES = 128
CACHE = "/tmp/rcm_cache.npz"


def luma(x):
    return 0.299 * x[:, 2:3] + 0.587 * x[:, 1:2] + 0.114 * x[:, 0:3]


def load_pair(base, case):
    I = cv2.imread(os.path.join(base, "I", case + ".png"), cv2.IMREAD_COLOR)
    T = cv2.imread(os.path.join(base, "T", case + ".png"), cv2.IMREAD_COLOR)
    Rp = cv2.imread(os.path.join(base, "R", case + ".png"), cv2.IMREAD_COLOR)
    if I is None or T is None or Rp is None:
        return None
    T = cv2.resize(T, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_AREA)
    return I, T, Rp.astype(np.float32) - 128.0


def cache_all(dev):
    base = os.path.join(P, "data/train_sets/real_pairs_1020")
    meta = json.load(open(os.path.join(base, "meta.json"), encoding="utf-8"))
    cases = sorted(meta)
    try:
        kwargs = dict(save_memory=True, grad_ckpt=False)
        sig = inspect.signature(build)
        if "mona" in sig.parameters:
            kwargs["mona"] = None
        net, net_c = build(**kwargs)
    except Exception as e:  # noqa: BLE001
        print("[build] retry without extra kwargs:", e)
        net, net_c = build()
    net.eval().to(dev); net_c.eval().to(dev)
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net_c.parameters():
        p.requires_grad_(False)

    Is, Ts, T0s, Ms, scn = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, c in enumerate(cases):
            got = load_pair(base, c)
            if got is None:
                continue
            I, T, R = got
            I128 = cv2.resize(I, (RES, RES), interpolation=cv2.INTER_AREA)
            T128 = cv2.resize(T, (RES, RES), interpolation=cv2.INTER_AREA)
            R128 = cv2.resize(R, (RES, RES), interpolation=cv2.INTER_AREA)
            m = np.abs(R128).mean(axis=2)
            m = m / max(float(m.mean()), 1e-6)
            xi = torch.from_numpy(I128.astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
            code = net_c(xi)
            _, outs = net(xi, code, prompt=True)
            t0img = outs[-1][:, 3:].clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
            Is.append(I128.astype(np.float32) / 255.)
            Ts.append(T128.astype(np.float32) / 255.)
            T0s.append(t0img)
            Ms.append(m.astype(np.float32))
            scn.append(meta[c]["scene"])
            if (i + 1) % 50 == 0:
                print("[cache] %d/%d  %.0fs" % (i + 1, len(cases), time.time() - t0), flush=True)
    np.savez(CACHE, I=np.stack(Is), T=np.stack(Ts), T0=np.stack(T0s),
             m=np.stack(Ms), scene=np.array(scn))
    print("[cache] saved", CACHE, "n=", len(Is))


class ModNet(nn.Module):
    """[I, T0, m] -> residual. Zero-init last layer => starts exactly at T0."""

    def __init__(self, cin=7, w=32):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, w, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.b[-1].weight); nn.init.zeros_(self.b[-1].bias)

    def forward(self, x):
        return self.b(x)


def train_arm(d, tr, va, mode, dev, iters=900, bs=8, seed=0):
    torch.manual_seed(seed)
    net = ModNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    I = torch.from_numpy(d["I"]).permute(0, 3, 1, 2).to(dev)
    T = torch.from_numpy(d["T"]).permute(0, 3, 1, 2).to(dev)
    T0 = torch.from_numpy(d["T0"]).permute(0, 3, 1, 2).to(dev)
    M = torch.from_numpy(d["m"])[:, None].to(dev)
    tr = torch.from_numpy(tr); va = torch.from_numpy(va)
    Itr, Ttr, T0tr, Mtr = I[tr], T[tr], T0[tr], M[tr]
    Iva, Tva, T0va, Mva = I[va], T[va], T0[va], M[va]
    n = Itr.shape[0]
    for it in range(iters):
        idx = torch.randint(0, n, (bs,), device=dev)
        m = Mtr[idx]
        if mode == "shuffled":
            m = Mtr[torch.randperm(n, device=dev)[:bs]]
        if mode == "nomap":
            m = torch.zeros_like(m)
        x = torch.cat([Itr[idx], T0tr[idx], m], 1)
        out = (T0tr[idx] + net(x)).clamp(0, 1)
        loss = charbonnier(out, Ttr[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 300 == 0:
            print("   [%s] it %d loss %.5f" % (mode, it + 1, float(loss.detach())), flush=True)
    # eval
    net.eval()
    with torch.no_grad():
        out_va_real = (T0va + net(torch.cat([Iva, T0va, Mva], 1))).clamp(0, 1)
        psnr = float(-10 * torch.log10(F.mse_loss(out_va_real, Tva)))
        psnr0 = float(-10 * torch.log10(F.mse_loss(T0va, Tva)))
        # ---- override test (only meaningful for a trained arm) ----
        base = out_va_real
        res = {"psnr": psnr, "psnr_T0": psnr0, "ov": {}}
        for tag, mv in (("zeros", torch.zeros_like(Mva)),
                        ("ones", torch.ones_like(Mva)),
                        ("noise", torch.rand_like(Mva)),
                        ("shuffled", Mva[torch.randperm(Mva.shape[0], device=dev)])):
            o = (T0va + net(torch.cat([Iva, T0va, mv], 1))).clamp(0, 1)
            dmu = (luma(o) - luma(base)).abs().mean().item() * 255.0
            rms = (o - base).pow(2).mean().sqrt().item()
            res["ov"][tag] = (dmu, rms)
    return net, res


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CACHE):
        cache_all(dev)
    d = np.load(CACHE, allow_pickle=True)
    scenes = d["scene"].astype(str)
    ks = sorted(set(scenes))
    va_scenes = set(ks[-2:])                       # hold out 2 scenes
    va = np.array([s in va_scenes for s in scenes])
    tr = ~va
    print("scenes=%d  train frames=%d (%d scenes)  val frames=%d (%d scenes: %s)"
          % (len(ks), int(tr.sum()), len(set(scenes[tr])), int(va.sum()),
             len(va_scenes), sorted(va_scenes)))

    out = {}
    for mode in ("real", "shuffled", "nomap"):
        print("\n===== arm: m = %s =====" % mode, flush=True)
        _, r = train_arm(d, tr, va, mode, dev)
        out[mode] = r
        print("   val PSNR %.3f dB | frozen-RDNet T0 %.3f dB | dPSNR %+.3f"
              % (r["psnr"], r["psnr_T0"], r["psnr"] - r["psnr_T0"]))
        for tag, (dmu, rms) in r["ov"].items():
            print("   override[%-8s] dmu=%.3f gray  rms=%.4f" % (tag, dmu, rms))

    print("\n================ VERDICT ================")
    r = out["real"]
    worst = max(v[0] for v in r["ov"].values())
    best_rms = max(v[1] for v in r["ov"].values())
    k1 = (worst >= 1.0) or (best_rms >= 0.01)
    k2 = (r["psnr"] - out["shuffled"]["psnr"]) >= 0.3
    print("  K1 map is USED      : max|dmu|=%.3f gray (gate>=1.0) or rms=%.4f (gate>=0.01) -> %s"
          % (worst, best_rms, "PASS" if k1 else "FAIL"))
    print("  K2 map MATTERS      : PSNR(real)-PSNR(shuffled)=%+.3f dB (gate>=0.3) -> %s"
          % (r["psnr"] - out["shuffled"]["psnr"], "PASS" if k2 else "FAIL"))
    print("  dPSNR real/nomap    : %+.3f / %+.3f dB vs frozen T0"
          % (r["psnr"] - r["psnr_T0"], out["nomap"]["psnr"] - r["psnr_T0"]))
    if not k1:
        print("  => CLOSED: architecture routes around the conditioning path; the map is decorative.")
    elif k2:
        print("  => OPEN: the module uses the map and the correct map helps. Proceed to a learned map.")
    else:
        print("  => MECHANISM ONLY: the module consumes m but the oracle map does not help (low-power val).")
    print("\n[note] val = 2 held-out scenes -> K2 has very low power; K1 is the decisive kill gate.")


if __name__ == "__main__":
    main()
