#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-evaluate the PGC arms from saved checkpoints, with net.eval() ON. ASCII-only."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = "/DataDisk/zmy_code/TintedGlass_Enhance"
sys.path.insert(0, os.path.join(P, "scripts"))
sys.path.insert(0, os.path.join(P, "RDNet"))
os.chdir(os.path.join(P, "scripts"))

from finetune_rdnet_peft import build  # noqa: E402

CACHE = "/tmp/pgc_cache.npz"


class ScalarFiLM(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True), nn.Linear(32, 2 * dim))

    def forward(self, code, s):
        g, b = self.fc(s).chunk(2, dim=1)
        return code * (1 + g) + b


def luma(x):
    return 0.299 * x[:, 2] + 0.587 * x[:, 1] + 0.114 * x[:, 0]


def main():
    dev = torch.device("cuda:0")
    d = np.load(CACHE, allow_pickle=True)
    sc = d["scene"].astype(str)
    ks = sorted(set(sc))
    va = np.array([s in set(ks[-2:]) for s in sc])
    Iall = torch.from_numpy(d["I"][va]); Tall = torch.from_numpy(d["T"][va])
    rall = torch.from_numpy(d["rho"][va]).float()[:, None]
    print("val n=%d scenes=%s" % (int(va.sum()), ks[-2:]))
    print("mu(I)=%.4f  mu(T)=%.4f  rho_mean=%.4f"
          % (float(luma(Iall).mean()), float(luma(Tall).mean()), float(rall.mean())))

    ugrid = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10)
    out = {}
    for mode in ("cf", "fixed"):
        path = "/tmp/pgc_%s.pt" % mode
        if not os.path.exists(path):
            print("  [%s] no ckpt" % mode); continue
        net, net_c = build()
        ck = torch.load(path, map_location="cpu")
        film = ScalarFiLM(6)
        film.load_state_dict(ck["film"])
        miss = net.load_state_dict(ck["mona"], strict=False)
        net.to(dev).eval(); net_c.to(dev).eval(); film.to(dev).eval()
        print("  [%s] loaded (missing=%d unexpected=%d)" % (mode, len(miss.missing_keys), len(miss.unexpected_keys)))
        rows = []
        with torch.no_grad():
            for u in ugrid:
                tot = 0.0; cnt = 0; se = 0.0
                for a in range(0, Iall.shape[0], 6):
                    I = Iall[a:a + 6].to(dev); T = Tall[a:a + 6].to(dev); r = rall[a:a + 6].to(dev)
                    _, outs = net(I, film(net_c(I), r * u), prompt=True)
                    th = outs[-1][:, 3:].clamp(0, 1)
                    tot += float(luma(th).sum()); cnt += th.numel() // 3
                    se += float(((th - T) ** 2).sum())
                rows.append((u, tot / cnt, float(-10 * np.log10(se / (Iall.numel())))))
            # frozen baseline (no film)
            se0 = 0.0; tot0 = 0.0; cnt0 = 0
            for a in range(0, Iall.shape[0], 6):
                I = Iall[a:a + 6].to(dev); T = Tall[a:a + 6].to(dev)
                _, outs = net(I, net_c(I), prompt=True)
                th = outs[-1][:, 3:].clamp(0, 1)
                se0 += float(((th - T) ** 2).sum()); tot0 += float(luma(th).sum()); cnt0 += th.numel() // 3
        base = float(-10 * np.log10(se0 / (Iall.numel())))
        print("   %5s %10s %9s" % ("u", "mu(out)", "PSNR"))
        for u, mu, ps in rows:
            print("   %5.2f %10.4f %9.3f" % (u, mu, ps))
        print("   frozen RDNet: mu=%.4f  PSNR=%.3f dB" % (tot0 / cnt0, base))
        us = np.array([r[0] for r in rows]); mus = np.array([r[1] for r in rows])
        slope = float(np.polyfit(us, mus, 1)[0])
        cg = (slope / float(rall.mean())) / float(luma(Iall).mean())
        print("   d mu/du = %.4f | ideal = mu(T) = %.4f | ControlGain = %.3f"
              % (slope, float(luma(Tall).mean()), cg))
        out[mode] = cg
        del net, net_c, film
        torch.cuda.empty_cache()

    print("\n================ VERDICT ================")
    if "cf" in out and "fixed" in out:
        a, b = out["cf"], out["fixed"]
        print("  A counterfactual : ControlGain=%.3f  (gate >= 0.50) -> %s" % (a, "PASS" if a >= 0.5 else "FAIL"))
        print("  B fixed          : ControlGain=%.3f  (gate <= 0.20) -> %s" % (b, "PASS" if abs(b) <= 0.2 else "FAIL"))
        print("  ratio |A|/|B| = %.1fx" % (abs(a) / max(abs(b), 1e-9)))


if __name__ == "__main__":
    main()
