#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch finetune_rdnet_s2.py: add --omega-mode {real,random}. ASCII-only."""
import io
import os
import sys

P = "/DataDisk/zmy_code/TintedGlass_Enhance/scripts/finetune_rdnet_s2.py"
src = io.open(P, encoding="utf-8").read()

if "--omega-mode" in src:
    print("[skip] already patched")
    sys.exit(0)

# ---- 1) helper function: insert before the first 'def ' after the imports block
HELPER = '''
_RAND_CNT = [0]


def area_matched_random_mask(OM):
    """Replace each Omega mask with a random disk of (approx.) the same area.

    Fair control for "does the HOLE STRUCTURE matter, or is any region of the same size enough?"
    """
    B, _, H, W = OM.shape
    out = torch.zeros_like(OM)
    ys = torch.arange(H, device=OM.device, dtype=torch.float32)[:, None]
    xs = torch.arange(W, device=OM.device, dtype=torch.float32)[None, :]
    for b in range(B):
        A = float(OM[b].sum().item())
        if A < 1.0:
            continue
        r = (A / 3.14159265) ** 0.5
        cy = float(torch.rand(1, device=OM.device).item()) * H
        cx = float(torch.rand(1, device=OM.device).item()) * W
        m = ((ys - cy) ** 2 + (xs - cx) ** 2) <= (r * r)
        out[b, 0] = m.to(out.dtype)
    _RAND_CNT[0] += 1
    if _RAND_CNT[0] <= 5:
        ra = float(out.sum().item()) / max(float(OM.sum().item()), 1e-6)
        print("[omega-random] realized area ratio = %.3f" % ra, flush=True)
    return out

'''

anchor = "\ndef charbonnier("
if anchor not in src:
    print("[err] anchor 'def charbonnier(' not found"); sys.exit(1)
src = src.replace(anchor, HELPER + "\ndef charbonnier(", 1)

# ---- 2) add the CLI flag next to --lam-om
old_arg = "    ap.add_argument('--lam-om', type=float, default=3.0)"
if old_arg not in src:
    print("[err] --lam-om arg line not found"); sys.exit(1)
new_arg = old_arg + "\n    ap.add_argument('--omega-mode', choices=['real', 'random'], default='real',\n                    help='real = use the Omega mask; random = area-matched random disk (control)')"
src = src.replace(old_arg, new_arg, 1)

# ---- 3) hook it in the loss block
old_blk = "                if has_om.max() > 0:\n                    om_sum = OM.sum(dim=(1, 2, 3)) + 1e-6"
if old_blk not in src:
    print("[err] loss block anchor not found"); sys.exit(1)
new_blk = ("                if has_om.max() > 0:\n"
           "                    if args.omega_mode == 'random':\n"
           "                        OM = area_matched_random_mask(OM)\n"
           "                    om_sum = OM.sum(dim=(1, 2, 3)) + 1e-6")
src = src.replace(old_blk, new_blk, 1)

io.open(P + ".bak_1026", "w", encoding="utf-8").write(io.open(P, encoding="utf-8").read()) if False else None
import shutil
shutil.copy(P, P + ".bak_1026")
io.open(P, "w", encoding="utf-8").write(src)
print("[ok] patched", P)
print("  helper inserted:", "area_matched_random_mask" in src)
print("  arg added      :", "--omega-mode" in src)
print("  hook inserted  :", "args.omega_mode == 'random'" in src)
