"""CANONICAL Omega definition -- the ONE source of truth.

Definition (2026-09-20 v2, user-approved 2026-10-27):
  linearise with gamma=2.2 ; global gain s = q05(I_lin/T_lin) ; Tsc = s*T
  Rg = I - Tsc ; bg = block-90th percentile of Rg (48 px)
  Omega = { bg - Rg > tau } & { 2*Rg < bg },  tau = max(6, 5*sigma_MAD)
  keep connected components with area >= 0.5% ; fill interior holes <= 25%

Semantics: Omega is the OCCLUDER FOOTPRINT (reflection loss + illumination
shadow are inseparable under a single-frame statistic).  It is NOT "pure
reflection removal".  See docs/2026.10.26___.md.

Known limitations (do not silently "fix"):
  L1 only a GLOBAL gain s; spatially varying gain (measured ~26% in/out of
     Omega) is not representable -> the mask can fire on window frames.
  L2 no gT criterion -> cannot separate shadow from reflection loss.
"""
import cv2
import numpy as np

G = 2.2

def omega_of(I, T):
    Il = (np.clip(I, 0, 255) / 255.0) ** G
    Tl = (np.clip(T, 0, 255) / 255.0) ** G
    valid = Tl.mean(axis=2) > (10 / 255.) ** G
    ratio = np.where(valid[..., None], Il / np.where(Tl > 1e-8, Tl, np.nan), np.nan)
    s = float(np.nanquantile(ratio.mean(axis=2), 0.05))
    Tsc = (np.clip(s * Tl, 0, 1)) ** (1 / G) * 255.0
    Rg = (I.astype(np.float64) - Tsc).mean(axis=2)
    R8 = np.clip(Rg + 128, 0, 255).astype(np.uint8)
    # 帧级稳健噪声（3×3 中值残差 MAD）
    med3 = cv2.medianBlur(R8, 3).astype(np.float64) - 128.0
    sig = 1.4826 * float(np.median(np.abs(Rg - med3)))
    tau = max(6.0, 5 * sig)
    h, w = Rg.shape
    blk = 48
    hh, ww = h // blk, w // blk
    t = Rg[:hh * blk, :ww * blk].reshape(hh, blk, ww, blk).transpose(0, 2, 1, 3).reshape(hh, ww, -1)
    bg = cv2.resize(np.percentile(t, 90, axis=2).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR).astype(np.float64)
    S = bg - Rg
    # 帧级闸：该帧几乎无反射 → Ω=∅
    if float(np.percentile(Rg, 90)) < max(12.0, 6 * sig):
        return dict(s=s, tau=tau, Rg=Rg, S=S, Om=np.zeros_like(Rg, bool), gate=True)
    M = ((S > tau) & (2 * Rg < bg)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(M, 8)
    Om = np.zeros_like(M, bool)
    for i in range(1, n):
        if st[i][4] / M.size >= 0.005:
            Om |= (lab == i)
    # 填内洞（≤25% 面积）
    inv = (~Om).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n2):
        x, y, ww, hh, area = st2[i]
        touches = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if not touches and area / M.size <= 0.25:
            Om |= (lab2 == i)
    return dict(s=s, tau=tau, Rg=Rg, S=S, Om=Om, gate=False)
