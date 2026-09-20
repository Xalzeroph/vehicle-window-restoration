import sys, os, cv2
import numpy as np
sys.path.insert(0, 'scripts')
from eval_g2_baseline import omega_of, luma, PAD

CASES = [('jun','data/test_sets/real_pair_0917','T.png','VID_20260609_172143','I.png','runs/pred/real_pair_0917_rdnet'),
         ('jun','data/test_sets/real_pair_0917','T.png','VID_20260609_170941','I.png','runs/pred/real_pair_0917_rdnet'),
         ('apr','data/test_sets/real_pair_apr_0917','GT.png','VID_20260426_173349','I_000000.png','runs/pred/apr_rdnet')]
for tag, d, gtn, sc, fn, pr in CASES:
    I = cv2.imread(f'{d}/{sc}/{fn}'); T = cv2.imread(f'{d}/{sc}/{gtn}')
    key = sc if fn == 'I.png' else f'{sc}__{fn[:-4]}'
    P = cv2.imread(f'{pr}/{key}/ytmt_ucs_sirs_l.png')
    if I is None or T is None or P is None:
        print(f'{tag}/{sc}: 缺文件 I={I is not None} T={T is not None} P={P is not None} key={key}'); print('  pred dir:', os.listdir(pr)[:5] if os.path.isdir(pr) else 'N/A'); continue
    if P.shape != I.shape: P = cv2.resize(P, (I.shape[1], I.shape[0]))
    o = omega_of(I, T); Om = o['Om']
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*PAD+1, 2*PAD+1))
    ring = (cv2.dilate(Om.astype(np.uint8), ker) > 0) & (~Om)
    lI, lA, lT = luma(I), luma(P), luma(T)
    C_I = lI[ring].mean() - lI[Om].mean()
    C_A = lA[ring].mean() - lA[Om].mean()
    C_T = lT[ring].mean() - lT[Om].mean()
    SA = omega_of(P, T)
    dI_med = float(np.median(o['S'][Om]))
    print(f'== {tag}/{sc}  Omega={Om.mean()*100:.1f}% ==')
    print(f'   luma Omega-in : I {lI[Om].mean():.1f} -> rdnet {lA[Om].mean():.1f} (d{lA[Om].mean()-lI[Om].mean():+.1f}) | T {lT[Om].mean():.1f}')
    print(f'   luma ring     : I {lI[ring].mean():.1f} -> rdnet {lA[ring].mean():.1f} (d{lA[ring].mean()-lI[ring].mean():+.1f})')
    print(f'   C: I {C_I:.1f} | rdnet {C_A:.1f} | T {C_T:.1f}  ==> dC={(C_I-C_A)/C_I:+.3f}')
    if SA is not None:
        SAom = SA['S'][Om]
        print(f'   depth: dI_med {dI_med:.1f} | S_A[Om] med {np.median(SAom):.1f} p25 {np.percentile(SAom,25):.1f} | '
              f'frac(S_A<=0.4thr) {(SAom <= 0.4*dI_med).mean():.3f} | 对照 frac(S_I<=0.4thr) {(o["S"][Om] <= 0.4*dI_med).mean():.3f}')
