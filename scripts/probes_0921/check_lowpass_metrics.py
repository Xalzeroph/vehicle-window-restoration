import os, sys, cv2
import numpy as np
os.chdir('/DataDisk/zmy_code/TintedGlass_Enhance')
D = 'data/test_sets/real_pair_may_0917'
PRED = 'runs/pred/may_rdt_s2_main_s101_10000'
print(f'{"case":<24} {"反射区亮度回归比":>14} {"细节保留率(反射区)":>16} {"细节保留率(全图)":>14}')
print('  （亮度回归比: 0=反射加亮完全去掉 1=原样；细节保留率: P的梯度能量/T的梯度能量，1=完好 0=抹平）')
for sc, fn in [('VID_20260512_164653','I_000000.png'), ('VID_20260512_163859','I_000000.png'),
               ('VID_20260512_162203','I_000000.png'), ('VID_20260512_162623','I_000700.png'),
               ('VID_20260512_162901','I_000675.png')]:
    I = cv2.imread(f'{D}/{sc}/{fn}').astype(np.float64)
    T = cv2.imread(f'{D}/{sc}/GT.png').astype(np.float64)
    P = cv2.imread(f'{PRED}/{sc}__{fn[:-4]}/ytmt_ucs_sirs_l.png').astype(np.float64)
    Dm = np.abs(I - T).mean(2)
    refl = Dm >= np.percentile(Dm, 85)
    lI, lT, lP = I.mean(2), T.mean(2), P.mean(2)
    reg = (lP[refl].mean() - lT[refl].mean()) / max(lI[refl].mean() - lT[refl].mean(), 1e-6)
    g = lambda x: np.abs(cv2.Laplacian(x.astype(np.float32), cv2.CV_32F)).mean()
    det_r = g(P)[refl] if False else None
    Pg = np.abs(cv2.Laplacian(cv2.cvtColor(P.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F))
    Tg = np.abs(cv2.Laplacian(cv2.cvtColor(T.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F))
    Ig = np.abs(cv2.Laplacian(cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F))
    print(f'{sc[-6:]}__{fn[:-4]:<8} {reg:>14.3f} {Pg[refl].mean()/max(Tg[refl].mean(),1e-6):>16.3f} {Pg.mean()/max(Tg.mean(),1e-6):>14.3f}  (I/T={Ig.mean()/max(Tg.mean(),1e-6):.2f})')
