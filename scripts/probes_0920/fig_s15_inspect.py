import sys, os, cv2
import numpy as np
sys.path.insert(0, 'scripts')
from eval_g2_baseline import omega_of

H, WP = 300, 400
D = 'data/test_sets/real_pair_apr_0917'
PICK = [('VID_20260426_173349', 'I_000000.png', '②正 ρ+0.27'),
        ('VID_20260426_173237', 'I_000000.png', '②正 ρ+0.17'),
        ('VID_20260426_173827', 'I_000009.png', 'ρ最高 0.54 但②负')]
def fit(im):
    x = cv2.resize(im, (max(1, int(im.shape[1]*H/im.shape[0])), H), interpolation=cv2.INTER_AREA)
    c = np.zeros((H, WP, 3), np.uint8)
    c[:, :min(x.shape[1], WP)] = x[:, :WP] if x.shape[1] > WP else x
    return c
rows = []
for sc, fn, note in PICK:
    I = cv2.imread(f'{D}/{sc}/{fn}'); T = cv2.imread(f'{D}/{sc}/GT.png')
    key = f'{sc}__{fn[:-4]}'
    P = cv2.imread(f'runs/pred/apr_rdnet/{key}/ytmt_ucs_sirs_l.png')
    if I is None or T is None or P is None:
        print(f'{sc}: 缺 (I={I is not None} T={T is not None} P={P is not None})'); continue
    if P.shape != I.shape: P = cv2.resize(P, (I.shape[1], I.shape[0]))
    o = omega_of(I, T); Om = o['Om']
    cont, _ = cv2.findContours(Om.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = []
    for im, lab in ((I, 'I'), (T, 'T'), (P, 'rdnet')):
        v = im.copy()
        cv2.drawContours(v, cont, -1, (0, 0, 255), 2)
        cv2.putText(v, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(fit(v))
    lab = np.zeros((H, 250, 3), np.uint8)
    for i, t in enumerate([f'{sc[-6:]}', note, 'I | T | rdnet', '红=Omega 轮廓']):
        cv2.putText(lab, t, (6, 30+i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    rows.append(np.hstack([lab] + panels))
gap = np.full((8, rows[0].shape[1], 3), 235, np.uint8)
out = rows[0]
for r in rows[1:]: out = np.vstack([out, gap, r])
os.makedirs('fig', exist_ok=True)
cv2.imwrite('fig/s15_inspect_0920.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
print('写出 fig/s15_inspect_0920.jpg', out.shape)
