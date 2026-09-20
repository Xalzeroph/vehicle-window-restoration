import csv, os, sys, cv2
import numpy as np
sys.path.insert(0, 'scripts')
from eval_g2_baseline import omega_of, luma, PAD, SETS

rows = []
for tag, (d, gtn, pr1, pf1, pr2, pf2) in SETS.items():
    for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
        sd = os.path.join(d, sc)
        T = cv2.imread(os.path.join(sd, gtn))
        for fn in (sorted(f for f in os.listdir(sd) if f.startswith('I')) or ['I.png']):
            I = cv2.imread(os.path.join(sd, fn))
            if I is None or T is None or I.shape != T.shape: continue
            key = sc if fn == 'I.png' else f'{sc}__{fn[:-4]}'
            o = omega_of(I, T)
            if o is None or not o['Om'].any(): continue
            Om = o['Om']
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*PAD+1, 2*PAD+1))
            ring = (cv2.dilate(Om.astype(np.uint8), ker) > 0) & (~Om)
            lI, lT = luma(I), luma(T)
            C_I = lI[ring].mean() - lI[Om].mean()
            C_T = lT[ring].mean() - lT[Om].mean()
            if C_I > 2:
                prog = (C_I - C_T) / C_I
            else:
                prog = None
            rows.append((tag, key, round(C_I,2), round(C_T,2), None if prog is None else round(prog,2)))
with open('/tmp/ct_all.csv','w',newline='') as fh:
    wr = csv.writer(fh); wr.writerow(['set','case','C_I','C_T','T_progress']); wr.writerows(rows)
import statistics as st
print('n =', len(rows))
for tag in ('jun','apr','may'):
    sub = [r for r in rows if r[0]==tag]
    cts = [r[3] for r in sub]; progs = [r[4] for r in sub if r[4] is not None]
    print(f'{tag}: n={len(sub)} C_I 中位 {st.median(r[2] for r in sub):.1f} | C_T 中位 {st.median(cts):.1f} '
          f'(负值占 {sum(1 for c in cts if c<0)}/{len(cts)}) | T_progress 中位 {st.median(progs):.2f} '
          f'(范围 {min(progs):.2f}~{max(progs):.2f})')
