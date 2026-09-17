import csv, os
import cv2, numpy as np, torch, pyiqa
from scipy.stats import pearsonr
SETS={'jun':('data/test_sets/real_pair_0917','T.png','runs/pred/real_pair_0917_rdnet'),
      'apr':('data/test_sets/real_pair_apr_0917','GT.png','runs/pred/apr_rdnet'),
      'may':('data/test_sets/real_pair_may_0917','GT.png','runs/pred/may_rdnet')}
BS=128; WS=[0.0,0.25,0.5,0.75,1.0]
def psnr(a,b):
    m=float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return 99.0 if m<=1e-12 else 10*np.log10(255.0**2/m)
def path(t,sc,st):
    b=SETS[t][2]; return f'{b}/{sc}/ytmt_ucs_sirs_l.png' if t=='jun' else f'{b}/{sc}__{st}/ytmt_ucs_sirs_l.png'
def T(b): return torch.from_numpy(cv2.cvtColor(b,cv2.COLOR_BGR2RGB).astype(np.float32)/255).permute(2,0,1).unsqueeze(0)
dev=torch.device('cuda:0'); lp=pyiqa.create_metric('lpips',device=dev)
dL1=[];dP=[];dLp=[]
with torch.no_grad():
    for tag,(d,gtn,_) in SETS.items():
        for sc in sorted(x for x in os.listdir(d) if x.startswith('VID_')):
            GT=cv2.imread(f'{d}/{sc}/{gtn}')
            if GT is None: continue
            for fn in (sorted(f for f in os.listdir(f'{d}/{sc}') if f.startswith('I')) or ['I.png']):
                I=cv2.imread(f'{d}/{sc}/{fn}'); pp=path(tag,sc,fn[:-4])
                if I is None or not os.path.exists(pp): continue
                wh=(GT.shape[1],GT.shape[0]); R=cv2.resize(cv2.imread(pp),wh,interpolation=cv2.INTER_LINEAR)
                h,w=GT.shape[:2]
                for y in range(0,h-BS+1,BS):
                    for x in range(0,w-BS+1,BS):
                        g=GT[y:y+BS,x:x+BS]
                        if g.mean()<15: continue
                        ib,rb=I[y:y+BS,x:x+BS],R[y:y+BS,x:x+BS]
                        tg,ti=T(g).to(dev),T(ib).to(dev)
                        li=float(lp(ti,tg)); l1i=float(np.abs(ib.astype(np.float64)-g.astype(np.float64)).mean()); p0=psnr(ib,g)
                        for wv in WS:
                            o=np.clip(ib*(1-wv)+rb*wv,0,255).astype(np.uint8)
                            dL1.append(float(np.abs(o.astype(np.float64)-g.astype(np.float64)).mean())-l1i)
                            dP.append(psnr(o,g)-p0)
                            dLp.append(float(lp(T(o).to(dev),tg))-li)
dL1,dP,dLp=map(np.array,(dL1,dP,dLp))
print(f'样本 {len(dL1)}（3223 块 × 5 个 w）')
r1,p1=pearsonr(dL1,dLp); r2,p2=pearsonr(dP,dLp); r3,p3=pearsonr(dL1,dP)
print(f'corr(ΔL1残留, ΔLPIPS)   = {r1:+.3f}  (p={p1:.1e})')
print(f'corr(ΔPSNR,   ΔLPIPS)   = {r2:+.3f}  (p={p2:.1e})')
print(f'corr(ΔL1残留, ΔPSNR)    = {r3:+.3f}  (p={p3:.1e})  <- 两指标共线性')
print()
print('判定：若 corr(ΔL1,ΔLPIPS) 仍为负 → "PSNR 冤枉门控"假设被否' )
np.save('/tmp/dsv/e8_arrays.npy', np.stack([dL1,dP,dLp]))
import csv
with open('eval/eval_e8_corr.csv','w',newline='') as fh:
    wr=csv.writer(fh); wr.writerow(['metric_a','metric_b','pearson_r','p'])
    wr.writerow(['dL1_residual','dLPIPS',round(r1,4),p1]); wr.writerow(['dPSNR','dLPIPS',round(r2,4),p2]); wr.writerow(['dL1_residual','dPSNR',round(r3,4),p3])
