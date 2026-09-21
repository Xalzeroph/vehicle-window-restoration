import cv2, numpy as np, os, importlib.util
os.chdir('/DataDisk/zmy_code/TintedGlass_Enhance')
spec=importlib.util.spec_from_file_location('eg2','scripts/eval_g2_baseline.py'); eg2=importlib.util.module_from_spec(spec); spec.loader.exec_module(eg2)
G=eg2.G
D='data/test_sets/real_pair_apr_0917'; OFF='runs/pred/apr_rdnet_native'
agg={'br':[], 'dk':[], 'om':[], 'gl':[]}
for sc in sorted(x for x in os.listdir(D) if x.startswith('VID_')):
    T=cv2.imread(f'{D}/{sc}/GT.png')
    for fn in sorted(f for f in os.listdir(f'{D}/{sc}') if f.startswith('I')):
        key=f'{sc}__{fn[:-4]}'
        I=cv2.imread(f'{D}/{sc}/{fn}'); P=cv2.imread(f'{OFF}/{key}/ytmt_ucs_sirs_l.png')
        o=eg2.omega_of(I,T)
        if not o: continue
        Om=o['Om']
        Il=(np.clip(I,0,255)/255.)**G; Tl=(np.clip(T,0,255)/255.)**G
        valid=Tl.mean(2)>(10/255.)**G
        ratio=np.where(valid[...,None], Il/np.where(Tl>1e-8,Tl,np.nan), np.nan)
        s=float(np.nanquantile(ratio.mean(2),0.05))
        R=(I.astype(np.float64)-(np.clip(s*Tl,0,1))**(1/G)*255).mean(2)
        stR=cv2.boxFilter(np.abs(cv2.Laplacian(R.astype(np.float32),cv2.CV_32F)),-1,(9,9))
        gI=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY).astype(np.float64)
        gP=cv2.cvtColor(P,cv2.COLOR_BGR2GRAY).astype(np.float64)
        gT=cv2.cvtColor(T,cv2.COLOR_BGR2GRAY).astype(np.float64)
        Omd=cv2.dilate(Om.astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)
        br=(R>12)&(stR>1)&(~Omd)                      # 亮结构反射区
        dk=(gI<np.percentile(gI,15))&(np.abs(R)<8)&(~Omd)  # 固定暗结构区（两图皆暗、非洞）
        def m(reg):
            if reg.sum()<500: return None
            return (float((gP-gI)[reg].mean()), float(np.abs(gP-gT)[reg].mean()), float(np.abs(gI-gT)[reg].mean()))
        for nm,reg in (('br',br),('dk',dk),('om',Om)):
            v=m(reg)
            if v: agg[nm].append(v)
        agg['gl'].append((float(np.abs(gP-gI).mean()), float(np.abs(gP-gT).mean()), float(np.abs(gI-gT).mean())))
import numpy as np
print(f'{"区域":<10}{"面积%":>7}{"改动(P−I)":>13}{"误差|P−T|":>13}{"输入误差|I−T|":>13}{"改善?":>7}')
for nm,label in (('br','亮结构反射'),('dk','固定暗结构'),('om','人影洞Ω'),('gl','全图')):
    a=np.array(agg[nm]); 
    print(f'{label:<10}{"":>7}{a[:,0].mean():>13.1f}{a[:,1].mean():>13.1f}{a[:,2].mean():>13.1f}'
          f'{"是" if a[:,1].mean()<a[:,2].mean() else "否":>7}')
