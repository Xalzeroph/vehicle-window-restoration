#!/usr/bin/env python3
"""两个判决探针（Opus 2026-09-22 预登记）。

P0-1 I* 检验：I* = I 的 Ω 填 GT 内容（羽化）→ 官方 RDNet 推理。判读：PASS = I* 上
  |mean(P−I)_Ω − mean(P−I)_环带| ≤ 3 灰阶且 |P−T|_Ω 不高于 |I*−T|_Ω（≈0）；FAIL = 官方仍特例化
  处理洞区 → "先修后去"链条作废。
P0-2 记忆判别：同场景取 2 个**非训练**帧（train_mask_0921 提供，命名 1:1 对应原视频帧号）跑
  S2 main s101@10k，比 |P_a−P_b| / |I_a−I_b|（≪1 = 输出收敛到同一目标 = 记忆签名；≈1 = 跟随输入）。
用法: CUDA_VISIBLE_DEVICES=1 python3 scripts/probes_0921/probe_istar_memory_0922.py
"""
import importlib.util, os, sys
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
sys.path.insert(0, os.path.join(PROJ, 'RDNet'))
os.chdir(PROJ)
spec = importlib.util.spec_from_file_location('eg2', os.path.join(PROJ, 'scripts/eval_g2_baseline.py'))
eg2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg2)
os.chdir(os.path.join(PROJ, 'RDNet'))
from finetune_rdnet_peft import build                       # noqa: E402

D = os.path.join(PROJ, 'data/test_sets/real_pair_apr_0917')
TM = os.path.join(PROJ, 'data/train_sets/train_mask_0921/I')
TEST_FRAMES = {}                                            # scene -> set of test frame numbers
for sc in sorted(x for x in os.listdir(D) if x.startswith('VID_')):
    TEST_FRAMES[sc] = {f[2:-4] for f in os.listdir(f'{D}/{sc}') if f.startswith('I')}
MEM_SCENES = ['VID_20260426_173237', 'VID_20260426_173349', 'VID_20260426_173827']


def mem_pair(sc):
    """该场景首末两个非训练帧（train_mask_0921 采样帧，排除测试 16 帧）。"""
    files = sorted(f for f in os.listdir(TM) if f.startswith(sc + '__'))
    non = [f.split('__')[1][:-4] for f in files
           if f.split('__')[1][:-4].replace('frame_', '') not in TEST_FRAMES[sc]]
    return (non[0], non[-1]) if len(non) >= 2 else (None, None)


def pad32(x, p=32):
    h, w = x.shape[:2]
    return cv2.copyMakeBorder(x, 0, (-h) % p, 0, (-w) % p, cv2.BORDER_REFLECT)


def infer(net, net_c, I, dev):
    Ip = pad32(I)
    x = torch.from_numpy(Ip[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        code = net_c(x)
        _, outs = net(x, code, prompt=True)
        T_hat = outs[-1][:, 3:]
    Tn = T_hat[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1]
    h, w = I.shape[:2]
    return (Tn[:h, :w] * 255).round().astype(np.uint8)


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float64)


def load(mona, ckpt):
    net, net_c = build(save_memory=True, mona=mona, grad_ckpt=False)
    ck = torch.load(ckpt, map_location='cpu')
    miss, _ = net.load_state_dict(ck['icnn'], strict=False)
    assert not [k for k in miss if 'mona' not in k], miss[:5]
    net.eval(); net_c.eval()
    return net, net_c


def istar_test(dev):
    net, net_c = load('none', os.path.join(PROJ, 'pretrained/rdnet.pth'))
    net.to(dev); net_c.to(dev)
    print('\n== P0-1 I* 检验（官方；判读：|Ω与环带改动差| ≤3 且 |P−T|_Ω 不高于 |I*−T|_Ω）==')
    print(f'{"帧":<22}{"Ω改动":>8}{"环带改动":>9}{"差":>7}{"|P*−T|Ω":>9}{"|P−T|Ω(原I)":>11}{"判别":>6}')
    npass, rows = 0, []
    for sc in sorted(TEST_FRAMES):
        T = cv2.imread(f'{D}/{sc}/GT.png')
        for fn in sorted(f for f in os.listdir(f'{D}/{sc}') if f.startswith('I')):
            I = cv2.imread(f'{D}/{sc}/{fn}')
            o = eg2.omega_of(I, T)
            if not o:
                continue
            Om = o['Om']
            alpha = cv2.GaussianBlur(cv2.dilate(Om.astype(np.uint8), np.ones((7, 7), np.uint8))
                                     .astype(np.float32), (0, 0), 2.0)
            alpha = np.clip(alpha, 0, 1)[..., None]
            Istar = (I.astype(np.float32) * (1 - alpha) + T.astype(np.float32) * alpha).astype(np.uint8)
            P = infer(net, net_c, Istar, dev)
            ring = cv2.dilate(Om.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~Om
            d_om = float((gray(P) - gray(Istar))[Om].mean())
            d_ring = float((gray(P) - gray(Istar))[ring].mean())
            pT = float(np.abs(gray(P) - gray(T))[Om].mean())
            iT_star = float(np.abs(gray(Istar) - gray(T))[Om].mean())
            ok = abs(d_om - d_ring) <= 3 and pT <= iT_star + 3
            npass += ok
            print(f'{sc[-6:]}/{fn[:-4]:<10}{d_om:>8.1f}{d_ring:>9.1f}{abs(d_om-d_ring):>7.1f}'
                  f'{pT:>9.1f}{iT_star:>11.1f}{"PASS" if ok else "FAIL":>6}')
            rows.append((sc, fn, d_om, d_ring, pT, ok))
    print(f'-- I* 检验：{npass}/{len(rows)} PASS')
    return npass, len(rows)


def memory_test(dev):
    
    net, net_c = load('both', os.path.join(PROJ, 'runs/ckpt/rdt_s2_main_s101/ckpt_10000.pth'))
    net.to(dev); net_c.to(dev)
    print('\n== P0-2 记忆判别（S2 s101@10k，非训练帧；比值 |P差|/|I差| ≪1=收敛/记忆，≈1=跟随输入）==')
    print(f'{"场景":<10}{"|Ia−Ib|":>8}{"|Pa−Pb|":>8}{"比值":>7}{"|Pa−T|":>8}{"|Pb−T|":>8}{"|I−T|均值":>9}')
    for sc in MEM_SCENES:
        fa, fb = mem_pair(sc)
        pa, pb = f'{TM}/{sc}__{fa}.png', f'{TM}/{sc}__{fb}.png'
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f'{sc[-6:]} 缺帧 {fa}/{fb}'); continue
        Ia, Ib = cv2.imread(pa), cv2.imread(pb)
        if Ia.shape != Ib.shape:
            Ib = cv2.resize(Ib, (Ia.shape[1], Ia.shape[0]))
        Pa, Pb = infer(net, net_c, Ia, dev), infer(net, net_c, Ib, dev)
        T = cv2.imread(f'{D}/{sc}/GT.png')
        dI = float(np.abs(gray(Ia) - gray(Ib)).mean())
        dP = float(np.abs(gray(Pa) - gray(Pb)).mean())
        pTa = float(np.abs(gray(Pa) - gray(T)).mean())
        pTb = float(np.abs(gray(Pb) - gray(T)).mean())
        iT = float(np.abs(gray(Ia) - gray(T)).mean())
        print(f'{sc[-6:]:<10}{dI:>8.1f}{dP:>8.1f}{dP/max(dI,1e-6):>7.2f}{pTa:>8.1f}{pTb:>8.1f}{iT:>9.1f}')


def main():
    dev = torch.device('cuda:0')
    istar_test(dev)
    
    memory_test(dev)


if __name__ == '__main__':
    main()
