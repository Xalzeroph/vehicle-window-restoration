#!/usr/bin/env python3
"""跨域触发实验（2026-10-08）：LOVO 三臂 + 官方，在 may（未见域）与 apr（域内）上做注入判决。
沿用 9.21 实验 B 口径：注入 apr-Ω 最大连通域形状的软边暗块（depth≈30）到帧内非 Ω 位置，
  corr_dark  = 1 − mean|P_inj − P|[mask] / δ   （1=把注入的暗块"修掉"→ 按暗触发；0=不理 → 不触发）
  corr_light = 同法注入亮块（+δ）的对照（洞修复器应保留亮块 ≈0）
输出 eval/probe_trigger_lovo.csv
"""
import os, sys, csv
import cv2, numpy as np, torch
os.environ.setdefault('TORCH_HOME', '/home/zmy/data/cache/torch')
os.environ.setdefault('HF_HOME', '/home/zmy/data/cache/huggingface')

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
sys.path.insert(0, os.path.join(PROJ, 'RDNet'))
os.chdir(os.path.join(PROJ, 'RDNet'))
from finetune_rdnet_peft import build                     # noqa: E402
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
from eval_protocol import compute_mask                    # noqa: E402

DEV = torch.device('cuda:0')
SETS = {'may': (os.path.join(PROJ, 'data/test_sets/real_pair_may_0917'), 'GT.png'),
        'apr': (os.path.join(PROJ, 'data/test_sets/real_pair_apr_0917'), 'GT.png')}
FRAMES = [('may', 'VID_20260512_162203', 'I_000000'), ('may', 'VID_20260512_162623', 'I_000700'),
          ('may', 'VID_20260512_162901', 'I_000675'), ('may', 'VID_20260512_163859', 'I_000000'),
          ('may', 'VID_20260512_164653', 'I_000000'),
          ('apr', 'VID_20260426_173237', 'I_000000'), ('apr', 'VID_20260426_173604', 'I_000007')]
ARMS = [('official', None, 'none'),
        ('k2', 'runs/ckpt/rdt_s2_lovo_k2_s101/ckpt_10000.pth', 'both'),
        ('k4', 'runs/ckpt/rdt_s2_lovo_k4_s101/ckpt_10000.pth', 'both'),
        ('k6', 'runs/ckpt/rdt_s2_main_s101/ckpt_10000.pth', 'both')]
DEPTH = 30.0


def pad32(x):
    h, w = x.shape[:2]
    return cv2.copyMakeBorder(x, 0, (-h) % 32, 0, (-w) % 32, cv2.BORDER_REFLECT), h, w


def load_net(ckpt, mona):
    net, net_c = build(save_memory=True, mona=mona, grad_ckpt=False)
    if ckpt:
        ck = torch.load(os.path.join(PROJ, ckpt) if not os.path.isabs(ckpt) else ckpt, map_location='cpu')
        miss, _ = net.load_state_dict(ck['icnn'], strict=False)
        assert not [k for k in miss if 'mona' not in k], miss[:5]
    net.to(DEV).eval(); net_c.to(DEV)
    return net, net_c


def infer(net, net_c, I):
    Ip, h, w = pad32(I)
    x = torch.from_numpy(Ip[:, :, ::-1].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(DEV)
    with torch.no_grad():
        code = net_c(x)
        _, outs = net(x, code, prompt=True)
        T = outs[-1][:, 3:]
    T = T[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1]
    return (T[:h, :w] * 255).round().astype(np.uint8)


def blob_shape():
    """注入形状 = apr 173237 Ω 最大连通域，缩放到 ~180px 长边，软边"""
    d = SETS['apr'][0]
    I = cv2.imread(f'{d}/VID_20260426_173237/I_000000.png')
    T = cv2.imread(f'{d}/VID_20260426_173237/GT.png')
    M, _, _ = compute_mask(I, T)
    n, lab, st, _ = cv2.connectedComponentsWithStats(M.astype(np.uint8), 8)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[i][:4]
    m = (lab[y:y+h, x:x+w] == i).astype(np.float32)
    s = 200.0 / max(w, h)
    m = cv2.resize(m, (max(8, int(w*s)), max(8, int(h*s))), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (0, 0), 6.0)
    return (m - m.min()) / (m.max() - m.min() + 1e-9)


def pick_site(gray, mask_h, mask_w, avoid, h, w):
    """选非 Ω、局部亮度适中、离 Ω 远的落点"""
    best, bs = None, -1
    for _ in range(600):
        y0 = np.random.randint(0, h - mask_h); x0 = np.random.randint(0, w - mask_w)
        if avoid[y0:y0+mask_h, x0:x0+mask_w].mean() > 0.02:
            continue
        lm = gray[y0:y0+mask_h, x0:x0+mask_w].mean()
        if not (55 <= lm <= 150):
            continue
        s = lm
        if s > bs:
            bs, best = s, (y0, x0)
    return best


rows = []
BLOB = blob_shape()
for arm, ckpt, mona in ARMS:
    print(f'\n########## arm = {arm} ##########', flush=True)
    net, net_c = load_net(ckpt, mona)
    for setname, sc, fn in FRAMES:
      try:
          d = SETS[setname][0]
          I = cv2.imread(f'{d}/{sc}/{fn}.png')
          T = cv2.imread(f'{d}/{sc}/GT.png')
          M, _, _ = compute_mask(I, T)
          gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float32)
          mh, mw = BLOB.shape
          np.random.seed(20261008)
          site = pick_site(gray, mh, mw, M, I.shape[0], I.shape[1])
          if site is None:
              print(f'  [skip] {setname} {sc} {fn}: 无合适落点'); continue
          y0, x0 = site
          sub = I[y0:y0+mh, x0:x0+mw].astype(np.float32)
          ring = cv2.GaussianBlur(gray, (0, 0), 12)[y0:y0+mh, x0:x0+mw][BLOB < 0.1]
          d0 = float(ring.mean() - (sub * BLOB[..., None]).sum(axis=(0, 1)).mean() / (BLOB.sum() + 1e-9))
          delta = max(10.0, DEPTH - d0)
          mm = BLOB[..., None]
          I_dark = I.copy(); I_dark[y0:y0+mh, x0:x0+mw] = np.clip(sub - delta * mm, 0, 255).astype(np.uint8)
          I_lite = I.copy(); I_lite[y0:y0+mh, x0:x0+mw] = np.clip(sub + delta * mm, 0, 255).astype(np.uint8)
          P, Pd, Pl = infer(net, net_c, I), infer(net, net_c, I_dark), infer(net, net_c, I_lite)
          m3 = mm[..., 0] > 0.5
          def sl(img):
              return img[y0:y0+mh, x0:x0+mw].astype(np.float32)
          res_d = float(np.abs(sl(Pd) - sl(P)).mean(axis=2)[m3].mean())
          res_l = float(np.abs(sl(Pl) - sl(P)).mean(axis=2)[m3].mean())
          corr_d, corr_l = 1 - res_d / delta, 1 - res_l / delta
          g_dark = cv2.cvtColor(I_dark[y0:y0+mh, x0:x0+mw], cv2.COLOR_BGR2GRAY).astype(np.float32)
          depth1 = float(ring.mean() - g_dark[BLOB > 0.5].mean())
          rows.append(dict(arm=arm, set=setname, frame=f'{sc}__{fn[2:]}', site=f'{x0},{y0}',
                           delta=round(delta, 1), depth1=round(depth1, 1),
                           corr_dark=round(corr_d, 3), corr_light=round(corr_l, 3)))
          print(f'  {setname} {sc} {fn[2:]:<7} site=({x0},{y0}) delta={delta:4.0f}  corr_dark={corr_d:+.2f}  corr_light={corr_l:+.2f}', flush=True)
      except Exception as e:
        print(f'  [error] {setname} {sc} {fn}: {type(e).__name__}: {e}', flush=True)
    del net, net_c
    torch.cuda.empty_cache()

os.makedirs(os.path.join(PROJ, 'eval'), exist_ok=True)
with open(os.path.join(PROJ, 'eval/probe_trigger_lovo.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('\n===== 汇总（各臂 corr_dark 中位）=====')
for a, _, _ in ARMS:
    for sn in ('may', 'apr'):
        v = [r['corr_dark'] for r in rows if r['arm'] == a and r['set'] == sn]
        vl = [r['corr_light'] for r in rows if r['arm'] == a and r['set'] == sn]
        if v:
            print(f'  {a:<9} {sn}: corr_dark 中位 {np.median(v):+.2f} (n={len(v)})   corr_light 中位 {np.median(vl):+.2f}')
print('[写出] eval/probe_trigger_lovo.csv')
