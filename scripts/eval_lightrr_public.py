#!/usr/bin/env python3
"""LightRR 官方权重在公共基准上的复现（2026-10-12）

协议
----
**严格照抄 LightRR 自己的评测路径**，以便与 README 报告数字对表：
  - `--if_align` 开：输入与真值都 resize 到 (h//32*32, w//32*32)（默认 PIL 最近邻）；
  - 指标：skimage PSNR/SSIM，`data_range=255`，**逐通道算再取通道均值**
    （照抄 `util/index.py` 的 `Bandwise`）；
  - 输出 clip 到 [0,1] 再 ×255（照抄 `util/util.py` 的 `tensor2im`）。
同时报一个 **identity 基线**（什么都不做），因为全图 PSNR 会让"不动"赢过很多方法，
没有 identity 的绝对数字无法判断方法是否真的有用。

与官方 `eval_sirs.py` 的差异（有意为之）
--------------------------------------
1. 官方把 `base_dir` 硬编码为 `./datasets_link/reflection-removal`，且**只评 Wild**；
2. 官方依赖 `data/real_test.txt`，**该文件在仓库里不存在** → 这里改为目录列举；
3. 这里四个集全评，并输出逐图 CSV。

用法
----
python3 scripts/eval_lightrr_public.py \
    --repo /home/zmy/mycode/LightRR-CVPR26 \
    --ckpt /home/zmy/mycode/LightRR-CVPR26/weights/lightrr/lightrr_woNature.pt \
    --data-root /DataDisk/zmy_code/datasets \
    --out <proj>/eval/lightrr_public_1012.csv \
    --device cuda:2
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]


def to_tensor(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).contiguous()


def tensor_to_uint8_f(t: torch.Tensor) -> np.ndarray:
    """照抄 util/util.py tensor2im（保留 float 便于算指标）。"""
    a = t.detach()[0].cpu().float().numpy()
    a = np.clip(a, 0, 1)
    return np.transpose(a, (1, 2, 0)) * 255.0


def bw_psnr(x, y):
    return float(np.mean([compare_psnr(x[..., c], y[..., c], data_range=255)
                          for c in range(x.shape[-1])]))


def bw_ssim(x, y):
    return float(np.mean([compare_ssim(x[..., c], y[..., c], data_range=255)
                          for c in range(x.shape[-1])]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/home/zmy/mycode/LightRR-CVPR26')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-root', default='/DataDisk/zmy_code/datasets')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--if-align', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--resize', default='real20=420',
                    help='按集缩小长边，格式 name=px(,name=px)。real20 必须缩：'
                         'CrossChannelAtt 是全局注意力，O((H*W)^2)，'
                         '1114x717 仅注意力矩阵就要约 25 GB。论文用 real20_420。')
    ap.add_argument('--datasets', default='real20:real/real20,'
                                          'postcard:postcard/postcard199,'
                                          'objects:SIR2/SIR,'
                                          'wild:SIR2/wild')
    args = ap.parse_args()

    resize_map = {}
    for kv in filter(None, args.resize.split(',')):
        k, v = kv.split('=')
        resize_map[k.strip()] = int(v)

    repo = os.path.abspath(args.repo)
    os.chdir(repo)
    sys.path.insert(0, repo)

    from models.arch import LightRR_9M

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = LightRR_9M().to(dev).eval()
    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck['weight'] if isinstance(ck, dict) and 'weight' in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[ckpt] {args.ckpt}')
    print(f'[ckpt] epoch={ck.get("epoch") if isinstance(ck, dict) else None} '
          f'iters={ck.get("iterations") if isinstance(ck, dict) else None}')
    print(f'[ckpt] missing={len(missing)} unexpected={len(unexpected)}')
    print(f'[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f} M  dev={dev}')

    rows, summary = [], []
    for item in args.datasets.split(','):
        name, rel = item.split(':', 1)
        d = os.path.join(args.data_root, rel)
        bdir, tdir = os.path.join(d, 'blended'), os.path.join(d, 'transmission_layer')
        if not (os.path.isdir(bdir) and os.path.isdir(tdir)):
            print(f'[skip] {name}: 目录不符 {d}')
            continue
        fns = sorted((f for f in os.listdir(bdir)
                      if os.path.exists(os.path.join(tdir, f))), key=natural_sort_key)
        if args.limit:
            fns = fns[:args.limit]
        print(f'\n=== {name}  n={len(fns)}  {d}')

        acc, acc_id, t0 = [], [], time.time()
        for i, fn in enumerate(fns):
            t_img = Image.open(os.path.join(tdir, fn)).convert('RGB')
            m_img = Image.open(os.path.join(bdir, fn)).convert('RGB')
            if m_img.size != t_img.size:
                m_img = m_img.resize(t_img.size)
            # 长边缩到 420（real20），照 LightRR 的 real20_420 协议
            tgt = resize_map.get(name)
            if tgt and max(t_img.size) != tgt:
                ow, oh = t_img.size
                if ow >= oh:
                    nw, nh = tgt, max(2, int(round(tgt * oh / ow)))
                else:
                    nh, nw = tgt, max(2, int(round(tgt * ow / oh)))
                t_img = t_img.resize((nw, nh), Image.BICUBIC)
                m_img = m_img.resize((nw, nh), Image.BICUBIC)
            if args.if_align:
                h, w = t_img.height // 32 * 32, t_img.width // 32 * 32
                if (h, w) != (t_img.height, t_img.width):
                    t_img = t_img.resize((w, h))
                    m_img = m_img.resize((w, h))

            T = tensor_to_uint8_f(to_tensor(t_img))
            M = tensor_to_uint8_f(to_tensor(m_img))
            x = to_tensor(m_img).to(dev)
            with torch.no_grad():
                outs, _ = model(x)
            P = tensor_to_uint8_f(outs[-1].clamp(0, 1))

            ps, ss = bw_psnr(P, T), bw_ssim(P, T)
            pi, si = bw_psnr(M, T), bw_ssim(M, T)
            acc.append((ps, ss))
            acc_id.append((pi, si))
            rows.append({'dataset': name, 'fn': fn, 'h': T.shape[0], 'w': T.shape[1],
                         'psnr': round(ps, 4), 'ssim': round(ss, 4),
                         'psnr_identity': round(pi, 4), 'ssim_identity': round(si, 4),
                         'gain_psnr': round(ps - pi, 4)})
            if (i + 1) % 50 == 0 or i == len(fns) - 1:
                el = time.time() - t0
                print(f'  {i+1}/{len(fns)}  {el/(i+1):.3f}s/img  '
                      f'PSNR(so far)={np.mean([a for a,_ in acc]):.2f}')

        el = time.time() - t0
        mp = float(np.mean([a for a, _ in acc])); ms = float(np.mean([b for _, b in acc]))
        ip = float(np.mean([a for a, _ in acc_id])); is_ = float(np.mean([b for _, b in acc_id]))
        summary.append({'dataset': name, 'n': len(fns), 'psnr': mp, 'ssim': ms,
                        'psnr_id': ip, 'ssim_id': is_, 'gain': mp - ip,
                        'spi': el / max(len(fns), 1)})
        print(f'  -> LightRR {mp:.2f}/{ms:.4f}   identity {ip:.2f}/{is_:.4f}   '
              f'gain {mp-ip:+.2f} dB   {el/max(len(fns),1):.3f}s/img')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['dataset'])
        w.writeheader(); w.writerows(rows)
    print(f'\n[write] {args.out}  ({len(rows)} rows)')

    print('\n' + '=' * 96)
    print(f'{"dataset":<11}{"n":>6}{"PSNR":>8}{"SSIM":>9}{"PSNR(id)":>10}'
          f'{"SSIM(id)":>10}{"gain(dB)":>10}{"s/img":>8}')
    print('-' * 96)
    for s in summary:
        print(f'{s["dataset"]:<11}{s["n"]:>6}{s["psnr"]:>8.2f}{s["ssim"]:>9.4f}'
              f'{s["psnr_id"]:>10.2f}{s["ssim_id"]:>10.4f}{s["gain"]:>+10.2f}{s["spi"]:>8.3f}')
    if summary:
        print('-' * 96)
        print(f'{"AVG":<11}{"":>6}{np.mean([s["psnr"] for s in summary]):>8.2f}'
              f'{np.mean([s["ssim"] for s in summary]):>9.4f}'
              f'{np.mean([s["psnr_id"] for s in summary]):>10.2f}'
              f'{np.mean([s["ssim_id"] for s in summary]):>10.4f}'
              f'{np.mean([s["gain"] for s in summary]):>+10.2f}')
    print('\nREADME(w/o Nature) 报告: Real20 24.48/0.841  Objects 26.74/0.921  '
          'PostCard 24.91/0.907  Wild 26.71/0.912  Avg 25.88/0.911')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
