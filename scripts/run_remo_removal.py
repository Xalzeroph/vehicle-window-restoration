#!/usr/bin/env python3
"""RemoVal（CVPR2024 Zhu et al.）去反射推理壳（2026-09-19 轮，第二方法臂）。

复用官方 testing.py 的模型构造与权重口径（RR.pth 去除 + RD.pth 检测），
但不走其 dataset（my_dataset_eval 会把非 16 倍数尺寸 resize 且要求 GT 配对）：
自读图 → RGB [0,1] → net_Det → net，原生尺寸（网络内部 check_image_size 补边）→ 存 PNG。

用法：
  python scripts/run_remo_removal.py --set data/test_sets/real_pair_0917 --layout jun \
      --out runs/pred/real_pair_0917_remo
  python scripts/run_remo_removal.py --flat <img_dir> --out <pred_dir>   # 平铺（OpenRR 移植用）
"""
import argparse, os, sys
import cv2
import numpy as np

REMO_ROOT = '/DataDisk/zmy_code/Reflection_RemoVal_CVPR2024-main'
CKPT_RR = f'{REMO_ROOT}/ckpt/RR.pth'
CKPT_RD = f'{REMO_ROOT}/ckpt/RD.pth'


def load_model(ckpt_rr=CKPT_RR, ckpt_rd=CKPT_RD, device_str='cuda:0'):
    import torch
    sys.path.insert(0, REMO_ROOT)
    from networks.NAFNet_arch import NAFNet_wDetHead
    from networks.network_RefDet import RefDet

    net = NAFNet_wDetHead(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28],
                          dec_blk_nums=[1, 1, 1, 1], global_residual=False, drop_flag=False,
                          drop_rate=0.4, concat=True, merge_manner=0)
    net_Det = RefDet(backbone='efficientnet-b3', proj_planes=16, pred_planes=32,
                     use_pretrained=False, fix_backbone=False, has_se=False,
                     num_of_layers=6, expansion=4)

    def _load(m, p):
        try:
            sd = torch.load(p, map_location='cpu', weights_only=True)
        except Exception:
            sd = torch.load(p, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd and not any(
                k.startswith('intro') or k.startswith('encoders') for k in sd):
            sd = sd['state_dict']
        m.load_state_dict(sd, strict=True)

    _load(net, ckpt_rr)
    _load(net_Det, ckpt_rd)
    dev = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    net.to(dev).eval()
    net_Det.to(dev).eval()
    return net, net_Det, dev


def infer(net, net_Det, img_bgr, dev, pad=16):
    """img_bgr: uint8 HxWx3（BGR）→ 去反射输出 uint8 BGR（原生尺寸）。

    先 reflect 补边到 16 倍数（=NAFNet_wDetHead.padder_size）：主分支会按 padder 补边、
    检测分支不会，原生尺寸直接进会 concat 尺寸错位；补边后两分支同尺寸，输出裁回原尺寸。
    """
    import torch
    h, w = img_bgr.shape[:2]
    H = (h + pad - 1) // pad * pad
    W = (w + pad - 1) // pad * pad
    if (H, W) != (h, w):
        img_bgr = cv2.copyMakeBorder(img_bgr, 0, H - h, 0, W - w, cv2.BORDER_REFLECT_101)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        sparse = net_Det(t)
        out = net(t, sparse)
    o = torch.clamp(out, 0, 1)[0].permute(1, 2, 0).cpu().numpy()
    o = (o * 255).round().astype(np.uint8)[:h, :w]
    return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)


def collect(args):
    """返回 [(输入图路径, 输出图路径)]。"""
    jobs = []
    if args.flat:
        for fn in sorted(os.listdir(args.flat)):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                jobs.append((os.path.join(args.flat, fn),
                             os.path.join(args.out, os.path.splitext(fn)[0] + '.png')))
        return jobs
    for sc in sorted(d for d in os.listdir(args.set) if d.startswith('VID_')):
        sd = os.path.join(args.set, sc)
        if args.layout == 'jun':
            jobs.append((os.path.join(sd, 'I.png'), os.path.join(args.out, sc, 'remo.png')))
        else:
            for fn in sorted(f for f in os.listdir(sd) if f.startswith('I_')):
                jobs.append((os.path.join(sd, fn),
                             os.path.join(args.out, f'{sc}__{fn[:-4]}', 'remo.png')))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='data/test_sets/real_pair_0917')
    ap.add_argument('--layout', choices=['jun', 'multi'], default='jun')
    ap.add_argument('--flat', default='')
    ap.add_argument('--out', default='runs/pred/real_pair_0917_remo')
    ap.add_argument('--ckpt-rr', default=CKPT_RR)
    ap.add_argument('--ckpt-rd', default=CKPT_RD)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    jobs = collect(args)
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        print('无输入图，退出'); sys.exit(1)

    net, net_Det, dev = load_model(args.ckpt_rr, args.ckpt_rd, args.device)
    print(f'模型加载完成 device={dev}，共 {len(jobs)} 张')
    for ip, op in jobs:
        img = cv2.imread(ip)
        if img is None:
            print(f'{ip}: 读取失败 -> 跳过'); continue
        out = infer(net, net_Det, img, dev)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        cv2.imwrite(op, out)
        print(f'{os.path.relpath(op, args.out)}  {img.shape[1]}x{img.shape[0]}')


if __name__ == '__main__':
    main()
