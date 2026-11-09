#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 + N2 训练脚本（2026-10-31）：scripts/finetune_rdnet_s2.py main() 的 fork。

唯一实质改动 = **N2 hook**（有符号双字典两步近端输出校正，见 scripts/n2_dict_core.py）
挂在最终 RGB 透射输出之后。数据、采样比、九项损失定义与系数、调度、裁剪尺度、
存档逻辑全部沿用基线；基线脚本 `finetune_rdnet_s2.py` 一个字节未改。

本文件相对基线的差异（逐条，便于审计）
------------------------------------
1. 新增参数：--n2-mode {on,off} / --n2-dict / --n2-eta-init / --n2-k-inner /
   --n2-gradlog-iters / --n2-bridge-json / --save-mode {full,n2only}
2. 构造：net/net_c 建好后，从 **探针锁定的字典 JSON** 载入 N2Dict（脚本内不做 SVD），
   n2 参数并入同一个 Adam；**新增模块关闭（--n2-mode off）时执行路径与基线逐行一致**。
3. 前向：`for ci, out_col in enumerate(outs)` 里，**仅最后一列（ci == len(outs)-1）**
   的 T_hat 改读修正后的输出；其余三列（w_c=2,3,4 的深监督）保持原样。因此九项损失里
   只有"读最后一列透射输出"的项会给新增参数梯度（逐项梯度在 bridge.json 里记录）。
4. η 直接可训、初始化 0、**每个 optimizer step 后投影到 [0,1]**（审查要求；不用平方参数化，
   不用 sigmoid 截断）。η=0 ⇒ 输出逐像素等于基线输出。
5. 每 --log-every 记录 η、‖∂L/∂η‖、‖∂L/∂θ_dict‖（未缩放梯度）；在 --n2-gradlog-iters
   指定的一次或两次迭代上做**逐项损失 → 新增参数的梯度分解**（11 次 retain_graph 反传）。
6. ckpt 增加 `n2` 键；--save-mode n2only 只存 N2 权重 + args（冒烟用，避免 1 GB 全量存档）。
7. 收尾写 bridge.json：第二道"训练桥接"的证据（η 是否离零、字典是否拿到非零梯度、
   逐项有效梯度、峰值显存、时长、步数）。

用法（冒烟 100 it，bs4；**这只是冒烟，不是正式训练**）
  cd /DataDisk
  export HOME=/DataDisk/zmy_code/.home_zmy
  export TINTED_ALLOW_MISSING_TSRC=1
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \\
  /DataDisk/miniconda/envs/xreflection/bin/python3 \\
    /DataDisk/zmy_code/TintedGlass_Enhance/scripts/finetune_rdnet_s2_n2.py \\
    --n2-mode on --n2-dict runs/n2_1031/n2_dict_mayjun.json \\
    --omega-dir data/train_sets/real_pairs_1020 --omega-domains may,jun \\
    --arm main --seed 101 --iters 100 --batch 4 --seg-iters 34 --log-every 10 \\
    --out runs/n2_1031/smoke_s101 --save-mode n2only

正式命令（三种子配对，**不要现在启动**）见产物里的 run_cmd / report。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# 基线脚本（未改一个字节）：复用其 build / 数据集 / 九项损失的 helper
import finetune_rdnet_s2 as B                                  # noqa: E402
from n2_dict_core import N_PARAM_TOTAL, load_config, n2_from_config   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    # ---- 与基线**完全一致**的参数（缺一个都会改变数据/损失口径） ----
    ap.add_argument('--arm', choices=['main', 'b1'], default='main')
    ap.add_argument('--out', default=None)
    ap.add_argument('--iters', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--mona', default='both')
    ap.add_argument('--lam-r', type=float, default=1.5)
    ap.add_argument('--lam-recons', type=float, default=0.2)
    ap.add_argument('--lam-excl', type=float, default=0.1)
    ap.add_argument('--lam-om', type=float, default=3.0)
    ap.add_argument('--omega-mode', choices=['real', 'random', 'randomfixed', 'full'],
                    default='real')
    ap.add_argument('--lam-hinge', type=float, default=1.0)
    ap.add_argument('--lam-dark', type=float, default=0.0)
    ap.add_argument('--gate-mode', choices=['none', 'new', 'old'], default='none')
    ap.add_argument('--gate-beta', type=float, default=2.0)
    ap.add_argument('--gate-tau', type=float, default=0.06)
    ap.add_argument('--lam-shift', type=float, default=0.0)
    ap.add_argument('--alpha-shift', type=float, default=0.25)
    ap.add_argument('--lam-dists', type=float, default=0.2)
    ap.add_argument('--lam-grad', type=float, default=5.0)
    ap.add_argument('--lam-lap', type=float, default=2.0)
    ap.add_argument('--s-jitter', type=float, default=0.0)
    ap.add_argument('--omega-dir', default=None)
    ap.add_argument('--omega-domains', default=None)
    ap.add_argument('--manifest', default='')
    ap.add_argument('--assert-under', default='')
    ap.add_argument('--no-dists', action='store_true')
    ap.add_argument('--legacy-fidelity', action='store_true')
    ap.add_argument('--sample-p', default=None)
    ap.add_argument('--arch', choices=['mona', 'clean1029'], default='mona')
    ap.add_argument('--synth-occ', choices=['none', 'exact'], default='none')
    ap.add_argument('--synth-shape', choices=['s1', 's2', 'mix'], default='mix')
    ap.add_argument('--r-pool', default=None)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--seg-iters', type=int, default=2000)
    ap.add_argument('--resume', default='')
    # ---- N2 新增 ----
    ap.add_argument('--n2-mode', choices=['on', 'off'], default='off',
                    help='off = 与基线同路径（不含新增参数）；on = 挂 N2 输出校正')
    ap.add_argument('--n2-dict', default='',
                    help='探针锁定的字典 JSON（scripts/n2_probe_cpu.py build-dict 产出）')
    ap.add_argument('--n2-eta-init', type=float, default=0.0,
                    help='η 初值；审查要求 0（关闭态=基线），不得偷偷改大')
    ap.add_argument('--n2-k-inner', type=int, default=0,
                    help='覆盖字典 JSON 的 K（0=用 JSON 值）；改 K 必须与探针一致')
    ap.add_argument('--n2-gradlog-iters', default='1',
                    help='在这些迭代上做逐项损失→新增参数的梯度分解（逗号分隔）')
    ap.add_argument('--n2-bridge-json', default='',
                    help='bridge.json 路径（默认 <out>/bridge.json）')
    ap.add_argument('--save-mode', choices=['full', 'n2only'], default='full')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = args.out or os.path.join(PROJ, f'runs/ckpt/rdt_s2n2_{args.arm}_s{args.seed}')
    if not os.path.isabs(out):
        out = os.path.join(PROJ, out)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(f'cuda:{args.gpu}')

    # ---------------- 主干（与基线同） ----------------
    if args.arch == 'clean1029':
        os.environ['TINTED_ARCH'] = 'clean1029'
        net, net_c = B.build_clean1029()
    else:
        net, net_c = B.build(save_memory=True, mona=args.mona, grad_ckpt=True)
    for n, p in net.named_parameters():
        if n.startswith('baseball.'):
            p.requires_grad_(False)
    from rdnet_peft import set_mona_trainable
    n_mona = set_mona_trainable(net, True)
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'[s2n2] trainable {n_tr/1e6:.2f}M (mona {n_mona} 个) arm={args.arm}', flush=True)
    net.to(dev)
    net_c.to(dev)

    # ---------------- N2 模块（模块关闭时完全不存在，保证与基线同路径） ----------------
    n2 = None
    dict_meta = {}
    if args.n2_mode == 'on':
        if not args.n2_dict:
            raise SystemExit('[s2n2] --n2-mode on 需要 --n2-dict（探针锁定的字典 JSON）')
        dp = args.n2_dict if os.path.isabs(args.n2_dict) else os.path.join(PROJ, args.n2_dict)
        if not os.path.exists(dp):
            raise SystemExit(f'[s2n2] 缺字典文件 {dp}：先跑 scripts/n2_probe_cpu.py build-dict')
        n2 = n2_from_config(dp, eta_init=args.n2_eta_init,
                            k_inner=(args.n2_k_inner or None))
        n2.to(dev)
        _, _, _, dict_meta = load_config(dp)
        print(f'[s2n2] N2 {n2} | 新增参数 {sum(p.numel() for p in n2.parameters())} '
              f'(dict {n2.dt.numel()+n2.dr.numel()} + eta {n2.n_outer}) | dict={os.path.basename(dp)}',
              flush=True)
        print(f'[s2n2] 字典来源 meta: pool={dict_meta.get("meta", {}).get("pool")} '
              f'domains={dict_meta.get("meta", {}).get("domains")} '
              f'n_patch={dict_meta.get("meta", {}).get("n_patch")}', flush=True)

    import pyiqa
    if args.no_dists or args.lam_dists <= 0:
        dists_net = None
        print('[s2n2] DISTS 未启用（λ_dists=0 或 --no-dists）', flush=True)
    else:
        dists_net = pyiqa.create_metric('dists', device=dev).net
        dists_net.eval().requires_grad_(False)

    ds = B.S2Dataset(arm=args.arm, size=224, s_jitter=args.s_jitter,
                     omega_dir=args.omega_dir, domains=args.omega_domains,
                     sample_p=args.sample_p)
    ds.synth_occ = args.synth_occ
    ds.shape_mode = args.synth_shape
    ds.r_pool = args.r_pool
    if args.synth_occ != 'none' and B._SO_OK:
        ds.sil_lib = B._SO.load_lib()
        print(f'[s2n2] 合成遮挡已开: shape={args.synth_shape}', flush=True)
    elif args.synth_occ != 'none':
        raise SystemExit('--synth-occ 需要 synth_occlusion_v2 可用')
    print(f'[s2n2] T 源 {len(ds.t_files)} 张 | R 白名单 {len(ds.om_ids)}+{len(ds.rrw_ids)-1} 层 | '
          f'开源 {len(ds.os_pairs)} 对 | p={ds.p}', flush=True)

    if args.assert_under:
        root = os.path.realpath(args.assert_under)
        bad = []
        for case in ds.om_ids:
            for sub in ('I', 'T', 'R', 'omega'):
                p = os.path.realpath(os.path.join(ds.omega_dir, sub, case + '.png'))
                if not p.startswith(root + os.sep):
                    bad.append(p)
        if bad:
            raise SystemExit(f'[s2n2] 入口断言失败：{len(bad)} 个路径不在 {root} 下')
        print(f'[s2n2] 入口断言通过：{len(ds.om_ids)} 个 case 全在 {root} 下', flush=True)

    # 实验清单（与基线同格式；新增 N2 字段供消融归档）
    if args.manifest:
        import socket
        import subprocess
        mp = args.manifest if os.path.isabs(args.manifest) else os.path.join(PROJ, args.manifest)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        try:
            gitrev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=PROJ,
                                    capture_output=True, text=True).stdout.strip()
        except Exception:                                # noqa: BLE001
            gitrev = '?'
        man = dict(
            args=vars(args), git_rev=gitrev, host=socket.gethostname(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            torch=torch.__version__, cuda=torch.version.cuda,
            trainable_params=n_tr, mona_keys=n_mona, n2_new_params=n_new,
            n2=dict(k_inner=None if n2 is None else n2.k_inner,
                    n_outer=None if n2 is None else n2.n_outer,
                    h=None if n2 is None else n2.h,
                    tau=None if n2 is None else float(n2.tau),
                    dict=args.n2_dict),
            world_size=1, batch_per_gpu=args.batch, effective_batch=args.batch,
            omega_dir=ds.omega_dir, n_omega_cases=len(ds.om_ids),
            omega_domains=sorted({str(v.get('domain', '')) for v in ds.om_meta.values()}),
            omega_scenes=len({str(v.get('scene', '')) for v in ds.om_meta.values()}),
            n_t_src=len(ds.t_files), n_opensrc=len(ds.os_pairs), sample_p=ds.p.tolist(),
            start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        )
        with open(mp, 'w', encoding='utf-8') as fh:
            json.dump(man, fh, ensure_ascii=False, indent=1)
        print(f'[s2n2] manifest -> {mp}', flush=True)
        args._manifest_path = mp

    params = [p for p in net.parameters() if p.requires_grad]
    if n2 is not None:
        params += list(n2.parameters())
    n_new = 0 if n2 is None else sum(p.numel() for p in n2.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 200) * (0.1 + 0.45 *
                        (1 + math.cos(math.pi * min(i, args.iters) / args.iters))))
    scaler = torch.cuda.amp.GradScaler()
    start_it = 0
    if args.resume:
        ck_r = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ck_r['icnn'])
        if n2 is not None and ck_r.get('n2'):
            n2.load_state_dict(ck_r['n2'])
        opt.load_state_dict(ck_r['opt'])
        scaler.load_state_dict(ck_r['scaler'])
        start_it = int(ck_r.get('iter', 0))
        sched.last_epoch = start_it
        print(f'[s2n2] resume @ {start_it}', flush=True)

    logf = open(os.path.join(out, 'train_log.csv'), 'a')
    if start_it == 0:
        logf.write('iter,seg,L_t,L_r,L_om,L_hinge,L_shift,L_dists,L_recons,L_excl,'
                   'L_grad,L_lap,L_dark,total,sec,'
                   'eta0,eta1,g_eta_norm,g_dict_norm,delta_abs,frac_zero\n')
    segs = [(224, args.batch), (256, min(args.batch, 6)), (320, min(args.batch, 4))]
    gradlog_it = {int(x) for x in str(args.n2_gradlog_iters).split(',') if x.strip()}
    bridge = dict(n2_mode=args.n2_mode, dict=args.n2_dict, k_inner=None, n_outer=None,
                  h=None, tau=None, eta_init=args.n2_eta_init, n_new_param=n_new,
                  batch=args.batch, seg_iters=args.seg_iters, lr=args.lr, seed=args.seed,
                  eta_trace=[], g_eta_first=None, g_dict_norm_first=None,
                  g_dict_norm_max=0.0, per_term_grad_norm=None, terms_zero_grad=None,
                  eta_moved=False, dict_grad_nonzero=False, peak_mem_gb=None,
                  wall_sec=None, iters=0)
    if n2 is not None:
        bridge.update(k_inner=n2.k_inner, n_outer=n2.n_outer, h=n2.h, tau=float(n2.tau))

    it, t0, prev_seg = start_it, time.time(), None
    L_t = L_r = L_om = L_hinge = L_shift = L_dists = L_recons = L_excl = torch.zeros(())
    L_grad = L_lap = L_dark = total = torch.zeros(())
    while it < args.iters:
        seg_id = it // args.seg_iters
        s, b = segs[seg_id % len(segs)]
        if s != prev_seg:
            torch.cuda.reset_peak_memory_stats(dev)
            print(f'[s2n2] seg -> {s}² bs{b} @iter {it}', flush=True)
            prev_seg = s
        ds.size = s
        dl = torch.utils.data.DataLoader(ds, batch_size=b, shuffle=True, num_workers=4,
                                         drop_last=True, pin_memory=True)
        for I, T, R, OM, meta in dl:
            if it >= args.iters or it // args.seg_iters != seg_id:
                break
            I = I.to(dev, non_blocking=True)
            T = T.to(dev, non_blocking=True)
            R = R.to(dev, non_blocking=True)
            OM = OM.to(dev, non_blocking=True)
            meta = meta.to(dev, non_blocking=True)
            has_R = meta[:, 0].view(-1, 1, 1, 1)
            has_om = meta[:, 1].view(-1, 1, 1, 1)
            C_I = meta[:, 2].view(-1, 1, 1, 1)
            w_s = meta[:, 3].view(-1, 1, 1, 1)
            w_g = torch.ones_like(R[:, :1])
            if args.gate_mode != 'none':
                with torch.no_grad():
                    if args.gate_mode == 'new':
                        g_lab = (R.abs().amax(1, keepdim=True) /
                                 (args.gate_tau + B.EPS)).clamp(0.0, 1.0)
                    else:
                        gi = B.grad_mag(I).amax(1, keepdim=True)
                        gt = B.grad_mag(T).amax(1, keepdim=True)
                        g_lab = (gi > gt).float()
                    w_g = 1.0 + args.gate_beta * g_lab
            with torch.no_grad():
                code = net_c(I)
            with torch.cuda.amp.autocast():
                _, outs = net(I, code, prompt=True)
                L_t = L_r = L_om = L_hinge = L_dists = L_shift = torch.zeros((), device=dev)
                L_dark = torch.zeros((), device=dev)
                wsum = 0.0
                hw = float(R.shape[2] * R.shape[3])
                L_grad = L_lap = torch.zeros((), device=dev)
                n2_diag = {}
                T_last_corr = None
                for ci, out_col in enumerate(outs):
                    R_hat, T_hat = out_col[:, :3], out_col[:, 3:]
                    # >>> N2 hook：只改最后一列（部署输出）的透射项，其余列保持基线深监督
                    if n2 is not None and ci == len(outs) - 1:
                        T_hat, n2_diag = n2(I, T_hat, return_diag=True)
                        T_last_corr = T_hat
                    # <<< N2 hook
                    wc = float(ci + 1)
                    wsum += wc
                    L_t = L_t + wc * (w_s * w_g * B.charbonnier(T_hat, T)).mean()
                    _ws = w_s.reshape(-1)
                    if args.lam_grad > 0:
                        if args.legacy_fidelity:
                            L_grad = L_grad + args.lam_grad * wc * B.grad_fidelity(T_hat, T)
                        else:
                            L_grad = L_grad + args.lam_grad * wc * (
                                _ws * B.grad_fidelity_elem(T_hat, T)).mean()
                    if args.lam_lap > 0:
                        if args.legacy_fidelity:
                            L_lap = L_lap + args.lam_lap * wc * B.lap_fidelity(T_hat, T)
                        else:
                            L_lap = L_lap + args.lam_lap * wc * (
                                _ws * B.lap_fidelity_elem(T_hat, T)).mean()
                    if has_R.max() > 0:
                        num = (w_s * w_g * B.charbonnier(R_hat, R) * has_R).sum()
                        den_r = (w_s * w_g * has_R).sum() * hw + B.EPS
                        L_r = L_r + args.lam_r * wc * num / den_r
                L_t = L_t / wsum
                L_r = L_r / wsum
                L_grad = L_grad / wsum
                L_lap = L_lap / wsum
                R_hat, T_hat = outs[-1][:, :3], outs[-1][:, 3:]
                if T_last_corr is not None:
                    T_hat = T_last_corr
                L_recons = args.lam_recons * (w_s * (I - T_hat - R_hat).abs()).mean()
                L_excl = args.lam_excl * B.grad_excl(T_hat, R_hat)
                if has_om.max() > 0:
                    if args.omega_mode == 'random':
                        OM = B.area_matched_random_mask(OM)
                    elif args.omega_mode == 'randomfixed':
                        OM = B.area_matched_random_mask(OM, deterministic=True)
                    elif args.omega_mode == 'full':
                        OM = torch.ones_like(OM)
                    om_sum = OM.sum(dim=(1, 2, 3)) + 1e-6
                    err = (OM * (T_hat - T).abs()).sum(dim=(1, 2, 3)) / om_sum
                    den = ((OM * (I - T).abs()).sum(dim=(1, 2, 3)) / om_sum).clamp(min=1e-3)
                    m_om = has_om.squeeze()
                    L_om = args.lam_om * (err / den * m_om).sum() / m_om.sum().clamp(min=1)
                    delta = (torch.clamp(0.5 * C_I, min=4.0) / 255.0)
                    hg = (OM * torch.nn.functional.relu(T - delta - T_hat)).sum(dim=(1, 2, 3)) / om_sum
                    L_hinge = args.lam_hinge * (hg / den * m_om).sum() / m_om.sum().clamp(min=1)
                    if args.lam_dark > 0:
                        mt = (OM * T).sum(dim=(1, 2, 3)) / om_sum
                        mh = (OM * T_hat).sum(dim=(1, 2, 3)) / om_sum
                        L_dark = args.lam_dark * ((((mh - mt).abs()) / (mt.abs() + B.EPS)) *
                                                  m_om).sum() / m_om.sum().clamp(min=1)
                    if args.lam_shift > 0:
                        OMd = torch.nn.functional.max_pool2d(OM, 3, 1, 1)
                        OMe = -torch.nn.functional.max_pool2d(-OM, 3, 1, 1)
                        bd = (OMd - OMe).clamp(min=0.0)
                        bd_sum = bd.sum(dim=(1, 2, 3)) + 1e-6
                        gT = B.grad_mag(T_hat)
                        gR = B.grad_mag(R_hat)
                        num_s = (bd * torch.nn.functional.relu(
                            gT - args.alpha_shift * gR)).sum(dim=(1, 2, 3)) / bd_sum
                        den_s = ((bd * gR).sum(dim=(1, 2, 3)) / bd_sum).clamp(min=1e-3)
                        L_shift = args.lam_shift * ((num_s / den_s) * m_om).sum() / \
                            m_om.sum().clamp(min=1)
                    idxs = (m_om > 0).nonzero().squeeze(-1)
                    dsum = torch.zeros((), device=dev)
                    if args.lam_dists > 0:
                        with torch.cuda.amp.autocast(enabled=False):
                            for bi in idxs[:4]:
                                m = OM[bi, 0] > 0.5
                                ys, xs = m.any(1).nonzero(), m.any(0).nonzero()
                                y0, y1 = max(0, int(ys.min()) - 10), int(ys.max()) + 11
                                x0, x1 = max(0, int(xs.min()) - 10), int(xs.max()) + 11
                                ta = T_hat[bi:bi + 1, :, y0:y1, x0:x1].float()
                                tb = T[bi:bi + 1, :, y0:y1, x0:x1].float()
                                sc = 320.0 / max(ta.shape[2], ta.shape[3])
                                if sc < 1:
                                    ta = torch.nn.functional.interpolate(
                                        ta, scale_factor=sc, mode='bilinear', align_corners=False)
                                    tb = torch.nn.functional.interpolate(
                                        tb, scale_factor=sc, mode='bilinear', align_corners=False)
                                dsum = dsum + args.lam_dists * dists_net(ta, tb)
                        L_dists = dsum / max(1, min(len(idxs), 4))
                total = (L_t + L_r + L_om + L_hinge + L_shift + L_dists + L_recons
                         + L_excl + L_grad + L_lap + L_dark)

            # ---- 逐项损失 → 新增参数的梯度分解（只在指定迭代；retain_graph） ----
            if n2 is not None and it in gradlog_it:
                terms = dict(L_t=L_t, L_r=L_r, L_om=L_om, L_hinge=L_hinge, L_shift=L_shift,
                             L_dists=L_dists, L_recons=L_recons, L_excl=L_excl,
                             L_grad=L_grad, L_lap=L_lap, L_dark=L_dark)
                plist = [n2.dt, n2.dr, n2.eta]
                gnorm, zero_terms = {}, []
                for nm, val in terms.items():
                    try:
                        vs = float(val.detach())
                    except Exception:                        # noqa: BLE001
                        vs = 0.0
                    if not val.requires_grad:
                        zero_terms.append(nm + "(无计算图)")
                        continue
                    gs = torch.autograd.grad(val, plist, retain_graph=True, allow_unused=True)
                    tot = 0.0
                    for g in gs:
                        if g is not None:
                            tot += float(g.abs().sum())
                    gnorm[nm] = tot
                    if tot == 0.0:
                        zero_terms.append(nm)
                    if vs == 0.0:
                        zero_terms.append(nm + "(值=0)")
                bridge['per_term_grad_norm'] = bridge.get('per_term_grad_norm') or {}
                bridge['per_term_grad_norm'][str(it)] = gnorm
                bridge['terms_zero_grad'] = bridge.get('terms_zero_grad') or {}
                bridge['terms_zero_grad'][str(it)] = sorted(set(zero_terms))
                print(f"[s2n2] gradlog @it {it}: " +
                      ' '.join(f'{k}={v:.3e}' for k, v in gnorm.items()), flush=True)
                print(f"[s2n2] 零梯度项: {bridge['terms_zero_grad'][str(it)]}", flush=True)

            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            if it % args.log_every == 0 or it == 1:
                if n2 is not None:
                    ge = n2.eta.grad
                    gd = (n2.dt.grad.abs().sum() + n2.dr.grad.abs().sum()) \
                        if (n2.dt.grad is not None and n2.dr.grad is not None) else None
                    bridge['g_dict_norm_max'] = max(
                        bridge['g_dict_norm_max'], float(gd) if gd is not None else 0.0)
                    if it == 1:
                        bridge['g_eta_first'] = None if ge is None else \
                            [float(x) for x in ge.detach().cpu()]
                        bridge['g_dict_norm_first'] = None if gd is None else float(gd)
            scaler.step(opt)
            scaler.update()
            if n2 is not None:                            # η 投影到 [0,1]（审查要求）
                with torch.no_grad():
                    n2.eta.clamp_(0.0, 1.0)
            sched.step()
            it += 1
            if it % args.log_every == 0 or it == 1:
                sec = time.time() - t0
                g = lambda x: float(x.detach())          # noqa: E731
                eta_s = '' if n2 is None else \
                    f'eta={" ".join(f"{x:.4f}" for x in n2.eta.detach().cpu())} ' \
                    f'g_eta={0.0 if n2.eta.grad is None else float(n2.eta.grad.abs().sum()):.3e} ' \
                    f'g_dict={bridge["g_dict_norm_max"]:.3e} '
                diag_s = '' if not n2_diag else \
                    f'|δ|={n2_diag["delta_abs_mean"]:.4f} zero={n2_diag["frac_both_zero"]:.3f} '
                print(f'[s2n2] it {it:5d} L_t={g(L_t):.4f} L_r={g(L_r):.4f} om={g(L_om):.3f} '
                      f'hinge={g(L_hinge):.3f} shift={g(L_shift):.4f} dists={g(L_dists):.3f} '
                      f'rcn={g(L_recons):.4f} excl={g(L_excl):.3f} grad={g(L_grad):.3f} '
                      f'lap={g(L_lap):.3f} dark={g(L_dark):.4f} total={g(total):.4f} {sec:.0f}s '
                      f'peak={torch.cuda.max_memory_allocated(dev)/2**30:.2f}G {eta_s}{diag_s}',
                      flush=True)
                eta0, eta1 = (float('nan'), float('nan')) if n2 is None else \
                    tuple(float(x) for x in n2.eta.detach().cpu()[:2])
                logf.write(f'{it},{s},{g(L_t):.5f},{g(L_r):.5f},{g(L_om):.5f},{g(L_hinge):.5f},'
                           f'{g(L_shift):.5f},{g(L_dists):.5f},{g(L_recons):.5f},{g(L_excl):.5f},'
                           f'{g(L_grad):.5f},{g(L_lap):.5f},{g(L_dark):.5f},{g(total):.5f},'
                           f'{sec:.0f},{eta0:.6f},{eta1:.6f},'
                           f'{0.0 if (n2 is None or n2.eta.grad is None) else float(n2.eta.grad.abs().sum()):.3e},'
                           f'{bridge["g_dict_norm_max"]:.3e},'
                           f'{n2_diag.get("delta_abs_mean", 0.0):.6f},'
                           f'{n2_diag.get("frac_both_zero", 0.0):.6f}\n')
                logf.flush()
                if n2 is not None:
                    bridge['eta_trace'].append(dict(it=it, eta=[float(x) for x in
                                                                n2.eta.detach().cpu()]))
                    if max(abs(float(x)) for x in n2.eta.detach().cpu()) > 1e-6:
                        bridge['eta_moved'] = True
                    if bridge['g_dict_norm_max'] > 0.0:
                        bridge['dict_grad_nonzero'] = True
            if it % args.save_every == 0:
                _save(args, out, net, n2, opt, scaler, it)
    _save(args, out, net, n2, opt, scaler, it, final=True)
    print(f'[s2n2] done {it} iter -> {out}', flush=True)

    # ---- bridge.json：第二道闸的证据 ----
    bridge['iters'] = it
    bridge['peak_mem_gb'] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 3)
    bridge['wall_sec'] = round(time.time() - t0, 1)
    if n2 is not None:
        bridge['eta_final'] = [float(x) for x in n2.eta.detach().cpu()]
        bridge['dict_abs_mean'] = float(n2.dt.abs().mean() + n2.dr.abs().mean())
    need = 200
    if n2 is None:
        bridge['bridge_pass'] = False
        bridge['bridge_reason'] = 'n2_mode=off：没有新增参数，第二道不适用'
    elif it < need:
        bridge['bridge_pass'] = False
        bridge['bridge_reason'] = f'步数 {it} < 预登记 {need}：只能算部分证据'
    elif not bridge['eta_moved']:
        bridge['bridge_pass'] = False
        bridge['bridge_reason'] = 'η 始终为 0（梯度被投影压回或恒为负）⇒ I1 式证据断裂'
    elif not bridge['dict_grad_nonzero']:
        bridge['bridge_pass'] = False
        bridge['bridge_reason'] = '字典梯度始终为 0 ⇒ 模块无法离开关闭态'
    else:
        bridge['bridge_pass'] = True
        bridge['bridge_reason'] = 'η 离零且字典拿到非零梯度（仍不保证 6000 步成功）'
    bj = args.n2_bridge_json or os.path.join(out, 'bridge.json')
    if not os.path.isabs(bj):
        bj = os.path.join(PROJ, bj)
    with open(bj, 'w', encoding='utf-8') as fh:
        json.dump(bridge, fh, ensure_ascii=False, indent=1)
    print(f'[s2n2] bridge -> {bj} | pass={bridge["bridge_pass"]} ({bridge["bridge_reason"]})',
          flush=True)

    # 清单收尾（与基线同口径：结束时间/时长/末次损失/峰值显存）
    mp = getattr(args, '_manifest_path', '')
    if mp and os.path.exists(mp):
        man = json.load(open(mp, encoding='utf-8'))
        g = lambda x: float(x.detach())                  # noqa: E731
        man.update(end_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                   wall_sec=round(time.time() - t0, 1), final_iter=it,
                   ckpt=os.path.join(out, 'ckpt_final.pth'),
                   loss_last=dict(L_t=g(L_t), L_r=g(L_r), L_om=g(L_om), L_hinge=g(L_hinge),
                                  L_dists=g(L_dists), L_recons=g(L_recons), L_excl=g(L_excl),
                                  L_grad=g(L_grad), L_lap=g(L_lap), L_dark=g(L_dark),
                                  total=g(total)),
                   bridge=dict(pass_=bridge['bridge_pass'], reason=bridge['bridge_reason'],
                               eta_final=bridge.get('eta_final')),
                   peak_mem_gb=round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2))
        with open(mp, 'w', encoding='utf-8') as fh:
            json.dump(man, fh, ensure_ascii=False, indent=1)
        print(f'[s2n2] manifest 收尾 -> {mp}', flush=True)


def _save(args, out, net, n2, opt, scaler, it, final=False):
    payload = dict(icnn=net.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                   iter=it, peft=dict(mode=args.mona), arm=args.arm, args=vars(args))
    if n2 is not None:
        payload['n2'] = n2.state_dict()
        payload['n2_cfg'] = dict(k_inner=n2.k_inner, n_outer=n2.n_outer, h=n2.h,
                                 tau=float(n2.tau), n_param=N_PARAM_TOTAL)
    if args.save_mode == 'n2only':
        payload.pop('icnn', None)
        payload.pop('opt', None)
        payload.pop('scaler', None)
    p = os.path.join(out, 'ckpt_final.pth' if final else f'ckpt_{it}.pth')
    torch.save(payload, p)
    if not final:
        torch.save(payload, os.path.join(out, 'ckpt_latest.pth'))
    print(f'[s2n2] saved {os.path.basename(p)} (save_mode={args.save_mode})', flush=True)


if __name__ == '__main__':
    main()
