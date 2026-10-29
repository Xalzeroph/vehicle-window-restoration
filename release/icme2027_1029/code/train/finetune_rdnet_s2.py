#!/usr/bin/env python3
"""S2 训练脚本（"修掉人影"）：RDNet + Mona 适配器，真实 Ω 配对 + 在线合成 + 开源直督。

数据（采样比 3:4:3，--arm b1 时 0:57:43）:
  omega   data/train_sets/train_pairs_omega_0920   apr 16 对（I/T/R有符号/Ω/C_I）
  synth   在线合成 T'=j·T, I'=j·(T+b·R)；a≡1 全加性；b~logU[0.4,1.5]；j~U[0.5,1]；
          R 白名单 = apr16 + RRW9（data/pools/r_signed_0920/rrw）；T 源 = VOC_s320/SIR2/nature/scenes/RRW-GT；
          R 高斯模糊增广（kernel{5,7,9,11} p=.1/.2/.3/.4，σ∈[2,5]）
  opensrc data/train_sets/train_pairs_opensrc_0920/meta.json（OpenRR+SIR2 直督，整体 loss×0.5）

损失（4 列输出 w_c=c+1）:
  L = Σ_c w_c[ w⊙Charb(T̂,T) + 1.5·w⊙Charb(R̂,R)|has_R ] + 0.2·w⊙|I−T̂−R̂| + 0.1·exclusion
    + omega: 3·w⊙(M_Ω⊙|T̂−T|)/den_Ω + 1.0·w⊙(M_Ω⊙relu(T−δ−T̂))/den_Ω + 0.2·DISTS(T̂,T)|_Ω窗
    其中 den_Ω = mean(M_Ω⊙|I−T|)（ρ 型归一化），δ = max(4, 0.5|C_I|)/255（过冲 hinge）
  感知项 DISTS（与判据② pyiqa-LPIPS 异族）；选模以 ①④ 为纲。

用法:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/finetune_rdnet_s2.py --arm main --seed 101 --gpu 0 \
      --out runs/ckpt/rdt_s2_main_s101 --iters 10000 --batch 6
"""
import argparse, csv, json, math, os, sys, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import hashlib
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "RDNet"))
os.chdir(os.path.join(PROJ, "RDNet"))
from finetune_rdnet_peft import build, charbonnier      # noqa: E402

EPS = 1e-3
OMEGA_DIR = os.path.join(PROJ, "data/train_sets/train_pairs_omega_0920")
OPENSRC_META = os.path.join(PROJ, "data/train_sets/train_pairs_opensrc_0920/meta.json")
RRW_R_DIR = os.path.join(PROJ, "data/pools/r_signed_0920/rrw")
T_SRCS = [os.path.join(PROJ, "data/external/VOC2012/VOC2012_s320"),
          "/home/zmy/mycode/datasets/SIR2/SIR/transmission_layer",
          "/home/zmy/mycode/datasets/nature/natural_T",
          "/home/zmy/vehicle_ssd/datasets/scenes",
          "/DataDisk/zmy_code/datasets/RRWDataset/reflection_camera1/GT_camera1"]


def load_rgb(path, size, rng, jitter_crop=None):
    """读图 -> [0,1] float32 (3,s,s)。jitter_crop=(y0,x0) 时按指定位置裁（多图同步）。"""
    img = cv2.imread(path)
    if img is None:
        raise IOError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if min(h, w) < size:
        sc = size / min(h, w)
        img = cv2.resize(img, (max(size, round(w * sc)), max(size, round(h * sc))),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    if jitter_crop is None:
        y0, x0 = rng.integers(0, h - size + 1), rng.integers(0, w - size + 1)
    else:
        y0, x0 = jitter_crop
    y0, x0 = int(min(y0, h - size)), int(min(x0, w - size))
    return img[y0:y0 + size, x0:x0 + size].astype(np.float32) / 255., (y0, x0)


def synth_R(rng, omega_dir):
    """从白名单 R 池取一张并解码为有符号 [-0.5,0.5]（uint8=+128 存储）。"""
    f = rng.choice(os.listdir(RRW_R_DIR)) if rng.random() < 0.36 else None
    if f is None:
        ids = os.listdir(os.path.join(omega_dir, "R"))
        f2 = rng.choice(ids)
        img = cv2.imread(os.path.join(omega_dir, "R", f2))[:, :, ::-1]
    else:
        img = cv2.imread(os.path.join(RRW_R_DIR, f))[:, :, ::-1]
    return img.astype(np.float32) / 255.0 - 128.0 / 255.0


def blur_R(R, rng):
    ks = int(rng.choice([5, 7, 9, 11], p=[0.1, 0.2, 0.3, 0.4]))
    sigma = float(rng.uniform(2, 5))
    return cv2.GaussianBlur(R, (ks, ks), sigma)


class S2Dataset(Dataset):
    def __init__(self, arm='main', size=224, s_jitter=0.0, omega_dir=None, domains=None,
                 sample_p=None):
        self.omega_dir = omega_dir or OMEGA_DIR
        self.size = size
        self.s_jitter = s_jitter
        self.arm = arm
        self.p = np.array([0.3, 0.4, 0.3]) if arm == 'main' else np.array([0.0, 0.57, 0.43])
        if sample_p is not None:
            p = np.asarray([float(x) for x in str(sample_p).split(',')], dtype=np.float64)
            if p.size != 3 or p.min() < 0 or p.sum() <= 0:
                raise SystemExit(f'--sample-p 需 3 个非负数，收到 {sample_p!r}')
            self.p = p / p.sum()
            print(f'[s2] 采样比覆盖为 {self.p.tolist()}', flush=True)
        with open(os.path.join(self.omega_dir, 'meta.json')) as fp:
            self.om_meta = json.load(fp)
        # 2026-10-21：按 domain 过滤（消融要钉死数据构成；不改默认行为）
        if domains:
            keep = {d.strip().lower() for d in str(domains).split(',') if d.strip()}
            before = len(self.om_meta)
            self.om_meta = {k: v for k, v in self.om_meta.items()
                            if str(v.get('domain', '')).lower() in keep}
            if not self.om_meta:
                raise SystemExit(f'[s2] --omega-domains={domains} 过滤后为空（原 {before} 条）')
            print(f'[s2] domain 过滤 {sorted(keep)}: {before} → {len(self.om_meta)} 条')
        self.om_ids = sorted(self.om_meta.keys())
        self.t_files = []
        for d in T_SRCS:
            fs = [os.path.join(d, f) for f in os.listdir(d)
                  if os.path.splitext(f)[1].lower() in ('.jpg', '.png', '.jpeg')]
            self.t_files += fs
        self.t_files = sorted(self.t_files)
        with open(OPENSRC_META) as fp:
            self.os_pairs = json.load(fp)
        self.rrw_ids = [os.path.join('rrw', f) for f in sorted(os.listdir(RRW_R_DIR))]

    def __len__(self):
        return 200000

    def __getitem__(self, idx):
        rng = np.random.default_rng(torch.initial_seed() % (2 ** 32) + idx)
        src = int(rng.choice(3, p=self.p))
        s = self.size
        has_R, has_om, C_I, w = 0.0, 0.0, 0.0, 1.0
        if src == 0:                                    # omega 真实配对（裁剪锚定 Ω，保证窗内含人影）
            case = self.om_ids[rng.integers(len(self.om_ids))]
            m = self.om_meta[case]
            omf = cv2.imread(os.path.join(self.omega_dir, 'omega', case + '.png'), 0)
            H, W = omf.shape
            ys, xs = np.nonzero(omf > 127)
            if len(ys):
                k = rng.integers(len(ys))
                y0 = int(np.clip(int(ys[k]) - s // 2, 0, max(0, H - s)))
                x0 = int(np.clip(int(xs[k]) - s // 2, 0, max(0, W - s)))
            else:
                y0, x0 = int(rng.integers(0, max(1, H - s + 1))), int(rng.integers(0, max(1, W - s + 1)))
            I, (y0, x0) = load_rgb(os.path.join(self.omega_dir, 'I', case + '.png'), s, rng, (y0, x0))
            T, _ = load_rgb(os.path.join(self.omega_dir, 'T', case + '.png'), s, rng, (y0, x0))
            R, _ = load_rgb(os.path.join(self.omega_dir, 'R', case + '.png'), s, rng, (y0, x0))
            R = R - 128.0 / 255.0
            om3, _ = load_rgb(os.path.join(self.omega_dir, 'omega', case + '.png'), s, rng, (y0, x0))
            om = (om3[:, :, :1] > 0.5).transpose(2, 0, 1).astype(np.float32)
            has_R, has_om, C_I = 1.0, (1.0 if om.any() else 0.0), float(m['C_I'])
        elif src == 1:                                  # 在线合成
            T, _ = load_rgb(self.t_files[rng.integers(len(self.t_files))], s, rng)
            Rn = synth_R(rng, self.omega_dir)
            H, W = Rn.shape[:2]
            if min(H, W) < s:
                Rn = cv2.resize(Rn, (max(s, round(W * s / min(H, W))),
                                     max(s, round(H * s / min(H, W)))), interpolation=cv2.INTER_AREA)
                H, W = Rn.shape[:2]
            y0, x0 = rng.integers(0, H - s + 1), rng.integers(0, W - s + 1)
            Rn = Rn[y0:y0 + s, x0:x0 + s]
            Rn = blur_R(Rn, rng)
            b = float(np.exp(rng.uniform(np.log(0.4), np.log(1.5))))
            j = float(rng.uniform(1.0 - self.s_jitter, 1.0))
            Iv = T + b * Rn
            hi = float(np.percentile(Iv, 99.5))          # 自适应曝光上限：防高饱和（实拍相机亦自动避过曝）
            if hi * j > 1.0:
                j = max(0.4, 0.999 / hi)
            I = np.clip(j * Iv, 0, 1).astype(np.float32)
            T = (j * T).astype(np.float32)               # T'=s·T（曝光同时作用于两层，透射零偏移保持）
            R = (j * b * Rn).astype(np.float32)
            om = np.zeros((1, s, s), np.float32)
            has_R = 1.0
        else:                                           # 开源直督
            p = self.os_pairs[rng.integers(len(self.os_pairs))]
            I, (y0, x0) = load_rgb(p['I'], s, rng)
            T, _ = load_rgb(p['T'], s, rng, (y0, x0))
            R = np.zeros_like(I)
            om = np.zeros((1, s, s), np.float32)
            w = 0.5
        if rng.random() < 0.5:                          # 水平翻转
            I, T, R, om = (x[:, :, ::-1].copy() for x in (I, T, R, om))
        c = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))  # noqa: E731
        return (c(I), c(T), c(R), torch.from_numpy(np.ascontiguousarray(om)),
                torch.tensor([has_R, has_om, C_I, w, float(src)], dtype=torch.float32))


_RAND_CNT = [0]


def grad_mag(x):
    """Forward-difference gradient magnitude (channel-max), keeps HxW."""
    gx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.maximum(gx, gy).mean(dim=1, keepdim=True)


def area_matched_random_mask(OM, deterministic=False):
    """Replace each Omega mask with a random disk of (approx.) the same area.

    Fair control for "does the HOLE STRUCTURE matter, or is any region of the same size enough?"
    deterministic=True: 每对图固定一个盘（以掩码内容为种子），用于分离"每步重抽"的增强效应。
    """
    B, _, H, W = OM.shape
    out = torch.zeros_like(OM)
    ys = torch.arange(H, device=OM.device, dtype=torch.float32)[:, None]
    xs = torch.arange(W, device=OM.device, dtype=torch.float32)[None, :]
    for b in range(B):
        A = float(OM[b].sum().item())
        if A < 1.0:
            continue
        r = (A / 3.14159265) ** 0.5
        r = min(r, 0.49 * min(H, W))          # keep the disk fully inside -> equal area
        if deterministic:
            key = int(hashlib.md5(OM[b].detach().float().cpu().numpy().tobytes()).hexdigest(), 16) % (2 ** 32)
            _rng = np.random.default_rng(key)
            uy, ux = float(_rng.random()), float(_rng.random())
        else:
            uy = float(torch.rand(1, device=OM.device).item())
            ux = float(torch.rand(1, device=OM.device).item())
        cy = r + uy * max(H - 2 * r, 1e-6)
        cx = r + ux * max(W - 2 * r, 1e-6)
        m = ((ys - cy) ** 2 + (xs - cx) ** 2) <= (r * r)
        out[b, 0] = m.to(out.dtype)
    _RAND_CNT[0] += 1
    if _RAND_CNT[0] <= 5:
        ra = float(out.sum().item()) / max(float(OM.sum().item()), 1e-6)
        print("[omega-random] realized area ratio = %.3f" % ra, flush=True)
    return out


def grad_excl(T_hat, R_hat):
    def gm(x):
        gx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(1)
        gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(1)
        return gx, gy
    tx, ty = gm(T_hat); rx, ry = gm(R_hat)
    return (tx * rx).mean() + (ty * ry).mean()


def grad_fidelity(T_hat, T):
    """全图一阶梯度匹配 ‖∇T̂−∇T‖₁（防"抹平"退化；SPSR 梯度损失口径）。"""
    gxh = (T_hat[..., :, 1:] - T_hat[..., :, :-1]) - (T[..., :, 1:] - T[..., :, :-1])
    gyh = (T_hat[..., 1:, :] - T_hat[..., :-1, :]) - (T[..., 1:, :] - T[..., :-1, :])
    return gxh.abs().mean() + gyh.abs().mean()


def grad_fidelity_elem(T_hat, T):
    """逐样本版（返回 [B]），用于乘样本权重 `w_s`。

    2026-10-21 修：原 `grad_fidelity`/`lap_fidelity` **未乘 `w_s`**（见
    docs/2026.9.21_S2v2修正版_负结果与归因.md §4）—— opensrc 分支像素项半权，
    而两项保真全权，等于把 opensrc 的梯度压力相对放大。此处改为先取逐样本均值再乘 w_s。
    """
    gxh = (T_hat[..., :, 1:] - T_hat[..., :, :-1]) - (T[..., :, 1:] - T[..., :, :-1])
    gyh = (T_hat[..., 1:, :] - T_hat[..., :-1, :]) - (T[..., 1:, :] - T[..., :-1, :])
    return gxh.abs().mean(dim=(1, 2, 3)) + gyh.abs().mean(dim=(1, 2, 3))


_LAP_K = None


def lap_fidelity(T_hat, T):
    """全图二阶 Laplacian 匹配 ‖ΔT̂−ΔT‖₁（SPSR 二阶项）。"""
    global _LAP_K
    if _LAP_K is None:
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=T.device)
        _LAP_K = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    f = lambda x: F.conv2d(x, _LAP_K, padding=1, groups=3)   # noqa: E731
    return (f(T_hat) - f(T)).abs().mean()


def lap_fidelity_elem(T_hat, T):
    """逐样本版（返回 [B]），见 `grad_fidelity_elem` 的说明。"""
    global _LAP_K
    if _LAP_K is None:
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=T.device)
        _LAP_K = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    f = lambda x: F.conv2d(x, _LAP_K, padding=1, groups=3)   # noqa: E731
    return (f(T_hat) - f(T)).abs().mean(dim=(1, 2, 3))


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument('--omega-mode', choices=['real', 'random', 'randomfixed', 'full'], default='real',
                    help='real = Omega mask; random = area-matched random disk, REDRAWN every iter (control); '
                         'randomfixed = same disk per pair (drawn once); full = whole-image relative loss (no mask)')
    ap.add_argument('--lam-hinge', type=float, default=1.0)
    ap.add_argument('--lam-dark', type=float, default=0.0,
                    help='E3: supervise the MEAN darkening inside Omega (oracle illumination); 0 = off')
    ap.add_argument('--lam-shift', type=float, default=0.0,
                    help='weight of L_shift (hole-boundary gradient term); 0 = off')
    ap.add_argument('--alpha-shift', type=float, default=0.25,
                    help='allowed |grad T| as a fraction of |grad R| on dOmega')
    ap.add_argument('--lam-dists', type=float, default=0.2)
    ap.add_argument('--lam-grad', type=float, default=5.0, help='全图一阶梯度保真（防抹平）')
    ap.add_argument('--lam-lap', type=float, default=2.0, help='全图二阶 Laplacian 保真（防抹平）')
    ap.add_argument('--s-jitter', type=float, default=0.0, help='合成曝光抖动幅度（0=关闭）')
    ap.add_argument('--omega-dir', default=None, help='真实 Ω 配对目录（默认 train_pairs_omega_0920；LOVO 子集用）')
    ap.add_argument('--omega-domains', default=None,
                    help='逗号分隔，按 meta 的 domain 字段过滤（如 apr）；默认不过滤')
    ap.add_argument('--manifest', default='',
                    help='实验清单 JSON 写出路径（配置/环境/可训练参数/数据构成，供消融归档）')
    ap.add_argument('--assert-under', default='',
                    help='入口断言：所有 omega case 路径须落在该目录下，否则中止（防手滑混入锁死域）')
    ap.add_argument('--no-dists', action='store_true', help='不创建 pyiqa DISTS（λ_dists 必须为 0）')
    ap.add_argument('--legacy-fidelity', action='store_true',
                    help='grad/lap 保真项回到 9-21 旧行为（**不乘 w_s**），用于复现旧负结果')
    ap.add_argument('--sample-p', default=None,
                    help='采样比覆盖，如 "0.6,0.2,0.2"（omega,synth,opensrc）；默认由 --arm 决定')
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--seg-iters', type=int, default=2000)
    ap.add_argument('--resume', default='')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = args.out or os.path.join(PROJ, f'runs/ckpt/rdt_s2_{args.arm}_s{args.seed}')
    if not os.path.isabs(out):
        out = os.path.join(PROJ, out)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(f'cuda:{args.gpu}')

    net, net_c = build(save_memory=True, mona=args.mona, grad_ckpt=True)
    for n, p in net.named_parameters():
        if n.startswith('baseball.'):
            p.requires_grad_(False)
    from rdnet_peft import set_mona_trainable
    n_mona = set_mona_trainable(net, True)
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'[s2] trainable {n_tr/1e6:.2f}M (mona {n_mona} 个) arm={args.arm}', flush=True)
    net.to(dev); net_c.to(dev)

    import pyiqa
    if args.no_dists or args.lam_dists <= 0:
        dists_net = None
        print('[s2] DISTS 未启用（λ_dists=0 或 --no-dists）', flush=True)
    else:
        dists_net = pyiqa.create_metric('dists', device=dev).net
        dists_net.eval().requires_grad_(False)

    ds = S2Dataset(arm=args.arm, size=224, s_jitter=args.s_jitter,
                   omega_dir=args.omega_dir, domains=args.omega_domains,
                   sample_p=args.sample_p)
    print(f'[s2] T 源 {len(ds.t_files)} 张 | R 白名单 {len(ds.om_ids)}+{len(ds.rrw_ids)-1} 层 | '
          f'开源 {len(ds.os_pairs)} 对 | p={ds.p}', flush=True)

    # 入口断言：所有 omega case 必须落在 --assert-under 之下（防手滑混入锁死域）
    if args.assert_under:
        root = os.path.realpath(args.assert_under)
        bad = []
        for case in ds.om_ids:
            for sub in ('I', 'T', 'R', 'omega'):
                p = os.path.realpath(os.path.join(ds.omega_dir, sub, case + '.png'))
                if not p.startswith(root + os.sep):
                    bad.append(p)
        if bad:
            raise SystemExit(f'[s2] 入口断言失败：{len(bad)} 个路径不在 {root} 下，例 {bad[:3]}')
        print(f'[s2] 入口断言通过：{len(ds.om_ids)} 个 case 全在 {root} 下', flush=True)

    # 实验清单（消融归档用）
    if args.manifest:
        import subprocess, socket
        mp = args.manifest if os.path.isabs(args.manifest) else os.path.join(PROJ, args.manifest)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        try:
            gitrev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=PROJ,
                                    capture_output=True, text=True).stdout.strip()
        except Exception:
            gitrev = '?'
        man = dict(
            args=vars(args), git_rev=gitrev, host=socket.gethostname(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            torch=torch.__version__, cuda=torch.version.cuda,
            trainable_params=n_tr, mona_keys=n_mona,
            world_size=1, batch_per_gpu=args.batch, effective_batch=args.batch,
            omega_dir=ds.omega_dir,
            n_omega_cases=len(ds.om_ids),
            omega_domains=sorted({str(v.get('domain', '')) for v in ds.om_meta.values()}),
            omega_scenes=len({str(v.get('scene', '')) for v in ds.om_meta.values()}),
            n_t_src=len(ds.t_files), n_opensrc=len(ds.os_pairs), sample_p=ds.p.tolist(),
            start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        )
        with open(mp, 'w', encoding='utf-8') as fh:
            json.dump(man, fh, ensure_ascii=False, indent=1)
        print(f'[s2] manifest -> {mp}', flush=True)
        args._manifest_path = mp
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 200) * (0.1 + 0.45 *
                        (1 + math.cos(math.pi * min(i, args.iters) / args.iters))))
    scaler = torch.cuda.amp.GradScaler()
    start_it = 0
    if args.resume:
        ck_r = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ck_r['icnn'])
        opt.load_state_dict(ck_r['opt']); scaler.load_state_dict(ck_r['scaler'])
        start_it = int(ck_r.get('iter', 0)); sched.last_epoch = start_it
        print(f'[s2] resume @ {start_it}', flush=True)

    logf = open(os.path.join(out, 'train_log.csv'), 'a')
    if start_it == 0:
        logf.write('iter,seg,L_t,L_r,L_om,L_hinge,L_shift,L_dists,L_recons,L_excl,L_grad,L_lap,L_dark,total,sec\n')
    segs = [(224, args.batch), (256, min(args.batch, 6)), (320, min(args.batch, 4))]
    it, t0, prev_seg = start_it, time.time(), None
    while it < args.iters:
        seg_id = it // args.seg_iters
        s, b = segs[seg_id % len(segs)]
        if s != prev_seg:
            torch.cuda.reset_peak_memory_stats(dev)
            print(f'[s2] seg -> {s}² bs{b} @iter {it}', flush=True)
            prev_seg = s
        ds.size = s
        dl = DataLoader(ds, batch_size=b, shuffle=True, num_workers=4, drop_last=True,
                        pin_memory=True)
        for I, T, R, OM, meta in dl:
            if it >= args.iters or it // args.seg_iters != seg_id:
                break
            I = I.to(dev, non_blocking=True); T = T.to(dev, non_blocking=True)
            R = R.to(dev, non_blocking=True); OM = OM.to(dev, non_blocking=True)
            meta = meta.to(dev, non_blocking=True)
            has_R = meta[:, 0].view(-1, 1, 1, 1); has_om = meta[:, 1].view(-1, 1, 1, 1)
            C_I = meta[:, 2].view(-1, 1, 1, 1); w_s = meta[:, 3].view(-1, 1, 1, 1)
            with torch.no_grad():
                code = net_c(I)
            with torch.cuda.amp.autocast():
                _, outs = net(I, code, prompt=True)
                L_t = L_r = L_om = L_hinge = L_dists = L_shift = torch.zeros((), device=dev)
                L_dark = torch.zeros((), device=dev)
                wsum = 0.0
                hw = float(R.shape[2] * R.shape[3])
                L_grad = L_lap = torch.zeros((), device=dev)
                for ci, out_col in enumerate(outs):
                    R_hat, T_hat = out_col[:, :3], out_col[:, 3:]
                    wc = float(ci + 1); wsum += wc
                    L_t = L_t + wc * (w_s * charbonnier(T_hat, T)).mean()
                    _ws = w_s.reshape(-1)
                    if args.lam_grad > 0:
                        if args.legacy_fidelity:      # 复现 9-21 旧行为（漏乘 w_s）
                            L_grad = L_grad + args.lam_grad * wc * grad_fidelity(T_hat, T)
                        else:
                            L_grad = L_grad + args.lam_grad * wc * (_ws * grad_fidelity_elem(T_hat, T)).mean()
                    if args.lam_lap > 0:
                        if args.legacy_fidelity:
                            L_lap = L_lap + args.lam_lap * wc * lap_fidelity(T_hat, T)
                        else:
                            L_lap = L_lap + args.lam_lap * wc * (_ws * lap_fidelity_elem(T_hat, T)).mean()
                    if has_R.max() > 0:
                        num = (w_s * charbonnier(R_hat, R) * has_R).sum()
                        den_r = (w_s * has_R).sum() * hw + EPS
                        L_r = L_r + args.lam_r * wc * num / den_r
                L_t = L_t / wsum; L_r = L_r / wsum
                L_grad = L_grad / wsum; L_lap = L_lap / wsum
                R_hat, T_hat = outs[-1][:, :3], outs[-1][:, 3:]
                L_recons = args.lam_recons * (w_s * (I - T_hat - R_hat).abs()).mean()
                L_excl = args.lam_excl * grad_excl(T_hat, R_hat)
                if has_om.max() > 0:
                    if args.omega_mode == 'random':
                        OM = area_matched_random_mask(OM)
                    elif args.omega_mode == 'randomfixed':
                        OM = area_matched_random_mask(OM, deterministic=True)
                    elif args.omega_mode == 'full':
                        OM = torch.ones_like(OM)
                    om_sum = OM.sum(dim=(1, 2, 3)) + 1e-6
                    err = (OM * (T_hat - T).abs()).sum(dim=(1, 2, 3)) / om_sum
                    den = ((OM * (I - T).abs()).sum(dim=(1, 2, 3)) / om_sum).clamp(min=1e-3)
                    m_om = has_om.squeeze()
                    L_om = args.lam_om * (err / den * m_om).sum() / m_om.sum().clamp(min=1)
                    delta = (torch.clamp(0.5 * C_I, min=4.0) / 255.0)
                    hg = (OM * F.relu(T - delta - T_hat)).sum(dim=(1, 2, 3)) / om_sum
                    L_hinge = args.lam_hinge * (hg / den * m_om).sum() / m_om.sum().clamp(min=1)
                    if args.lam_dark > 0:      # E3: oracle illumination (mean darkening inside Omega)
                        mt = (OM * T).sum(dim=(1, 2, 3)) / om_sum
                        mh = (OM * T_hat).sum(dim=(1, 2, 3)) / om_sum
                        L_dark = args.lam_dark * ((((mh - mt).abs()) / (mt.abs() + EPS)) * m_om).sum() / m_om.sum().clamp(min=1)
                    # 2026-10-26: L_shift -- the hole boundary is an edge in R but not in T
                    if args.lam_shift > 0:
                        OMd = F.max_pool2d(OM, 3, 1, 1)
                        OMe = -F.max_pool2d(-OM, 3, 1, 1)
                        bd = (OMd - OMe).clamp(min=0.0)
                        bd_sum = bd.sum(dim=(1, 2, 3)) + 1e-6
                        gT = grad_mag(T_hat); gR = grad_mag(R_hat)
                        num_s = (bd * F.relu(gT - args.alpha_shift * gR)).sum(dim=(1, 2, 3)) / bd_sum
                        den_s = ((bd * gR).sum(dim=(1, 2, 3)) / bd_sum).clamp(min=1e-3)
                        L_shift = args.lam_shift * ((num_s / den_s) * m_om).sum() / m_om.sum().clamp(min=1)
                    idxs = (m_om > 0).nonzero().squeeze(-1)
                    dsum = torch.zeros((), device=dev)
                    # 2026-10-21：λ_dists=0 时不再白算 DISTS 前向（原先每 iter 最多 4 次）
                    if args.lam_dists > 0:
                        with torch.cuda.amp.autocast(enabled=False):
                            for bi in idxs[:4]:                   # 每 iter 最多 4 张省时
                                m = OM[bi, 0] > 0.5
                                ys, xs = m.any(1).nonzero(), m.any(0).nonzero()
                                y0, y1 = max(0, int(ys.min()) - 10), int(ys.max()) + 11
                                x0, x1 = max(0, int(xs.min()) - 10), int(xs.max()) + 11
                                ta = T_hat[bi:bi + 1, :, y0:y1, x0:x1].float()
                                tb = T[bi:bi + 1, :, y0:y1, x0:x1].float()
                                sc = 320.0 / max(ta.shape[2], ta.shape[3])
                                if sc < 1:
                                    ta = F.interpolate(ta, scale_factor=sc, mode='bilinear', align_corners=False)
                                    tb = F.interpolate(tb, scale_factor=sc, mode='bilinear', align_corners=False)
                                dsum = dsum + args.lam_dists * dists_net(ta, tb)
                        L_dists = dsum / max(1, min(len(idxs), 4))
                total = (L_t + L_r + L_om + L_hinge + L_shift + L_dists + L_recons
                         + L_excl + L_grad + L_lap + L_dark)
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(opt); scaler.update(); sched.step()
            it += 1
            if it % args.log_every == 0 or it == 1:
                sec = time.time() - t0
                g = lambda x: float(x.detach())      # noqa: E731
                print(f'[s2] it {it:5d} L_t={g(L_t):.4f} L_r={g(L_r):.4f} om={g(L_om):.3f} '
                      f'hinge={g(L_hinge):.3f} shift={g(L_shift):.4f} dists={g(L_dists):.3f} rcn={g(L_recons):.4f} '
                      f'excl={g(L_excl):.3f} grad={g(L_grad):.3f} lap={g(L_lap):.3f} dark={g(L_dark):.4f} total={g(total):.4f} {sec:.0f}s '
                      f'peak={torch.cuda.max_memory_allocated(dev)/2**30:.2f}G', flush=True)
                logf.write(f'{it},{s},{g(L_t):.5f},{g(L_r):.5f},{g(L_om):.5f},{g(L_hinge):.5f},'
                           f'{g(L_shift):.5f},{g(L_dists):.5f},{g(L_recons):.5f},{g(L_excl):.5f},{g(L_grad):.5f},{g(L_lap):.5f},{g(L_dark):.5f},{g(total):.5f},'
                           f'{sec:.0f}\n'); logf.flush()
            if it % args.save_every == 0:
                torch.save(dict(icnn=net.state_dict(), opt=opt.state_dict(),
                                scaler=scaler.state_dict(), iter=it,
                                peft=dict(mode=args.mona), arm=args.arm, args=vars(args)),
                           os.path.join(out, f'ckpt_{it}.pth'))
                torch.save(dict(icnn=net.state_dict(), opt=opt.state_dict(),
                                scaler=scaler.state_dict(), iter=it,
                                peft=dict(mode=args.mona), arm=args.arm, args=vars(args)),
                           os.path.join(out, 'ckpt_latest.pth'))
                print(f'[s2] saved ckpt_{it}.pth', flush=True)
    torch.save(dict(icnn=net.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                    iter=it, peft=dict(mode=args.mona), arm=args.arm, args=vars(args)),
               os.path.join(out, 'ckpt_final.pth'))
    print(f'[s2] done {it} iter -> {out}', flush=True)

    # 清单收尾：补结束时间/时长/末次损失（供消融归档与「干预是否生效」核查）
    mp = getattr(args, '_manifest_path', '')
    if mp and os.path.exists(mp):
        man = json.load(open(mp, encoding='utf-8'))
        g = lambda x: float(x.detach())          # noqa: E731
        man.update(end_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                   wall_sec=round(time.time() - t0, 1), final_iter=it,
                   ckpt=os.path.join(out, 'ckpt_final.pth'),
                   loss_last=dict(L_t=g(L_t), L_r=g(L_r), L_om=g(L_om), L_hinge=g(L_hinge),
                                  L_dists=g(L_dists), L_recons=g(L_recons), L_excl=g(L_excl),
                                  L_grad=g(L_grad), L_lap=g(L_lap), L_dark=g(L_dark), total=g(total)),
                   peak_mem_gb=round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2))
        with open(mp, 'w', encoding='utf-8') as fh:
            json.dump(man, fh, ensure_ascii=False, indent=1)
        print(f'[s2] manifest 收尾 -> {mp}', flush=True)


if __name__ == '__main__':
    main()
