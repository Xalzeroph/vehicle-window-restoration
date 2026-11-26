#!/usr/bin/env python3
"""RDNet + Mona teacher training on the frozen paired photo manifest.

Each sample provides observed input and reflection-free transmission target. Supervise
reflection as the observed residual I-T; this is not a separately captured
physical reflection layer. Native paired data is used for DARL and the reference RDNet+Mona control.
"""
import argparse, csv, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, HERE)
sys.path.insert(0, RDNET)
os.chdir(RDNET)
from models.arch.classifier import PretrainedConvNext  # noqa: E402
from rdnet_peft import (DARLRDNet, patch_focalnet, mona_params, set_mona_trainable,   # noqa: E402
                        is_mona_name, mona_state_dict_keys)

EPS = 1e-3


def charbonnier(a, b):
    return torch.sqrt((a - b) ** 2 + EPS ** 2)


def _seed_worker(worker_id):
    # fork 出的 worker 共享父进程 numpy 状态 → 各 worker 增强序列相同；用 torch 已种好的
    # worker 种子重种 numpy（torch.initial_seed 每 worker/每 epoch 唯一）
    np.random.seed(torch.initial_seed() % (2 ** 32))


class TintDataset(Dataset):
    """Native paired image samples from the frozen real data manifest."""

    def __init__(self, manifest, size_min=224, size_max=320):
        self.root = os.path.dirname(os.path.abspath(manifest))
        with open(manifest, encoding="utf-8") as stream:
            self.items = json.load(stream)["items"]
        self.size = size_min
        self.smin, self.smax = size_min, size_max

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(os.path.join(self.root, item["input"])) as opened:
            observed = opened.convert("RGB")
        with Image.open(os.path.join(self.root, item["target"])) as opened:
            target = opened.convert("RGB")
        size = self.size
        width, height = observed.size
        y = np.random.randint(height - size + 1)
        x = np.random.randint(width - size + 1)
        crop = (x, y, x + size, y + size)
        observed = np.asarray(observed.crop(crop), dtype=np.float32).transpose(2, 0, 1) / 255.0
        target = np.asarray(target.crop(crop), dtype=np.float32).transpose(2, 0, 1) / 255.0
        if np.random.rand() < 0.5:
            observed = observed[:, :, ::-1].copy()
            target = target[:, :, ::-1].copy()
        residual = observed - target
        alpha = np.ones((1, size, size), dtype=np.float32)
        tensor = lambda image: torch.from_numpy(np.ascontiguousarray(image))
        return tensor(observed), tensor(target), tensor(residual), tensor(alpha)

def build(pretrained_ckpt=None, save_memory=True, mona="both", grad_ckpt=True):
    """构造 FullNet_NLP -> patch Mona -> 加载 rdnet.pth（顺序不可换，见模块 docstring）。"""
    net = DARLRDNet(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2], num_subnet=4,
                      loss_col=4, num_classes=1000, drop_path=0, save_memory=save_memory,
                      inter_supv=True, head_init_scale=None, kernel_size=3)
    n_blk = patch_focalnet(net, mona, grad_ckpt=grad_ckpt, verbose=True)  # 必须在构造后（构造内 strict 加载 focal.pth）
    ck = torch.load(pretrained_ckpt or os.path.join(PROJ, "pretrained/rdnet.pth"),
                    map_location="cpu")
    sd = ck.get("icnn", ck)
    n_net_mona = len(mona_params(net))
    n_sd_mona = len(mona_state_dict_keys(sd))
    miss, unex = net.load_state_dict(sd, strict=False)
    miss_mona = [k for k in miss if is_mona_name(k)]
    miss_core = [k for k in miss if not is_mona_name(k) and not k.startswith('attribution.')]
    print(f"[rdt-peft] rdnet.pth: mona={mona} blocks={n_blk} "
          f"missing={len(miss)}(非mona {len(miss_core)}/mona新增 {len(miss_mona)}) "
          f"unexpected={len(unex)}  net_mona_keys={n_net_mona} ckpt_mona_keys={n_sd_mona}")
    assert not miss_core, f"非 mona 权重未加载（key 被改动的信号）: {miss_core[:8]}"
    assert not unex, f"rdnet.pth 有 net 不认识的 key: {unex[:8]}"
    if mona != "none":
        if n_sd_mona == 0:      # 官方 rdnet.pth：mona 全是新增参数，允许（且应当）全 missing
            assert len(miss_mona) == n_net_mona, \
                f"mona key 数异常: net={n_net_mona} missing={len(miss_mona)}"
            print(f"[rdt-peft] 断言通过: 非 mona missing=0; mona {n_net_mona} 个全是新增参数(随机初始化)")
        else:                   # 从 mona ckpt 出发：mona 权重必须全部命中
            assert len(miss_mona) == 0 and n_sd_mona == n_net_mona, \
                f"mona 权重未完整加载: net={n_net_mona} ckpt={n_sd_mona} missing={len(miss_mona)}"
            print(f"[rdt-peft] 断言通过: missing=0 含 mona {n_net_mona} 个（来自 ckpt）")
    else:
        assert not miss_mona and not miss_core, f"权重结构不匹配: {miss[:8]}"
        print("[rdt-peft] 断言通过: missing=0（mona=none，无新增参数）")
    net_c = PretrainedConvNext("convnext_small_in22k")
    ck2 = torch.load(os.path.join(PROJ, "pretrained/cls_model.pth"), map_location="cpu")
    sd2 = ck2.get("icnn", ck2.get("model", ck2))
    miss2, unex2 = net_c.load_state_dict(sd2, strict=False)
    print(f"[rdt-peft] cls_model.pth: missing={len(miss2)} unexpected={len(unex2)}")
    net_c.eval().requires_grad_(False)
    return net, net_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True, help="Frozen real-pair training manifest.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mona", choices=["both", "ffn", "none"], default="both",
                    help="Mona 插入点: both=调制残差后+FFN 残差后; ffn=仅 FFN 残差后; none=不插(plain)")
    ap.add_argument("--grad-ckpt", choices=["on", "off"], default="on",
                    help="FocalNet block 重算换显存（mona 可训会把冻结前向也拉进计算图，"
                         "on 时 224²bs8 才放得下；数学等价，约慢 1.2-1.4x）")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze", choices=["backbone", "none"], default="backbone",
                    help="backbone=冻 FocalNet（mona 除外）；none=全参微调")
    ap.add_argument("--lam-wlow", type=float, default=1.0,
                    help="逆衰减加权开关（>0=开）：w(α)=clamp((wlow_ref/α)²,1,wlow_cap)，T/R 共用")
    ap.add_argument("--wlow-ref", type=float, default=0.3,
                    help="w(α) 锚点：α≥ref 处 w=1（0.3=原 pilot 配方，训练档位下几乎不激活）")
    ap.add_argument("--wlow-cap", type=float, default=10.0)
    ap.add_argument("--lam-r", type=float, default=1.5)
    ap.add_argument("--lam-recons", type=float, default=0.2)
    ap.add_argument("--lam-attr", type=float, default=0.02,
                    help="Weak three-way attribution supervision from paired real-image gradients")
    ap.add_argument("--save-memory", action="store_true", default=True)
    ap.add_argument("--resume", default="", help="ckpt 路径，续训（恢复 net/opt/scaler/iter）")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seg-iters", type=int, default=2000, help="裁剪尺寸/批大小切换周期")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = args.out or os.path.join(PROJ, f"runs/ckpt/rdt_mona_{args.mona}_s{args.seed}")
    if not os.path.isabs(out):
        out = os.path.join(PROJ, out)
    os.makedirs(out, exist_ok=True)

    dev = torch.device(f"cuda:{args.gpu}")
    net, net_c = build(save_memory=args.save_memory, mona=args.mona,
                       grad_ckpt=(args.grad_ckpt == "on"))
    if args.freeze == "backbone":
        for n, p in net.named_parameters():
            if n.startswith("baseball."):
                p.requires_grad_(False)
        # mona 在 baseball.* 之下，会被上一行一起冻掉 -> 必须显式解冻（坑 2）
        n_mona = set_mona_trainable(net, True)
        print(f"[rdt-peft] 冻 baseball.* 后显式解冻 mona 参数: {n_mona} 个")
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in net.parameters())
    mona_tr = sum(p.numel() for _, p in mona_params(net) if p.requires_grad)
    bb_tr = sum(p.numel() for n, p in net.named_parameters()
                if n.startswith("baseball.") and p.requires_grad)
    print(f"[rdt-peft] trainable {n_tr/1e6:.1f}M / {n_all/1e6:.1f}M  (freeze={args.freeze}, mona={args.mona})"
          f"  | mona 可训 {mona_tr/1e6:.2f}M, baseball.* 可训 {bb_tr/1e6:.2f}M")
    net.to(dev); net_c.to(dev)

    ds = TintDataset(args.train_manifest)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.1 + 0.45 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)))
    scaler = torch.cuda.amp.GradScaler()
    start_it = 0
    if args.resume:
        ck_r = torch.load(args.resume, map_location="cpu")
        if ck_r.get("peft", {}).get("mode", args.mona) != args.mona:
            raise RuntimeError(f"resume ckpt 的 mona 模式={ck_r.get('peft', {}).get('mode')} "
                               f"≠ --mona {args.mona}（权重结构不同，拒绝续训）")
        net.load_state_dict(ck_r["icnn"])
        if "opt" in ck_r:
            opt.load_state_dict(ck_r["opt"])
        if "scaler" in ck_r:
            scaler.load_state_dict(ck_r["scaler"])
        start_it = int(ck_r.get("iter", 0))
        sched.last_epoch = start_it
        print(f"[rdt-peft] resume {args.resume} @ iter {start_it}", flush=True)
    logf = open(os.path.join(out, "train_log.csv"), "w")
    logf.write("iter,L_t,L_r,L_recons,L_attr,total,sec\n")
    print(f"[rdt-peft] train manifest={args.train_manifest} paired images={len(ds)} batch={args.batch}", flush=True)

    # 裁剪尺寸与批大小绑定：320²@bs8 在 10GB 卡 OOM（实测 256² bs8≈7.5G）
    segs = [(224, args.batch), (256, min(args.batch, 6)), (320, min(args.batch, 4))]
    it, t0, run = start_it, time.time(), {}
    prev_seg = None
    while it < args.iters:
        seg_id = it // args.seg_iters
        s, b = segs[seg_id % len(segs)]   # 定期换裁剪尺寸 + 批大小
        if s != prev_seg:
            torch.cuda.reset_peak_memory_stats(dev)
            print(f"[rdt-peft] seg -> {s}² bs{b} @iter {it}（峰值显存统计已重置）", flush=True)
            prev_seg = s
        ds.size = s
        dl = DataLoader(ds, batch_size=b, shuffle=True, num_workers=4, drop_last=True,
                        pin_memory=True, worker_init_fn=_seed_worker)
        for I, T, R, A in dl:
            # 段边界即停（原脚本靠 epoch 边界切段；默认 seg_iters=2000 = 4×epoch(bs8,500) 两者等价）
            if it >= args.iters or it // args.seg_iters != seg_id:
                break
            I = I.to(dev, non_blocking=True); T = T.to(dev, non_blocking=True)
            R = R.to(dev, non_blocking=True); A = A.to(dev, non_blocking=True)
            with torch.no_grad():
                code = net_c(I)
            with torch.cuda.amp.autocast():
                attribution_logits, x_img_out = net(I, code, prompt=True)
                L_t = L_r = torch.zeros((), device=dev)
                wsum = 0.0
                for i, out_col in enumerate(x_img_out):
                    R_hat, T_hat = out_col[:, :3], out_col[:, 3:]
                    if args.lam_wlow > 0:
                        w_alpha = ((args.wlow_ref / A.clamp(min=0.02)) ** 2).clamp(1.0, args.wlow_cap)
                        lt = (charbonnier(T_hat, T) * w_alpha).mean()
                        lr_i = (charbonnier(R_hat, R) * w_alpha).mean()
                    else:
                        lt = charbonnier(T_hat, T).mean()
                        lr_i = charbonnier(R_hat, R).mean()
                    wc = float(i + 1); wsum += wc
                    L_t = L_t + lt * wc
                    L_r = L_r + args.lam_r * lr_i * wc
                L_t = L_t / wsum; L_r = L_r / wsum
                R_hat, T_hat = x_img_out[-1][:, :3], x_img_out[-1][:, 3:]
                L_recons = args.lam_recons * (I - T_hat - R_hat).abs().mean()
                if args.lam_attr:
                    def gradient_energy(image):
                        horizontal = F.pad(
                            (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(1, keepdim=True),
                            (0, 1, 0, 0)
                        )
                        vertical = F.pad(
                            (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(1, keepdim=True),
                            (0, 0, 0, 1)
                        )
                        return horizontal + vertical

                    with torch.no_grad():
                        edge_t = gradient_energy(T)
                        edge_r = gradient_energy(R)
                        difference = (edge_t - edge_r).abs()
                        mass = edge_t + edge_r + 1e-3
                        soft_labels = torch.cat((
                            (edge_t - edge_r).relu(),
                            (edge_r - edge_t).relu(),
                            edge_t + edge_r - difference + 1e-3
                        ), dim=1) / mass
                    L_attr = sum(
                        -(F.adaptive_avg_pool2d(soft_labels, logits.shape[-2:])
                          * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
                        for logits in attribution_logits
                    ) / len(attribution_logits)
                else:
                    L_attr = torch.zeros((), device=dev)
                total = L_t + L_r + L_recons + args.lam_attr * L_attr
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(opt); scaler.update(); sched.step()
            it += 1
            if it % args.log_every == 0 or it == 1:
                sec = time.time() - t0
                run = dict(L_t=float(L_t.detach()), L_r=float(L_r.detach()),
                           L_recons=float(L_recons.detach()), L_attr=float(L_attr.detach()),
                           total=float(total.detach()))
                peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
                print(f"[rdt-peft] it {it:5d} L_t={run['L_t']:.4f} L_r={run['L_r']:.4f} "
                      f"L_recons={run['L_recons']:.4f} L_attr={run['L_attr']:.4f} total={run['total']:.4f} {sec:.0f}s "
                      f"peak={peak:.2f}G", flush=True)
                logf.write(",".join([str(it)] + [f"{run[k]:.5f}" for k in
                                                 ("L_t", "L_r", "L_recons", "L_attr", "total")]
                                    + [f"{sec:.1f}"]) + "\n")
                logf.flush()
            if it % args.save_every == 0:
                torch.save({"icnn": net.state_dict(), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "args": vars(args), "iter": it,
                            "peft": {"mode": args.mona}},
                           os.path.join(out, f"ckpt_{it}.pth"))
    torch.save({"icnn": net.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "args": vars(args), "iter": it,
                "peft": {"mode": args.mona}},
               os.path.join(out, "ckpt_final.pth"))
    print(f"done -> {out}/ckpt_final.pth  ({time.time()-t0:.0f}s)  "
          f"peak={torch.cuda.max_memory_allocated(dev)/2**30:.2f}G")


if __name__ == "__main__":
    main()
