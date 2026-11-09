#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 scripts/finetune_arch1030.py = scripts/finetune_rdnet_s2.py 的**逐行副本 + 6 处定点补丁**。

为什么用生成器而不是直接手写 678 行：本项目纪律是"只准新增文件、原 s2 一字不改"，
生成器能让"新文件与基线文件的差异"逐条可审、可复算 —— 每处替换都断言命中次数，
少一处/多一处立即中止，不会静默漂移。

跑法（服务器，项目根）:
    /DataDisk/miniconda/envs/xreflection/bin/python3 scripts/make_finetune_arch1030.py
产物:
    scripts/finetune_arch1030.py
"""
from __future__ import annotations

import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "finetune_rdnet_s2.py")
DST = os.path.join(HERE, "finetune_arch1030.py")

# --------------------------------------------------------------------------- 新增函数
BUILD_FN = '''def build_subband1030(pretrained_ckpt=None, mona='both', grad_ckpt=True, save_memory=True):
    """ARCH_H1: RDNet + Mona + Haar 子带残差调制（只新增文件，原类一字不改）。

    顺序不可换（与 finetune_rdnet_peft.build 相同）：FullNet_NLP_Subband(...) 构造内部
    strict 加载 focal.pth -> patch_focalnet 必须在构造之后 -> 最后载 rdnet.pth。
    断言：非 mona、非 subband 的 key 必须全部命中；意外 key 必须为 0。
    """
    from models.arch.rdnet_subband1030 import FullNet_NLP_Subband
    from models.arch.classifier import PretrainedConvNext
    from rdnet_peft import is_mona_name, patch_focalnet

    net = FullNet_NLP_Subband(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
                              num_subnet=4, loss_col=4, num_classes=1000, drop_path=0,
                              save_memory=save_memory, inter_supv=True,
                              head_init_scale=None, kernel_size=3)
    n_blk = patch_focalnet(net, mona, grad_ckpt=grad_ckpt, verbose=True)
    ck = torch.load(pretrained_ckpt or os.path.join(PROJ, 'pretrained/rdnet.pth'),
                    map_location='cpu')
    sd = ck.get('icnn', ck)
    miss, unex = net.load_state_dict(sd, strict=False)

    def _is_new(k):
        return is_mona_name(k) or k.startswith('subband.')

    miss_new = [k for k in miss if _is_new(k)]
    miss_core = [k for k in miss if not _is_new(k)]
    n_new_net = len([k for k in net.state_dict() if _is_new(k)])
    print(f'[arch1030] rdnet.pth: mona={mona} blocks={n_blk} '
          f'missing={len(miss)}(非新增 {len(miss_core)}/新增 {len(miss_new)}) '
          f'unexpected={len(unex)}  新增 key={n_new_net}(net) vs {len(miss_new)}(missing)',
          flush=True)
    assert not miss_core, f'非新增权重未加载（key 被动过的信号）: {miss_core[:8]}'
    assert not unex, f'rdnet.pth 有 net 不认识的 key: {unex[:8]}'
    assert len(miss_new) == n_new_net, \\
        f'新增 key 数不一致: missing={len(miss_new)} net={n_new_net}'
    if mona == 'none':
        assert not [k for k in miss_new if is_mona_name(k)], 'mona=none 时不应有 mona 新增 key'

    net_c = PretrainedConvNext('convnext_small_in22k')
    net_c.load_state_dict(torch.load(os.path.join(PROJ, 'pretrained/cls_model.pth'),
                                     map_location='cpu')['icnn'])
    net_c.eval().requires_grad_(False)
    n_sub = sum(p.numel() for n, p in net.named_parameters() if n.startswith('subband.'))
    print(f'[arch1030] subband 新增参数 {n_sub} 个（预期 2969）', flush=True)
    return net, net_c

'''

# --------------------------------------------------------------------------- 补丁表
# (name, old, new, expected_count)
PATCHES = [
    ("docstring 头",
     '#!/usr/bin/env python3\n"""S2 训练脚本（"修掉人影"）：RDNet + Mona 适配器',
     '#!/usr/bin/env python3\n'
     '# 2026-10-30 ARCH_H1 副本：本文件由 scripts/make_finetune_arch1030.py 从\n'
     '# scripts/finetune_rdnet_s2.py **逐行复制**后打 6 处定点补丁而成；原 s2 一个字节未改。\n'
     '# 唯一实质改动：--arch subband1030 -> FullNet_NLP_Subband（Haar 子带残差调制，\n'
     '# 只在最后一列改 transmission 残差），其余（数据/9 项损失/调度/存档）与基线逐字相同。\n'
     '"""S2 训练脚本（"修掉人影"）：RDNet + Mona 适配器',
     1),
    ("插 build_subband1030",
     'EPS = 1e-3\nOMEGA_DIR = os.path.join(PROJ, "data/train_sets/train_pairs_omega_0920")',
     BUILD_FN + 'EPS = 1e-3\nOMEGA_DIR = os.path.join(PROJ, "data/train_sets/train_pairs_omega_0920")',
     1),
    ("argparse choices",
     "ap.add_argument('--arch', choices=['mona', 'clean1029'], default='mona',",
     "ap.add_argument('--arch', choices=['mona', 'clean1029', 'subband1030'], default='mona',",
     1),
    ("build 分派",
     "    if args.arch == 'clean1029':\n"
     "        os.environ['TINTED_ARCH'] = 'clean1029'\n"
     "        net, net_c = build_clean1029()\n"
     "    else:\n"
     "        net, net_c = build(save_memory=True, mona=args.mona, grad_ckpt=True)\n",
     "    if args.arch == 'clean1029':\n"
     "        os.environ['TINTED_ARCH'] = 'clean1029'\n"
     "        net, net_c = build_clean1029()\n"
     "    elif args.arch == 'subband1030':\n"
     "        net, net_c = build_subband1030(mona=args.mona, grad_ckpt=True)\n"
     "    else:\n"
     "        net, net_c = build(save_memory=True, mona=args.mona, grad_ckpt=True)\n",
     1),
    ("trainable 打印 + subband 计数",
     "    print(f'[s2] trainable {n_tr/1e6:.2f}M (mona {n_mona} 个) arm={args.arm}', flush=True)\n",
     "    print(f'[s2] trainable {n_tr/1e6:.2f}M (mona {n_mona} 个) arm={args.arm}', flush=True)\n"
     "    if getattr(net, 'subband', None) is not None:\n"
     "        n_sb = sum(p.numel() for k, p in net.named_parameters() if k.startswith('subband.'))\n"
     "        print(f'[s2] subband 新增可训 {n_sb} 参数; gate_identity={net.subband_gate_identity}',\n"
     "              flush=True)\n",
     1),
    ("空掩码 DISTS 守卫（非空样本严格 no-op）",
     "                            for bi in idxs[:4]:                   # 每 iter 最多 4 张省时\n"
     "                                m = OM[bi, 0] > 0.5\n",
     "                            for bi in idxs[:4]:                   # 每 iter 最多 4 张省时\n"
     "                                m = OM[bi, 0] > 0.5\n"
     "                                if not bool(m.any()):   # 空区域样本：跳过（非空样本 no-op）\n"
     "                                    continue\n",
     1),
    ("日志加 |g-1| 诊断",
     "                g = lambda x: float(x.detach())      # noqa: E731\n",
     "                g = lambda x: float(x.detach())      # noqa: E731\n"
     "                sb = ''\n"
     "                if getattr(net, 'subband', None) is not None and \\\n"
     "                        net.subband.last_aux is not None:\n"
     "                    sb = ' sb|g-1|=' + '/'.join(f'{float(v):.4f}' for v in net.subband.last_aux)\n",
     1),
    ("日志行带 sb",
     "                      f'peak={torch.cuda.max_memory_allocated(dev)/2**30:.2f}G', flush=True)",
     "                      f'peak={torch.cuda.max_memory_allocated(dev)/2**30:.2f}G{sb}', flush=True)",
     1),
    ("ckpt 里记 arch（ckpt_it / ckpt_latest / ckpt_final 共 3 处）",
     "peft=dict(mode=args.mona), arm=args.arm, args=vars(args)),",
     "peft=dict(mode=args.mona), arm=args.arm, arch=args.arch, args=vars(args)),",
     3),
]


def main():
    with open(SRC, encoding="utf-8", newline="") as fh:
        src = fh.read()
    assert "\r\n" not in src, "源文件含 CRLF，先归一化"
    txt = src
    for name, old, new, cnt in PATCHES:
        got = txt.count(old)
        if got != cnt:
            raise SystemExit("[make] 补丁 %r 命中 %d 次，期望 %d 次 —— 源文件已变，拒绝生成"
                             % (name, got, cnt))
        txt = txt.replace(old, new)
        print("[make] 补丁 %-32s 命中 %d 次 OK" % (name, got))
    with open(DST, "w", encoding="utf-8", newline="") as fh:
        fh.write(txt)
    py_compile.compile(DST, doraise=True)
    print("[make] 写出 %s  (%d 行, 源 %d 行)  py_compile OK"
          % (DST, txt.count("\n") + 1, src.count("\n") + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
