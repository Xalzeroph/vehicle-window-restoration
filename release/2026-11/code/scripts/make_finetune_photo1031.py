#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 `scripts/finetune_photo1031.py` = `scripts/finetune_rdnet_s2.py` 的**逐行副本 + 定点补丁**。

为什么用生成器而不是手写 678 行：项目纪律是"只准新增文件、原 s2 一字不改"。生成器让
"新文件与基线文件的差异"逐条可审、可复算：每处替换都断言命中次数，少一处/多一处立即
中止，不会静默漂移。（同 `scripts/make_finetune_arch1030.py` 的做法。）

跑法（服务器，项目根）:
    /DataDisk/miniconda/envs/xreflection/bin/python3 scripts/make_finetune_photo1031.py

产物:
    scripts/finetune_photo1031.py

补丁清单（共 11 处，全部有命中次数断言）
--------------------------------------
P1  文件头注释：标明这是 photo1031 副本
P2  插入 `_photo_cfg()` + `build_photo1031()`（放在 `EPS = 1e-3` 之前）
P3  `--arch` 增加 `photo1031` 选项；新增 `--photo {on,off}`
P4  build 分派：`--arch photo1031` -> `build_photo1031(...)`
P5  冻 backbone / 解冻 mona 之后：`--photo off` 时把适配器参数也设成不可训（真基线口径），
    并打印适配器参数量与开关状态
P6  清单 JSON 增加 `arch` / `photo`
P7  日志行增加适配器诊断（列数、|b|、std(S)）与 reserved 显存
P7b T 源目录可读性：默认与基线逐字相同；只有 `TINTED_ALLOW_MISSING_TSRC=1` 才跳过不可读目录
    （2026-10-31 服务器 /home 走 emergency_ro + EIO，`vehicle_ssd/datasets/scenes` 不可读）
P7c 清单 JSON 记录被跳过的 T 源（`t_src_skipped`）
P8  日志行尾追加诊断串
P9  三处 ckpt 存档增加 `arch` / `photo`
"""
from __future__ import annotations

import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "finetune_rdnet_s2.py")
DST = os.path.join(HERE, "finetune_photo1031.py")

# --------------------------------------------------------------------------- 新增函数
BUILD_FN = '''def _photo_cfg(args, net):
    """把 I1 适配器的配置写进 manifest / ckpt（非 photo1031 架构返回 None）。"""
    ph = getattr(net, 'photo', None)
    if ph is None:
        return None
    return dict(enabled=bool(getattr(net, 'photo_enabled', True)),
                n_params=sum(p.numel() for p in ph.parameters()),
                hidden=ph.hidden, grid=ph.grid, spatial=ph.spatial,
                f0_dim=ph.f0_dim, f3_dim=ph.f3_dim, stats_eps=ph.stats_eps,
                f0_src='c0(64ch)', f3_src='c3(512ch)', shared_across_cols=True,
                init='fc2=0, out_conv=0, out_conv.bias 已删', interp='bilinear align_corners=False',
                dc_split='b_c 全局 + S-mean_xy(S) 零均值（FP32 累积）')


def build_photo1031(pretrained_ckpt=None, mona='both', grad_ckpt=True, save_memory=True,
                    enabled=True):
    """I1: RDNet + Mona + 全局/空间缓变光度适配器（只新增文件，原类一字不改）。

    顺序不可换（与 finetune_rdnet_peft.build 相同）：FullNet_NLP_Photo(...) 构造内部
    strict 加载 focal.pth -> patch_focalnet 必须在构造之后 -> 最后载 rdnet.pth。
    断言：非 mona、非 photo 的 key 必须全部命中；意外 key 必须为 0；
          新增参数必须恰好 18,467（审核更正后的值）。
    """
    from models.arch.rdnet_photo1031 import FullNet_NLP_Photo, PHOTO_N_PARAMS
    from models.arch.classifier import PretrainedConvNext
    from rdnet_peft import is_mona_name, patch_focalnet

    net = FullNet_NLP_Photo(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2],
                            num_subnet=4, loss_col=4, num_classes=1000, drop_path=0,
                            save_memory=save_memory, inter_supv=True,
                            head_init_scale=None, kernel_size=3, photo_enabled=enabled)
    n_blk = patch_focalnet(net, mona, grad_ckpt=grad_ckpt, verbose=True)
    ck = torch.load(pretrained_ckpt or os.path.join(PROJ, 'pretrained/rdnet.pth'),
                    map_location='cpu')
    sd = ck.get('icnn', ck)
    miss, unex = net.load_state_dict(sd, strict=False)

    def _is_new(k):
        return is_mona_name(k) or k.startswith('photo.')

    miss_new = [k for k in miss if _is_new(k)]
    miss_core = [k for k in miss if not _is_new(k)]
    n_new_net = len([k for k in net.state_dict() if _is_new(k)])
    print(f'[photo1031] rdnet.pth: mona={mona} blocks={n_blk} '
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

    n_ph = sum(p.numel() for p in net.photo.parameters())
    print(f'[photo1031] 适配器参数 {n_ph} 个（预登记 {PHOTO_N_PARAMS}）；'
          f'enabled={net.photo_enabled}', flush=True)
    assert n_ph == PHOTO_N_PARAMS, f'适配器参数 {n_ph} ≠ 预登记 {PHOTO_N_PARAMS}（审核更正后的值）'
    return net, net_c


'''

# --------------------------------------------------------------------------- 补丁表
PATCHES = [
    ("P1 文件头",
     '#!/usr/bin/env python3\n"""S2 训练脚本（"修掉人影"）：RDNet + Mona 适配器',
     '#!/usr/bin/env python3\n'
     '# 2026-10-31 I1 副本：本文件由 scripts/make_finetune_photo1031.py 从\n'
     '# scripts/finetune_rdnet_s2.py **逐行复制**后打 9 处定点补丁而成；原 s2 一个字节未改。\n'
     '# 唯一实质改动：--arch photo1031 -> FullNet_NLP_Photo（全局均值 + 空间缓变光度适配器，\n'
     '# 18,467 参数，只校正每列输出的后 3 通道=透射层）；数据/四项损失/调度/存档与基线逐字相同。\n'
     '"""S2 训练脚本（"修掉人影"）：RDNet + Mona 适配器',
     1),
    ("P2 插 build_photo1031",
     'EPS = 1e-3\nOMEGA_DIR = os.path.join(PROJ, "data/train_sets/train_pairs_omega_0920")',
     BUILD_FN + 'EPS = 1e-3\nOMEGA_DIR = os.path.join(PROJ, "data/train_sets/train_pairs_omega_0920")',
     1),
    ("P3 --arch 选项 + --photo 开关",
     "    ap.add_argument('--arch', choices=['mona', 'clean1029'], default='mona',\n"
     '                    help="mona = 我们的 RDNet+Mona；clean1029 = 队友的新架构（无 Mona）")\n',
     "    ap.add_argument('--arch', choices=['mona', 'clean1029', 'photo1031'], default='mona',\n"
     '                    help="mona = 我们的 RDNet+Mona；clean1029 = 队友的新架构（无 Mona）；'
     'photo1031 = +I1 光度适配器")\n'
     "    ap.add_argument('--photo', choices=['on', 'off'], default='on',\n"
     "                    help='I1 适配器开关: on=启用; off=完全旁路(连适配器都不调用)且参数不可训，'\n"
     "                         '用于「关闭开关=基线」的梯度等价对照')\n",
     1),
    ("P4 build 分派",
     "    if args.arch == 'clean1029':\n"
     "        os.environ['TINTED_ARCH'] = 'clean1029'\n"
     "        net, net_c = build_clean1029()\n"
     "    else:\n"
     "        net, net_c = build(save_memory=True, mona=args.mona, grad_ckpt=True)\n",
     "    if args.arch == 'clean1029':\n"
     "        os.environ['TINTED_ARCH'] = 'clean1029'\n"
     "        net, net_c = build_clean1029()\n"
     "    elif args.arch == 'photo1031':\n"
     "        net, net_c = build_photo1031(mona=args.mona, grad_ckpt=True,\n"
     "                                     enabled=(args.photo == 'on'))\n"
     "    else:\n"
     "        net, net_c = build(save_memory=True, mona=args.mona, grad_ckpt=True)\n",
     1),
    ("P5 关开关时冻适配器 + 打印",
     "    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)\n"
     "    print(f'[s2] trainable {n_tr/1e6:.2f}M (mona {n_mona} 个) arm={args.arm}', flush=True)\n",
     "    if getattr(net, 'photo', None) is not None and not net.photo_enabled:\n"
     "        net.photo.requires_grad_(False)   # 真旁路口径：off 臂的可训集合 == 基线\n"
     "    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)\n"
     "    print(f'[s2] trainable {n_tr/1e6:.2f}M (mona {n_mona} 个) arm={args.arm}', flush=True)\n"
     "    if getattr(net, 'photo', None) is not None:\n"
     "        n_ph = sum(p.numel() for p in net.photo.parameters())\n"
     "        print(f'[s2] photo1031 适配器 {n_ph} 参数，enabled={net.photo_enabled}，'\n"
     "              f'可训={sum(p.numel() for p in net.photo.parameters() if p.requires_grad)}',\n"
     "              flush=True)\n",
     1),
    ("P6 manifest 加 arch/photo",
     "            trainable_params=n_tr, mona_keys=n_mona,\n",
     "            trainable_params=n_tr, mona_keys=n_mona,\n"
     "            arch=args.arch, photo=_photo_cfg(args, net),\n",
     1),
    ("P7 日志：适配器诊断 + reserved",
     "                g = lambda x: float(x.detach())      # noqa: E731\n",
     "                g = lambda x: float(x.detach())      # noqa: E731\n"
     "                ph = ''                              # >>> I1 适配器诊断\n"
     "                if getattr(net, 'photo_diag', None):\n"
     "                    _st = [v.tolist() for v in net.photo_diag['stats']]\n"
     "                    ph = (f\" ph[cols={net.photo_diag['cols']}\"\n"
     "                          f\" |b|={sum(r[0] for r in _st)/max(1, len(_st)):.4f}\"\n"
     "                          f\" stdS={sum(r[1] for r in _st)/max(1, len(_st)):.4f}\"\n"
     "                          f\" maxS={max(r[2] for r in _st):.4f}]\")\n",
     1),
    ("P7b T 源目录可读性（默认=基线行为，显式 opt-in 才跳过）",
     'T_SRCS = [os.path.join(PROJ, "data/external/VOC2012/VOC2012_s320"),\n'
     '          "/home/zmy/mycode/datasets/SIR2/SIR/transmission_layer",\n'
     '          "/home/zmy/mycode/datasets/nature/natural_T",\n'
     '          "/home/zmy/vehicle_ssd/datasets/scenes",\n'
     '          "/DataDisk/zmy_code/datasets/RRWDataset/reflection_camera1/GT_camera1"]\n',
     '# >>> I1 工程：T 源目录可读性（2026-10-31 服务器 /home 走 emergency_ro + EIO，\n'
     '#     /home/zmy/vehicle_ssd/datasets/scenes 不可读 -> 基线脚本在 S2Dataset 初始化即崩）。\n'
     '#     默认行为与基线**逐字相同**（不吞异常）；只有显式设置 TINTED_ALLOW_MISSING_TSRC=1\n'
     '#     才跳过不可读目录并大声打印 —— 这只用于工程冒烟，**不是正式配方**。\n'
     'T_SRCS_ALL = [os.path.join(PROJ, "data/external/VOC2012/VOC2012_s320"),\n'
     '              "/home/zmy/mycode/datasets/SIR2/SIR/transmission_layer",\n'
     '              "/home/zmy/mycode/datasets/nature/natural_T",\n'
     '              "/home/zmy/vehicle_ssd/datasets/scenes",\n'
     '              "/DataDisk/zmy_code/datasets/RRWDataset/reflection_camera1/GT_camera1"]\n'
     "T_SRCS_SKIPPED = []\n"
     "T_SRCS = list(T_SRCS_ALL)\n"
     "if os.environ.get('TINTED_ALLOW_MISSING_TSRC', '0') == '1':\n"
     "    T_SRCS = []\n"
     "    for _d in T_SRCS_ALL:\n"
     "        try:\n"
     "            os.listdir(_d)\n"
     "        except OSError as _e:\n"
     "            T_SRCS_SKIPPED.append([_d, str(_e)])\n"
     "            print(f'[s2][WARN] T 源不可读，跳过: {_d} ({_e})', flush=True)\n"
     "            continue\n"
     "        T_SRCS.append(_d)\n"
     "    print(f'[s2][WARN] TINTED_ALLOW_MISSING_TSRC=1: 跳过 {len(T_SRCS_SKIPPED)} 个 T 源，'\n"
     "          f'剩 {len(T_SRCS)} 个 —— 本臂数据构成 != 正式配方（仅用于工程冒烟）', flush=True)\n",
     1),
    ("P7c manifest 记录被跳过的 T 源",
     "            n_t_src=len(ds.t_files), n_opensrc=len(ds.os_pairs), sample_p=ds.p.tolist(),\n",
     "            n_t_src=len(ds.t_files), n_opensrc=len(ds.os_pairs), sample_p=ds.p.tolist(),\n"
     "            t_src_skipped=T_SRCS_SKIPPED,\n",
     1),
    ("P8 日志行尾带 ph + reserved",
     "                      f'peak={torch.cuda.max_memory_allocated(dev)/2**30:.2f}G', flush=True)",
     "                      f'peak={torch.cuda.max_memory_allocated(dev)/2**30:.2f}G/'\n"
     "                      f'{torch.cuda.max_memory_reserved(dev)/2**30:.2f}G{ph}', flush=True)",
     1),
    ("P9 ckpt 加 arch/photo（3 处）",
     "peft=dict(mode=args.mona), arm=args.arm, args=vars(args)),",
     "peft=dict(mode=args.mona), arm=args.arm, arch=args.arch,\n"
     "                             photo=_photo_cfg(args, net), args=vars(args)),",
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
        print("[make] 补丁 %-34s 命中 %d 次 OK" % (name, got))
    with open(DST, "w", encoding="utf-8", newline="") as fh:
        fh.write(txt)
    py_compile.compile(DST, doraise=True)
    print("[make] 写出 %s  (%d 行, 源 %d 行)  py_compile OK"
          % (DST, txt.count("\n") + 1, src.count("\n") + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
