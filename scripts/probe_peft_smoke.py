#!/usr/bin/env python3
"""Mona-PEFT 冒烟探针（一次性）：权重完整性 / 冻结正确 / 存档往返 / 评测断言。

覆盖 6 项里的 1、2、5、6 项（第 3、4 项由真实训练脚本跑，命令见 eval/probe_peft_smoke.md）:
  1) patch 前后各加载一次 rdnet.pth：非 mona missing=0、baseball.* key 集合不变
  2) 冻结统计：mona 全 True；baseball.* 全 False；其余与原脚本一致（参数量对比）
  5) 存档往返：重构造 + patch + 加载 mona ckpt，断言 mona missing=0
  6) 评测一致性：--mona none 载 mona ckpt / --mona both 载 plain ckpt 都必须明确报错

用法:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/probe_peft_smoke.py \
      --ckpt runs/ckpt/rdt_mona_smoke100/ckpt_final.pth \
      --plain-ckpt runs/ckpt/rdt_plain/ckpt_final.pth
"""
import argparse, os, sys, traceback
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RDNET = os.path.join(PROJ, "RDNet")
sys.path.insert(0, HERE)
sys.path.insert(0, RDNET)
os.chdir(RDNET)
from models.arch.RDnet_ import FullNet_NLP                       # noqa: E402
from rdnet_peft import (patch_focalnet, mona_params, mona_param_names,   # noqa: E402
                        set_mona_trainable, is_mona_name, mona_state_dict_keys)


def make_net():
    return FullNet_NLP(channels=[64, 128, 256, 512], layers=[2, 2, 4, 2], num_subnet=4,
                       loss_col=4, num_classes=1000, drop_path=0, save_memory=True,
                       inter_supv=True, head_init_scale=None, kernel_size=3)


def freeze_backbone(net, unfreeze_mona=True):
    for n, p in net.named_parameters():
        if n.startswith("baseball."):
            p.requires_grad_(False)
    return set_mona_trainable(net, True) if unfreeze_mona else 0


def rg_stats(net):
    """(mona 参数数, mona 中 requires_grad=True 数, baseball.* 参数数, baseball 中 True 数,
        其余参数中 True 数, 其余参数中 False 数)"""
    m_n = m_t = b_n = b_t = o_t = o_f = 0
    for n, p in net.named_parameters():
        if is_mona_name(n):
            m_n += 1; m_t += int(p.requires_grad)
        elif n.startswith("baseball."):
            b_n += 1; b_t += int(p.requires_grad)
        else:
            o_t += int(p.requires_grad); o_f += int(not p.requires_grad)
    return m_n, m_t, b_n, b_t, o_t, o_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="mona 训练产出的 ckpt（冒烟 100 iter）")
    ap.add_argument("--plain-ckpt", default=os.path.join(PROJ, "runs/ckpt/rdt_plain/ckpt_final.pth"))
    args = ap.parse_args()
    rdt_path = os.path.join(PROJ, "pretrained/rdnet.pth")
    sd_official = torch.load(rdt_path, map_location="cpu")
    sd_official = sd_official.get("icnn", sd_official)

    # ---------------- 1) 权重完整性：patch 前后 ----------------
    print("=" * 100)
    net = make_net()
    net.eval()
    miss0, unex0 = net.load_state_dict(sd_official, strict=False)
    keys_before = {k for k in net.state_dict() if k.startswith("baseball.")}
    print(f"[smoke-1] patch 前加载 rdnet.pth: missing={len(miss0)} unexpected={len(unex0)} "
          f"baseball_keys={len(keys_before)}")
    assert len(miss0) == 0 and len(unex0) == 0, "官方 rdnet.pth 在未 patch 模型上必须 0 missing"

    # 未 patch 时的可训参数量（= plain 主臂基线）
    freeze_backbone(net, unfreeze_mona=False)
    tr_plain = sum(p.numel() for p in net.parameters() if p.requires_grad)
    plain_other = sum(p.numel() for n, p in net.named_parameters()
                      if p.requires_grad and not n.startswith("baseball."))
    print(f"[smoke-2a] plain（未 patch）冻 baseball 后可训 {tr_plain/1e6:.2f}M，"
          f"其中非 baseball {plain_other/1e6:.2f}M")

    n_blk = patch_focalnet(net, "both", verbose=True)
    keys_after = {k for k in net.state_dict() if k.startswith("baseball.")}
    added = keys_after - keys_before
    removed = keys_before - keys_after
    print(f"[smoke-1] patch_focalnet(both): {n_blk} 个 block; baseball_keys {len(keys_before)} -> "
          f"{len(keys_after)} (+{len(added)} 新增, -{len(removed)} 丢失)")
    assert not removed, f"patch 丢掉了已有 baseball key: {sorted(removed)[:5]}"
    assert all(is_mona_name(k) for k in added), "新增 key 里有非 mona 的"
    miss1, unex1 = net.load_state_dict(sd_official, strict=False)
    miss1_mona = [k for k in miss1 if is_mona_name(k)]
    miss1_core = [k for k in miss1 if not is_mona_name(k)]
    print(f"[smoke-1] patch 后加载 rdnet.pth: missing={len(miss1)}（非 mona {len(miss1_core)} / "
          f"mona 新增 {len(miss1_mona)}）unexpected={len(unex1)}")
    assert len(miss1_core) == 0 and len(unex1) == 0, "patch 后非 mona 权重必须全命中"

    # ---------------- 2) 冻结正确 ----------------
    n_unfrozen = freeze_backbone(net, unfreeze_mona=True)
    m_n, m_t, b_n, b_t, o_t, o_f = rg_stats(net)
    bb_all = sum(p.numel() for n, p in net.named_parameters() if n.startswith("baseball."))
    mona_all = sum(p.numel() for _, p in mona_params(net))
    tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    other_tr = sum(p.numel() for n, p in net.named_parameters()
                   if p.requires_grad and not n.startswith("baseball."))
    allp = sum(p.numel() for p in net.parameters())
    print(f"[smoke-2] 显式解冻 mona 参数 {n_unfrozen} 个(张量); "
          f"mona 张量 {m_n} 个 requires_grad=True {m_t} 个; baseball.* 张量 {b_n} 个 "
          f"requires_grad=True {b_t} 个; 其余张量 True {o_t}/False {o_f}")
    print(f"[smoke-2] 参数量: 总 {allp/1e6:.2f}M = baseball {bb_all/1e6:.2f}M(全冻) + mona "
          f"{mona_all/1e6:.3f}M + 其余 {other_tr/1e6:.2f}M; 可训 {tr/1e6:.2f}M "
          f"(plain {tr_plain/1e6:.2f}M, Δ={(tr-tr_plain)/1e6:+.3f}M)")
    assert m_t == m_n and m_n > 0, "mona 参数必须全部可训"
    assert b_t == 0, "baseball.* 必须全冻"
    assert o_f == 0, "decoder/头等其余参数必须全可训"
    assert other_tr == plain_other, "非 baseball 可训量应与 plain 一致"
    assert abs((tr - tr_plain) - mona_all) < 1, "可训增量应恰为 mona 参数量"

    # ffn 模式的参数量（单开一个模型量一次）
    net_ffn = make_net()
    patch_focalnet(net_ffn, "ffn")
    mona_ffn = sum(p.numel() for _, p in mona_params(net_ffn))
    n_mona_mod_ffn = len([n for n, _ in net_ffn.named_modules() if is_mona_name(n)])
    n_mona_mod_both = len([n for n, _ in net.named_modules() if is_mona_name(n)])
    print(f"[smoke-2] mona 模块数: both={n_mona_mod_both} ffn={n_mona_mod_ffn}; "
          f"参数量: both={mona_all/1e6:.3f}M ffn={mona_ffn/1e6:.3f}M")
    del net_ffn

    # ---------------- 5) 存档往返（评测路径） ----------------
    print("=" * 100)
    ck = torch.load(args.ckpt, map_location="cpu")
    sd_mona = ck.get("icnn", ck)
    n_sd_mona = len(mona_state_dict_keys(sd_mona))
    net2 = make_net()
    patch_focalnet(net2, ck.get("peft", {}).get("mode", "both"))
    miss2, unex2 = net2.load_state_dict(sd_mona, strict=False)
    miss2_mona = [k for k in miss2 if is_mona_name(k)]
    miss2_core = [k for k in miss2 if not is_mona_name(k)]
    print(f"[smoke-5] 往返 ckpt={args.ckpt} peft={ck.get('peft')} iter={ck.get('iter')}")
    print(f"[smoke-5] 重构造+patch+加载: missing={len(miss2)}（mona {len(miss2_mona)} / 非 mona "
          f"{len(miss2_core)}）unexpected={len(unex2)}; ckpt mona keys={n_sd_mona} "
          f"net mona 参数={len(mona_params(net2))}")
    assert len(miss2) == 0 and len(unex2) == 0, "存档往返必须 0 missing / 0 unexpected"
    del net2

    # ---------------- 6) 评测脚本断言（必须报错而非静默） ----------------
    print("=" * 100)
    import eval_rdt_tint
    for mona, path, want_ok in (("both", args.ckpt, True),          # 正例：mona ckpt + --mona both
                                ("none", args.ckpt, False),         # 反例：mona ckpt + --mona none
                                ("ffn", args.ckpt, False),          # 反例：模式不匹配
                                ("both", args.plain_ckpt, False)):  # 反例：plain ckpt + --mona both
        try:
            net_e, net_c, _ = eval_rdt_tint.build(path, None, mona=mona)
            ok, msg = True, "加载成功"
            del net_e, net_c
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"
        tag = "OK" if ok == want_ok else "!! 不符合预期 !!"
        print(f"[smoke-6] --mona {mona:4s} + {os.path.relpath(path, PROJ)}: 预期"
              f"{'通过' if want_ok else '报错'} -> {'通过' if ok else '报错'} [{tag}] {msg}")
        assert ok == want_ok, f"smoke-6 失败: mona={mona} path={path}"
    print("=" * 100)
    print("[smoke] 1/2/5/6 全部通过")


if __name__ == "__main__":
    main()
