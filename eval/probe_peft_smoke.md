# Mona-PEFT 冒烟验证（S1-PEFT 线）

日期 2026-09-17 · 卡 GPU1/GPU2（8×3080 10GB，实测可用 9.65GB）· 环境 `/DataDisk/miniconda/envs/xreflection`
产物：`scripts/rdnet_peft.py`（Mona 包装/插入）· `scripts/finetune_rdnet_peft.py`（训练）·
`scripts/eval_rdt_tint.py`（+`--mona`）· `scripts/probe_peft_smoke.py`（本报告 1/2/5/6 项探针）

## 0. 设计要点（与方案的三处强制约定）

1. **插入点**：`FocalModulationBlock.forward` 里"调制残差之后"挂 `mona1`、"FFN 残差之后"挂 `mona2`
   （对齐官方 `SwinTransformerBlock` 的 `my_module_1/2`，都挂在残差后）。`hw_shapes=(self.H,self.W)`
   由 `BasicLayer.forward` 写入，与 `assert L == H*W` 同源。
2. **key 安全**：只用 `add_module("mona1"/"mona2")` 追加，不改/不重命名任何已有子模块；block 的类用
   `blk.__class__ = MonaFocalModulationBlock`（子类，实例布局相同，`_modules/_parameters` 原样不动）。
   `patch_focalnet` 必须在 `FullNet_NLP(...)` 构造**之后**调用（构造内部 strict 加载 `focal.pth`）。
3. **冻结**：mona 参数也在 `baseball.*` 前缀下 → 冻 `baseball.` 之后必须显式解冻 mona
   （`set_mona_trainable`），脚本里已打印条数。

**额外的一项改动（方案外，但为 10GB 卡所必需）**：`--grad-ckpt on`（默认）对每个 FocalNet block 做
`torch.utils.checkpoint(use_reentrant=False)`。原因：mona 可训 = 冻结的 FocalNet 前向也被拉进 autograd 图
（梯度要穿过它回到上游 mona），focal modulation 的 unfold 中间量（B,C,H,W,k²）全部留到反向 →
224²bs8 **直接 OOM**、bs4 就要 8.72G。checkpoint 重算换显存，数学等价（RNG 由 checkpoint 自持，
Mona 的 dropout 掩码重算一致），代价约 1.2–1.4× 时间。超参（30k iter / lr 1e-4 / cosine / AMP /
`segs=[(224,8),(256,6),(320,4)]` / 数据集 / seed）与 plain 完全一致。段切换判据加了显式 boundary break
（原脚本靠 epoch 边界；默认 `seg_iters=2000` 与 epoch 500 步整除，两者等价）。

## 1. 权重完整性：patch 前后加载 rdnet.pth

```
patch 前  : missing=0 unexpected=0  baseball_keys=504
patch(both): 24 个 block, 挂上 48 个 Mona; baseball_keys 504 -> 1272 (+768, -0)
patch 后  : missing=768 (= 非 mona 0 + mona 新增 768) unexpected=0
```
- 504 → 1272 的 768 个新增 key **全部**是 mona（`all(is_mona_name(k))` 断言通过），已有 504 个 key 一个不丢。
- patch 后的 768 missing 就是新增 Mona 参数本身（官方 rdnet.pth 里没有），**非 mona 部分 missing=0**；
  unexpected=0 双向确认没有 key 错位 → `baseball.*` 预训练权重照常命中。

## 2. 冻结正确性 / 参数量

```
plain(未 patch) 冻 baseball 后可训 60.68M            ← 与既有主臂 60.7M 一致
both : mona 张量 768/768 requires_grad=True; baseball.* 张量 504, True=0; 其余张量 1250, True=1250/False=0
       总 271.70M = baseball 211.01M(含 mona, 全冻) + 其余 60.68M
       可训 65.95M = plain 60.68M + 5.268M
ffn  : mona 模块 24 个 / 参数 2.634M（both 是 48 个 / 5.268M）
```
- mona 非 mona 的 baseball = 211.01 − 5.268 = **205.74M**，与已知"冻 205.7M"吻合。
- 可训增量 +5.27M ≈ 预期 +5.3M（逐层解析值 132·d+9728：192×2 + 384×2 + 768×18 + 1536×2 层）。
- decoder/头等非 baseball 参数全为 True 且参数量与 plain 完全相同（断言 `other_tr == plain_other`）。
- ffn 训练实跑：trainable 63.3M / 269.1M（60.68 + 2.63）。

## 3. 前向/反向：100 iter（GPU1, 224² bs8）

命令 `... --mona both --grad-ckpt on --iters 100 --batch 8 --seg-iters 100000 --log-every 10`
（日志 `/tmp/log_peft_smoke100.txt`，ckpt `/tmp/peft_smoke100/ckpt_final.pth`）

```
it   1 L_t=0.1760 L_r=0.1016 total=0.2808  peak=6.60G
it  20 L_t=0.1277  it 50 L_t=0.0788  it 80 L_t=0.1064  it 100 L_t=0.1038
it 100 total=0.1894  sec=177  peak=7.10G      (100 iter 净耗时 109s = 1.10 s/iter)
```
- L_t 0.176 → 0.104，无 NaN，单调性符合"随机档位 + 随机裁剪"的抖动预期（与 plain 同量级：plain it1 L_t=0.181/0.197）。
- 峰值 7.10G（含 static：271.7M 参数 fp32 + 65.9M 梯度 + Adam 两态 ≈ 1.9G）。

## 4. 显存分段（每档 20 iter，batch 随 plain 的 seg 表）

命令 `... --iters 60 --seg-iters 20 --log-every 10`（日志 `/tmp/log_peft_seg2.txt`）

| 段 | 配置 | peak | 稳态速度 |
|---|---|---|---|
| 224² | bs8 | **7.09G** | 1.10 s/iter |
| 256² | bs6 | **6.98G** | ~1.1 s/iter |
| 320² | bs4 | **7.26G** | ~1.2 s/iter |

- 三档全部 < 9G，卡容量 9.65G → **不 OOM，可以开训**。
- 对照（同机实测）：plain 224²bs8 = 6.18G（脚本注释记 6.3G，一致）；mona **不**开 checkpoint：
  224²bs8 直接 OOM，224²bs4 = 8.72G（且 8.7G 只跑 4 张图，样本效率反而低 2×）。
- 段切换时首 ~10 iter 有一次性 ~66s 预热（GPU 侧，疑 cuDNN v8 plan/分配器重缓存；原脚本切段亦有 ~15s）；
  真实 2000-iter 段下摊薄 ~3%，不影响。
- 30k iter 预估 **~9.5h**（三档 1.1–1.2 s/iter，5 个 6000-iter 循环），对照 plain 实测 5.47h。

## 5. 存档往返（评测路径可用）

```
ckpt peft={'mode': 'both'} iter=100
重建 + patch(both) + load_state_dict: missing=0（mona 0 / 非 mona 0）unexpected=0
ckpt mona keys=768 == net mona 参数 768
```
- eval 端：`build(ckpt, mona="both")` 打印 `[eval-rdt] mona=both 已加载(missing=0) keys=768 ckpt_peft={'mode': 'both'}`。
- 端到端实跑（`--mona both --sets synth_rrw_v21_0913 --no-save-pred`）：12 场景 × 3 档 34s，
  `gm−ingm` +2.49/+2.84/+2.96 dB（α=1.0/0.6/0.3）——100-iter 半成品 ckpt，只验证通路不出数。
- resume 实跑：`--resume .../ckpt_final.pth --iters 105` 从 it=100 接上跑到 105 正常存档；
  resume 模式不匹配（both ckpt + `--mona ffn`）会 `RuntimeError: mona 模式=both ≠ --mona ffn` 拒绝续训。

## 6. 评测断言一致性（必须报错而非静默）

| `--mona` | ckpt | 期望 | 实际 |
|---|---|---|---|
| both | mona ckpt | 通过 | 通过（missing=0） |
| none | mona ckpt | 报错 | `RuntimeError: --mona none 但 ckpt 含 768 个 mona key(net=0, 会被整体丢弃)：请用 --mona both/ffn` |
| ffn | mona(both) ckpt | 报错 | `RuntimeError: mona 权重未完整加载: ckpt=768 net=384 missing=0` |
| both | plain ckpt | 报错 | `RuntimeError: --mona both 但 ckpt 不含 mona 权重：会静默退回 plain，拒绝执行` |

- 方向说明：`--mona none` 时 net 侧没有 mona 张量，ckpt 里的 768 个 mona key 只能表现为 **unexpected=768**
  而不是 missing（load 行打印 `missing=0 unexpected=768`）。两种方向都意味着"mona 权重会被静默丢弃/错配"，
  断言均已在 load_state_dict 之后立刻拦截。
- 回归：`--mona` 默认 none，plain ckpt 回归实跑正常（rdt_plain, gm−ingm +5.83/+4.70/+3.29 dB，28s）。

## 结论

六项全过。**可以开训**，建议：

```
python3 scripts/finetune_rdnet_peft.py --mona both --grad-ckpt on \
    --out runs/ckpt/rdt_mona_s20260915 --iters 30000 --batch 8 --gpu <N>   # 30k ≈ 9.5h，单卡
python3 scripts/eval_rdt_tint.py --mona both --tag rdt_mona_s20260915 \
    --ckpt runs/ckpt/rdt_mona_s20260915/ckpt_final.pth --gpu <N>
```

注意：本报告所有 ckpt 都在 `/tmp`（冒烟产物，未入 `runs/`）；正式训练请用 `--out runs/ckpt/...`。
