# Stage B-main (lite) 预登记 — SFAM vs FLOPs-matched plain，3 seeds × 24k

- 日期：2026-11-06
- 授权：Lead m04025（早评 Go：D2−M1=+0.3243 dB，CI[+0.1358,+0.5129]，12/15）
- 状态：**先于任何 B-main 训练启动写定**（写定时 `/tmp/ri_dev/kd0bmain/runs/` 下无任何 ckpt；见 §7）
- 目的：把「SFAM 相对 FLOPs 匹配 plain 的增益」从单 seed 12k 的观测，升级为 **3 seeds × 24k** 的主判决；并给出相对 LightRR best-of-2（20.221）的配对差距与守门指标。

## §1 臂与冻结配置（6 runs）

| 臂 tag | 结构 | 损失 | KD | seed | det_seed |
|---|---|---|---|---|---|
| `D2_s101` / `D2_s202` / `D2_s303` | SFAM v2（14,431,735 params / 80.109 GFLOPs@320² b1） | char(1e-3)+0.2·SSIM+0.1·Haar | feature-KD（教师 FocalNet stage1–3，训练期 1×1 投影） | 101/202/303 | 1207/1308/1409 |
| `M1_s101` / `M1_s202` / `M1_s303` | plain NAFNet w41（27,990,252 params / 81.417 GFLOPs；FLOPs 匹配 +1.63%） | char(1e-3)+0.2·SSIM+0.1·Haar | feature-KD（同上；学生通道 82/164/328→教师 192/384/768） | 101/202/303 | 1207/1308/1409 |

- 共同冻结：池 `data/train_sets/real_pairs_1020` may130+jun44=**174 对**（pair_hash `b4eb0f811b7af920c73cc8b48e85994a4d3f3895a620bdd60e498e0c4f49280c`）；教师 `runs/quiet_abl_1106/full2_s101/ckpt_final.pth` sha256 `acb2dd068c106580e01740c7e19d8c77cc16a07dc8be4181cc82d3ffa891e5c2`；init 全臂共享同一副本（D2：`/tmp/ri_dev/kd_sfam_init_v2_s101.pth` sha `9d2cad5e2d47a0ce14129043e44749dfcee58c975dbb10024d927f1d9ce64d81`；M1：`/tmp/ri_dev/kd_stageb_mini_init_nafnet41_s101.pth` sha `a1eb5bf70fed089ac37cbd4ffa728b8aad35ed89032a204363d51d8c41d57f46`）。
- 优化：24,000 it；crop 224/256/320 各 4,000/4,000/16,000（与 Stage A/B-mini 相同 `target_size(it)` 规则）；bs4；AdamW lr2e-4 β(.9,.999) wd0；warmup 500 + cosine→2e-6；AMP；clip1；cudnn deterministic。
- ckpt：`--ckpt-every 12000` ⇒ 每臂保存 `ckpt_012000.pth` 与 `ckpt_024000.pth`（终评主点=24k；12k 点为副口径）+ `ckpt_final.pth`；**无 best-of**。
- 数据流：`S2Dataset` det 模式（epoch=0，RNG=SeedSequence([det_seed, epoch, idx])）。两个 trainer 的 DataLoader shuffle 种子规则沿用各自 Stage A 实现（D2 用 det_seed；M1 用 seed+1000·size-switch-it），**保持不变**以保证与各自 Stage A arm 可比；该差异披露于 §5。
- 运行：每臂 1 张空卡（mem<1500 MiB 视为空），nohup setsid；GPU2 的 S24 训练不受影响，GPU4 若被他人占用则避开。

## §2 评测口径

- 15-scene development panel `data/test_sets/real_pair_1019`；预测 uint8 PNG；`scripts/table_final_1019_annot_1031.py --tag kd0bmain --hole-mask-dir data/annot_1019/omega_masks_1030`；指标 `scripts/metrics_ext_1019.py`。
- 主统计：**3-seed 场景级平均**——每个 (arm, timepoint, scene) 先对 3 个 seed 的 PSNR 取平均，再与另一 arm 做**场景轴配对 t 检验（df=14，双侧 95% CI）**；同时报逐 seed 配对与 MDE=(t975+t80)·se。
- 第二口径：逐 seed 的 D2−M1（3 个 Δ 值）作为一致性检查。
- LightRR：读 `eval/table_final_1019_lightrrft1106.json`，逐场景取 `lightrrft_lr1e4`/`lightrrft_lr5e5` 的 **max（best-of-2）**，与 D2_24k 3-seed 场景均值配对 Δ+95% CI+胜场。
- 守门（与 M1_24k 比较）：PSNR_Ω（`psnr_hole`）、HF-PSNR（`hf_psnr`）、SSIM 的配对 Δ+CI，判据「不显著劣于」= CI 下界 ≥ −0.10 dB（PSNR_Ω/HF）与 ≥ −0.002（SSIM）。
- 效率：params/FLOPs@320² b1/FP32 延迟 b1/b4/峰值显存 单列表（SFAM v2 vs M1 w41 vs 教师参照）。

## §3 判据（写死；来自 Lead m04025）

1. **主门（架构）**：3-seed 场景轴配对 **D2−M1 均值 ≥ +0.15 dB 且 95% CI 下界 > 0**。**24k 口径为主，12k 为副**。
2. **对 LightRR**：D2(24k, 3-seed 均值) 相对 LightRR best-of-2 20.221 的配对 Δ，报点估计+95% CI+胜场（不做显著性门，只报差距）。
3. **守门**：PSNR_Ω / HF-PSNR / SSIM 不显著劣于 M1（定义见 §2）。
4. **三值判决**：Go / No-Go / 不可判；MDE 写清。
   - Go：24k Δ̂ ≥ +0.15 且 CI 下界 > 0。
   - 不可判：|Δ̂| < MDE 且 CI 跨 0。
   - No-Go：其余（Δ̂ < +0.15 且不属不可判；或 Δ̂ < 0 且 CI 上界 < 0）。
5. 绝对量参照（只作背景，不作判据）：D2_24k 若到 ~20.4–20.6 才值得考虑 Stage C（更长训练/90k 全协议）；到不了则收线在「FLOPs 匹配下结构+KD 增益 +0.3 dB」的工程结论。教师 full2 历史 22.57（同池近记忆化，不作公平比较），LightRR best-of-2 20.221。

## §4 预算 / ETA

- 6 runs × 24k it：D2（SFAM）≈4.3–4.8 h/run；M1（w41+teacher）≈4.3–4.7 h/run ⇒ 并行墙钟 ≈5 h，**≈26–28 GPU·h**（Lead 未设硬预算；按「lite」量级执行，如实报）。
- 预计启动：2026-11-06 约 17:45；完成 ≈22:30–23:00；评测约 20–30 min；报告随之生成。

## §5 风险与必须披露项

1. 模型 init 全臂共享同一副本，seed 只改变数据流（det_seed/loader 顺序）⇒ seed 方差可能偏小，3-seed CI 不能当作完整训练方差；须与「同一配置跨日期重跑差 +1.0 dB」的 run-level 不确定性一起读（历史 quiet ablation 的教训）。
2. 两个 trainer 的 loader shuffle 种子规则不同（§1）——这是 Stage A 已冻结的实现，不改，以免破坏与已有 arm 的可比性；在报告里明写。
3. 12k 点是在 24k cosine 计划下的中间点，**不等于** Stage A/B-mini 的 12k 训练（后者 cosine 到 12k 结束）；12k 只作副口径。
4. 教师在同一 174 对池上训练（teacher_psnr_vs_GT=25.964 dB）⇒ 学生可能近记忆化；B-main 仍沿用该池（与 Stage A 同），论文口径需按「pilot 池」声明。
5. 参数不匹配：M1 参数 1.94× SFAM（FLOPs 匹配）；该取舍是先验登记的选择，对结构主张更保守；报告中同时给出两者效率表。
6. GPU 共享：只用 mem<1500 MiB 的空卡；不动他人进程；单次长任务用 nohup setsid。
7. 单 seed 的 Stage A/B-mini 数字只作方向参照，不与本批 3-seed 结果混算。

## §6 产物路径

- 训练：`/tmp/ri_dev/kd0bmain/runs/{D2_s*,M1_s*}/`；日志 `/tmp/ri_dev/kd0bmain/logs/*.log`。
- 预测：`runs/pred/real_pair_1019_kd0bmain_<tag>_<12k|24k>/`。
- 权威表：`eval/table_final_1019_kd0bmain.{md,json,csv}`。
- 报告：`/tmp/ri_dev/SFAM_STAGE_B_MAIN_1106.md`（本地镜像 `code/local_tools/`）。
- 独立核验：`/tmp/ri_dev/verify_stagebmain_1106.py`。

## §7 写定时状态（启动前）

- `/tmp/ri_dev/kd0bmain/runs/` 不存在任何 ckpt（本文件写入时未启动任何 B-main 训练）。
- GPU 快照（17:35 左右）：0/1/3/5/6/7 空（≤639 MiB），2=S24 训练（~7.9 GB），4=他人 5.9 GB。
- 初始 ckpt/日志/报告均未生成；判据与对比组不因后续任何结果改动。
