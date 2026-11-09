# 预登记：Stage B-mini — FLOPs 匹配的 plain NAFNet 对照（M0/M1）+ SFAM 24k 延长（S24）

- 日期：2026-11-06（本文件在 M0/M1/S24 训练启动之前写定；见 §7「写定时状态」）
- 上游指令：Lead m03400（2026-11-06 15:26）。目标：拆掉 Stage A-bis 中「容量/算力 vs 结构」的混淆。
- 数据面板：15 场景 development panel（`data/test_sets/real_pair_1019`），annot_1031 手标掩码；uint8 PNG 口径；无 best-of-2。
- 三条披露先行：**(1) 纯加宽无法同时匹配 SFAM 的参数与 FLOPs**（见 §1.3，M0/M1 参数 1.94×，FLOPs +1.63%）；**(2) 单 seed / 12k it（S24 24k）**，MDE 约 0.26–0.30 dB，+0.15 门低于 MDE ⇒ 只作方向观测；**(3) 不可归因声明**：即使主门成立，本实验只支持「在 FLOPs 匹配下 SFAM+feature-KD 优于 plain+feature-KD」，不支持「空间-频率结构普遍更优」。

## §1 臂定义（4 臂，M0/M1 各 12k it；S24 24k it；S24b 不跑，见 §5）

### §1.1 共同输入
- 训练池：未清洗 `data/train_sets/real_pairs_1020`，may 130 + jun 44 = **174 对**；pair_list hash **b4eb0f811b7af920c73cc8b48e85994a4d3f3895a620bdd60e498e0c4f49280c**（与教师/缓存/Phase0/Stage A 完全同一池）。
- 教师：`runs/quiet_abl_1106/full2_s101/ckpt_final.pth` sha256 **acb2dd068c106580e01740c7e19d8c77cc16a07dc8be4181cc82d3ffa891e5c2**（冻结，stop-grad）。
- 输出 KD 伪标签（仅 S24b 会用；本批不跑）：`data/pools/kd_teacher_full2s101_1106/`（与 Phase0 B 同源）。

### §1.2 M0：FLOPs 匹配 plain NAFNet + 契约损失、无 KD（12k it）
```
NAFNet(img_channel=3, width=41, middle_blk_num=1,
       enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1], global_residual=True)
```
- 纯加宽，无结构改动；与 E2 的**唯一变量 = width 26→41**（损失/crop/池/seed/det-seed 全同）。
- 契约 GT 损失：`L_GT = charb(A,T;ε=1e-3) + 0.2·SSIM + 0.1·Haar`（10 个 Haar 子带等权，与 Stage A C/D、Stage A-bis E 臂逐字相同）；按 `w_src`（真实 1.0 / 合成 1.0 / 开源 0.5）加权，仅真实/合成参与；无 KD 项。

### §1.3 宽度选择依据（`kd_stageb_width_scan_1106.py` 实测，320² b1，FlopCounterMode）
| width | params | GFLOPs | FLOPs/SFAM | params/SFAM |
|---|---|---|---|---|
| 26（E1/E2 用） | 11,339,097 | 33.099 | 0.4132 | 0.7857 |
| 36 | 21,617,967 | 62.935 | 0.7856 | 1.4979 |
| 40 | 26,650,043 | 77.531 | 0.9678 | 1.8466 |
| **41（选定）** | **27,990,252** | **81.417** | **1.0163** | **1.9395** |
| 42 | 29,363,337 | 85.399 | 1.0660 | 2.0346 |
- SFAM v2 参照：**14,431,735 params / 80.109 GFLOPs**（`kd_sfam_test_v2_1106.json`）。
- **选定 width=41**：FLOPs 81.417G vs 80.109G = **+1.63%**（满足 Lead「FLOPs 差 ≤10%」）；提升 w=40（−3.23%）与 w=42（+6.60%）同样在门内，取最接近者。
- **披露**：纯加宽不能同时匹配参数；参数匹配的纯宽度是 w≈30（15,056,133 params / 43.893G = 0.548× FLOPs），会违反「FLOPs 匹配」这一决定性判据，故不选。**M0/M1 参数 27.99M = 1.94× SFAM，对结构主张更保守（plain 对照参数占优）**。

### §1.4 M1：M0 + feature-KD（12k it）
- 结构/K 同 M0；唯一新增 = 冻结教师前级 feature-KD：
  - 学生特征：`net.downs[0..2]` 输出（hook，stride 2/4/8，通道 `2w/4w/8w = 82/164/328`）；
  - 教师特征：`teacher.baseball(x_in)` 的 `x_base[0..2]`（stride 2/4/8，通道 192/384/768，**不用 stage 4**）；
  - 训练期可学 1×1 投影 `A_j`（82→192 / 164→384 / 328→768），**推理删除**；
  - `L_KD-F = (1/3)Σ_j mean((A_j X_j − sg V_j)²) / (sg mean(V_j²) + 1e-6)`，`V_j = Resize(sg Q_j, h_j, w_j)`；
  - `λ_f(t) = 0.01·min(1, t/1200)`；仅真实样本（`meta[:,4] < 0.5`）；教师全程 `eval()` + `no_grad`。
- 与 E1（plain-26 + feature-KD）的唯一变量 = width（26→41）⇒ `M1 − E1` 即「同损失同 KD 下纯加宽的作用」。

### §1.5 S24：SFAM v2 + feature-KD，24000 it（12k/24k 两点评测）
- 架构/init：SFAM v2（14,431,735 params，`kd_sfam_init_v2_s101.pth` sha256 **9d2cad5e2d47a0ce14129043e44749dfcee58c975dbb10024d927f1d9ce64d81**）。
- 损失/KD/优化器/data pipeline 与 D2 完全相同，**唯一改动 = iters 12000→24000 与 crop 分段同比延长**：
  - 12k：224/256/320 各 4000 it；**24k：各 8000 it**（分段比例保持 1:1:1）；
  - LR：warmup 500，cosine 24000 it → 2e-6；λ_f = 0.01·min(1, t/2400)（warmup 10% 随 iters 等比放大）；
  - ckpt：`ckpt_012000.pth`、`ckpt_024000.pth`（ckpt-every 12000）。
- **披露**：S24 的 12k 点处于 24k cosine 中段、crop 也在 224 段（因分段 8000），**不等于 D2 的 12k checkpoint**；因此 12k→24k 斜率只在 S24 自身曲线上解释，D2/A/B 的数字仅作绝对水平参照。若需 D2 的「12k→24k 延长」严格对照须另立预登记（本轮不做）。
- S24b（SFAM + output-KD 24k）：**本轮不跑**（预算，见 §5）。

## §2 共同冻结协议（所有新臂一致）
- 数据流确定性（Phase0 P0-1 已验收）：`S2Dataset(det=True, det_seed=1106)`，per-sample RNG = `SeedSequence([1106, epoch, idx])`；DataLoader 固定 `torch.Generator` + `worker_init_fn`；cudnn `benchmark=False, deterministic=True`。
- 训练：bs 4；AdamW lr 2e-4，betas (0.9,0.999)，eps 1e-8，**wd 0**；warmup 500 线性、cosine → 2e-6；AMP（fp16 autocast + GradScaler）；`clip_grad_norm_` 1.0；seed=101；iters 见各臂。
- crop 调度：224→256→320（按 §1.5 分段）；crop 采样与 E1/E2/D2 完全同一 draw plan（同 det_seed、同池）。
- 产出：每 2000 it ckpt（S24 每 12000）；**终评固定用最终 ckpt，不做 best-of-2**；训练日志 `train_log.csv`、可选 `loss_dump.txt`、`manifest.json`（脚本/init/pair hash/教师 sha/协议）。
- 评测：`scripts/eval_arm_1019.py` 口径 = 原尺寸推理、写 uint8 BGR `t_hat.png`；出表 `scripts/table_final_1019_annot_1031.py --tag kd0mini --hole-mask-dir data/annot_1019/omega_masks_1030`；指标 `scripts/metrics_ext_1019.py`（β/动作比/LapVar 等）。
- 统计：场景轴配对 t（df=14，双侧 95% CI）；MDE = (t₀.₉₇₅ + t₀.₂₀)·se；报告逐场景胜场数；Holm 仅用于描述性多组比较。

## §3 判据（Lead m03400 写死，不得事后修改）
### §3.1 主门（算力混淆是否解释结构增益）
- **D2(12k) − M1(12k)**（配对 Δ，95% CI，n=15）：
  - **≥ +0.15 dB 且 CI 下界 > 0 ⇒ 结构增益超出算力解释** ⇒ 进 Stage B-main（3 seeds，12k/24k + 匹配对照），另令；
  - **|Δ| ≤ 0.10 dB ⇒ 容量解释** ⇒ **停结构主张**；记负结果台账，ICME 正稿维持 Plan B（不追加 seed、不做 Stage B-main）；
  - **中间情形（0.10 < Δ < 0.15，或点估计 ≥0.15 但 CI 跨 0）⇒ 交 Lead 决定**，不得自行宣布。
- 辅助（描述性，不进主门）：D3−M1（SFAM+两 KD vs plain+feature-KD）、**M1−M0**（宽 plain 上 feature-KD 的真实增益，与 E1−E2 对照）、**M1−E1**（同损失同 KD 下纯加宽的效应，即「算力边际收益」）、M0−E2（同损失下纯加宽无 KD 的效应）。
### §3.2 S24（容量线对 SFAM 是否还活着）
- **S24@24k ≥ 20.3 且仍在上涨（S24_24k > S24_12k）⇒ 容量线未死** ⇒ Stage B-main 预算可到 3 seeds × 24k；
- **S24@24k ≤ 20.0 或掉头（S24_24k ≤ S24_12k）⇒ 收线**：即便架构线也无法公平赢 LightRR（best-of-2 **20.221**）；
- 20.0–20.3 或跨 0 的斜率 ⇒ 如实报不可判，交 Lead。
### §3.3 守门（同报，不单独判）
- M1 vs D2：PSNR_Ω、HF-PSNR、SSIM 均不得显著劣于 D2 超过 Stage A 口径（PSNR_Ω/HF ≥ −0.10、SSIM ≥ −0.002）；LapVar 对照 GT 39.6。
- 效率表必须给：参数量、FLOPs@320² b1、FP32 延迟 b1/b4、峰值显存；**明确列出 M1/D2 FLOPs 比 = 81.417/80.109 = 1.016（≤10% 要求满足）**。

## §4 已知风险与偏差披露
1. **参数不匹配**（§1.3）：M0/M1 1.94× SFAM 参数。若 M1 仍输给 D2，结论「算力解释不了结构增益」更强；若 M1 明显更强，则说明参数容量而非 FLOPs 是主因，**不得声称结构无用**，只能报「本对照下无法区分参数与结构」。
2. **单 seed**：D2−M1 的 MDE ≈ 0.26–0.30（场景 sd 与 Stage A 同量级）；+0.15 门低于 MDE ⇒ 命中时只作方向观测，Stage B-main 才做 3 seeds。
3. **S24 12k 点非 D2**（§1.5）：不得把 S24_12k 与 D2 当作同 checkpoint；曲线解读仅在 S24 自身。
4. **教师近记忆化**：教师在同 174 对训练池上 teacher-vs-GT ≈ 25.96–27.8 dB（测试 22.57），feature-KD 上界被结构性压缩；对 M1 与 D2 同等影响，不构成单侧偏差。
5. **S24b 未跑**（预算）：若 Lead 要「长预算下哪种 KD 稳」，需追加 ~3.8 GPU·h。
6. 契约损失改变纹理（Stage A-bis 已见 LapVar 48–49 vs A/B 54.3），M0/M1 与 D2 的差异中可能含纹理偏好，**报告全指标对照而非只报 PSNR**。
7. 共享机纪律：只抢空卡（<1500MiB），不杀他人进程，不改 batch/crop。

## §5 预算与止损
- 估算 GPU·h：M0 ≈ 3.0（w41 FLOPs 2.46× w26，12k）、M1 ≈ 3.0、S24 ≈ 3.8（SFAM 12k ≈1.9h ×2）⇒ **合计 ≈ 9.8 GPU·h ≤ 10**；3 卡并行，墙钟 ≈ 4 h。
- 止损：单臂 > 5 h 或 OOM 无法在 bs4/320² 下复现协议 ⇒ 停下报 Lead，不擅自改 batch/crop；M0/M1 若 320 段显存不足，允许的先验修复只有「AMP 保持 + 显存清理」，不允许降 batch。
- S24b 只有在 3 臂提前完成且 Lead 追加批准时再跑。

## §6 产物与路径
- 训练：`/tmp/ri_dev/kd0mini/runs/{M0,M1,S24}/`（含 init_copy.pth、manifest.json、train_log.csv、ckpt）；日志 `/tmp/ri_dev/kd0mini/<arm>.log`。
- 预测：`runs/pred/real_pair_1019_kd0mini_<arm>/`（S24 两个点各一份：`kd0mini_S24_12k`、`kd0mini_S24_24k`）。
- 表/指标：`eval/table_final_1019_kd0mini.{md,json,csv}`、`/tmp/ri_dev/kd0mini_metrics_ext.json`。
- 效率：`/tmp/ri_dev/kd_stageb_mini_eff_1106.json`（w41 params/FLOPs/延迟/显存）。
- 报告：`/tmp/ri_dev/SFAM_STAGE_B_MINI_1106.md`；本地镜像 `code/local_tools/`（含本预登记）。
- 独立核验：`verify_stagebmini_1106.py`（不依赖报告脚本重算主表/主门/CI/效率比）。

## §7 写定时状态（先于训练，时间戳证据）
- 本文件写定：2026-11-06 15:4x（服务器 `date` 于上传时记录）；此时 **M0/M1/S24 均未启动、无任何 ckpt**；已完成的只有：宽度 scan JSON（`/tmp/ri_dev/kd_stageb_width_scan_1106.json`）、init 生成（`kd_stageb_mini_init_nafnet41_s101.pth`）、代码适配（`kd_p0_contract_train_1106.py` 加 `--width`/M0/M1；`kd_sfam_train_v2_1106.py` 加 S24/S24b；`kd_p0_infer_1106.py` 加 `--width`）。
- 启动前门：M0/M1/S24 smoke（各 30 it）rc=0；M1 在 320²/bs4 峰值显存 < 9.0 GB（实测见 `/tmp/ri_dev/kd0mini/mem_probe.txt`）；≥3 张空卡。
- 判决回填：§8（跑完后填，不得改 §3 判据）。

## §8 结果与判决（待填）
