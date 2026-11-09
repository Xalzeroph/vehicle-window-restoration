# 预登记 · LightRR 微调后 1019 面板判定（ICME 动机实验）

> 2026-11-06 exp-dev 落盘。**评测尚未开始**；本档先于任何 LightRR-on-1019 结果写死判据。
> 服务器路径 `/tmp/ri_dev/PREREG_lightrr_eval_1106.md`；本地镜像 `code/local_tools/PREREG_lightrr_eval_1106.md`。
> 前置：训练可行性预登记 `/tmp/ri_dev/PREREG_lightrr_ft_1106.md`（工程可行性，已完成；训练于 2026-11-05 20:33 起跑）。
> 纪律：判据写死在开跑之前；任何事后调整必须单独标注并说明理由。1019 一律称 **development panel**。

---

## 0. 一句话

判定「9.5M 级高效轻量 SIRR（LightRR）在贴膜玻璃域**按其官方配方微调后**，是否仍显著低于我们的 RDNet 微调基线」，
作为 ICME 主线「轻量方法在此域不划算」的动机证据；若微调后并不差，则如实报并改叙事。

---

## 1. 训练现状（2026-11-05 21:51 实测，只读检查，未碰训练进程）

| 臂 | LR | GPU | 日志 | ckpt 根 |
|---|---|---|---|---|
| `ft_lr1e4` | 1e-4 | 2 | `/tmp/ri_dev/lightrr_ft_lr1e4.log` | `/tmp/ri_dev/lightrr_ft_ckpt/ft_lr1e4/Nov05_203336_2026/weights/` |
| `ft_lr5e5` | 5e-5 | 3 | `/tmp/ri_dev/lightrr_ft_lr5e5.log` | `/tmp/ri_dev/lightrr_ft_ckpt/ft_lr5e5/Nov05_203341_2026/weights/` |

- 当前 epoch **5/40**（iters ≈ 11.9k / 100k）；epoch 4 的 ckpt 已落盘（`*_004_00010000.pt`）。
- 实测 ~0.38 s/iter、~9.2k it/h ⇒ **ETA 2026-11-06 06:30–07:30**（保守区间；共享卡 GPU2/3 与他人 6.1 GB 作业共卡）。
- 训练环境 `/DataDisk/zmy_code/LightRR_env/lightrr/bin/python`；不启动/不重启/不杀任何训练进程。

---

## 2. 被判定对象（写死）

- 两臂 final ckpt（第 40 epoch；文件名形态 `*_040_0100000.pt`，另存 `*_latest.pt`；若 final 缺失用 latest 并标注）。
- **主判定 = 两臂 best-of-2**（取 1019 PSNR 较高者）——这是"最强基线口径"，防止 straw man；两臂结果分别报。
- 训练配方（写死，结论注脚必须带）：
  - 初始化：官方冻结 `LightRR-CVPR26/weights/lightrr/lightrr_woNature.pt`（9.50 M，epoch 58）；
  - 40 epoch × 2500 it = **100,000 it**，batchSize 1，lr ∈ {1e-4, 5e-5}，**seed 101**；
  - 数据 `/tmp/ri_dev/lightrr_data_fast`（长边缩到 640 加速；训练裁剪 224 不变）：70% 合成（合成源 = 600 张本域 T 图，替代官方 VOC）+ 30% 真实（`/tmp/ri_dev/clean_all_manifest.json` 前 400 对，跨 apr/jun/may 多场景）；
  - 官方验证/测试（Real20-420、SIR2、Nature，普通玻璃）**只用于看有没有训崩**，不外推本域。
- ⚠️ 该配方与 RDNet 基线不同（RDNet：30% 真实 may+jun 174 对 + 40% 在线合成 + 30% SIRR 公开直督、6000 it/bs4）⇒ 见 §7 混淆与措辞边界。

---

## 3. 平凡基线与参照（必须同管线重出/核对）

| 臂 | 全图 PSNR | 来源 |
|---|---|---|
| identity（T̂ = I，分数地板） | 18.44 | `eval/table_final_1019_r11annot.md` |
| LightRR 官方冻结 ckpt 零样本 | **19.13**（verify, n=15）/ 19.18（冻结稿主表） | `eval/real_pair_1019_lightrr_official_verify_summ.json` / `cvpr2027/tab/main_table.tex` |
| RDNet 官方冻结 ckpt 零样本 | 20.05 | r11annot |
| RDNet 域微调 base（`oldpool_s101`） | 21.99（三种子均值 22.49±0.44） | r11annot |
| RDNet 域微调 + L_quiet（`quiet_s101`） | **22.71**（三种子均值 22.99±0.34） | r11annot |

- 19.13 vs 19.18 的 0.05 dB 差是两套推理口径差，**判定前必须先解释**（用本档 §4 的冻结 ckpt 重跑对齐）。
- RDNet 参照 = 冻结稿数字，**本次不重训**，只做同管线出表核对。

---

## 4. 评测口径（写死）

- 面板：`data/test_sets/real_pair_1019`，**15 场景 × 1 对**（`I_000000.png` + `GT.png`），独立采集月（非训练池 apr/may/jun/nov02 的帧）；
  但 **1019 已被反复用于开发（Ω 重标 / 损失消融 / 九臂 / L_quiet 选型）⇒ development panel，不得写 frozen test set / unseen data**。
- 预测：uint8 PNG（BGR 存盘），尺寸与 I/GT 逐场景相同；路径
  `runs/pred/real_pair_1019_lightrr_ft_lr1e4/<scene>__I_000000/t_hat.png`（`ft_lr5e5`、`lightrr_official_rerun` 同构）。
- 出表（与冻结稿同口径）：
  ```bash
  PY=/DataDisk/miniconda/envs/xreflection/bin/python3
  cd /DataDisk/zmy_code/TintedGlass_Enhance
  $PY scripts/table_final_1019_annot_1031.py \
      --pred-map /tmp/predmap_lightrr1106.json --tag lightrrft1106 \
      --hole-mask-dir "$PWD/data/annot_1019/omega_masks_1030"
  ```
- 指标：**PSNR（主判据）**、SSIM、HF-PSNR、梯度余弦、LapVar、PSNR_Ω / 非洞区 / ΔΩ；
  另 `scripts/metrics_ext_1019.py` 出 β（残留反射系数）、α（逐图增益中位）、Δaff、Δpow、Lap 比。
- **管线验证门（先于判定；任一条不过 ⇒ P0，停止、不发言）**：
  1. 既有四条臂（identity / rdnet_zs / oldpool_s101 / quiet_s101）用同一条出表命令重出，须与 `eval/table_final_1019_r11annot.md` 一致到 **±0.02 dB**；
  2. LightRR 官方冻结 ckpt 用同一条推理管线重跑（`lightrr_official_rerun`），须与 verify 的 **19.1295 ± 0.05 dB** 一致；
  3. 完整性 15/15 场景；t_hat 逐场景尺寸 = GT；无 NaN / 全黑 / 全白 / 全常数图；
  4. 若历史表与当前掩码/表脚本口径已变（如 2026-11-03 重标 170621），以"本次同一脚本重出"为准，并在结果里写明两表差。

---

### 4.1 预跑记录（2026-11-06 22:04，**评测前**；只记录管线状态，不改判据）

- epoch-5 稳定 ckpt（`ft_lr1e4_005_00012500.pt`）单场景（`VID_20261019_164257`）冒烟：PSNR **19.385**、SSIM 0.7463（identity 18.580，Δ +0.81）；t_hat 为 8-bit RGB 1280×832，与 GT 同尺寸。
- 权威出表脚本同口径读入该 t_hat：PSNR **19.39**（n=1 行；identity 行 18.44 与 r11annot 一致到 0.00）⇒ **路径约定 + uint8 口径已对齐**。
- 官方冻结 ckpt 同管线重跑同场景：**19.944**（verify summ 该场景 19.8848，差 0.06 dB = float-vs-uint8 / 逐通道平均）⇒ **管线 parity 通过**。
- 已修 codex-dev 草稿两处 bug：① 非 skip 分支 `action` 未赋值导致崩溃；② 内部 PSNR/SSIM 用 RGB 预测对 BGR GT（少报 ~0.9 dB；PNG 存盘本身正确）。修后「脚本自报 = 出表 = 从 PNG 直接重算」三者一致。
- 记录：`/tmp/ri_dev/lightrr_infer_help_1106.txt`、`lightrr_e5smoke_summ_1106.json`、`lightrr_official_parity_summ_1106.json`、`table_final_1019_lightrrft_e5smoke.md`（本地镜像 `code/local_tools/`）。

## 5. 判据（Lead 已定死，不得事后改）

主判据 = **两臂 best-of-2 的全图 PSNR**（15 场景 uint8 口径）：

| 档 | 条件 | 结论 | 论文处置 |
|---|---|---|---|
| **P1** | 最好臂 **≤ 22.2 dB** | 动机**成立** | 可写「按官方配方域内微调 100k it 后，9.5M LightRR 仍比我们的 RDNet+L_quiet 基线（22.71；三种子 22.99）低 ≥0.5 dB」；必须并列披露：单种子、两档 lr、数据配方不同、best-of-2 |
| **P2** | **22.2 < PSNR ≤ 23.0** | 弱成立，需谨慎 | 只能写「未达到我们的基线」；**不得**写「失效 / 不能处理 / 不适用于贴膜域」；不得作为动机主证据，可作辅助 |
| **P3** | **PSNR > 23.0** | 动机**不成立** | 如实报「轻量网络微调后可与我们持平/更好」；ICME 叙事删除「轻量失效」，改其他论证 |
| **P0** | 管线验证门失败 / ckpt 异常 / 场景不全 | 不判定 | 先修管线，不发言 |

**附加约束（防侥幸，写死）**：
1. 两臂相差 > MDE（0.26 dB）且跨档（如 22.15 vs 22.30）⇒ 判「跨档歧义」，取**对动机更不利（更高）档**，并同时披露两臂。
2. 配对 Δ：`LightRR(best) − quiet_s101` 与 `− oldpool_s101` 各给逐场景配对 Δ、95% CI、胜出场景数（脚本 `/tmp/pair_stats.py`；CI = t_{14}·sd/√15，与冻结稿同法）。
3. |配对 Δ| 小于事后重算 MDE 且 CI 覆盖 0 ⇒ **不得**把「低于」写成「显著低于」。
4. 不做事后调参 / 不挑选 epoch 过门（唯一例外见 §5.1 预登记的上界补充）。

### 5.1 预登记的「最强口径」上界（可选，只作上界）

- 若训练日志的 real20 验证曲线（LightRR 官方验证集，普通玻璃，与 1019 无重叠）显示明显更好的 epoch，允许按「real20 最优 epoch」取一个 ckpt 额外评测，命名 `lightrr_ft_*_bestreal20`；
  仅作 **best-effort 上界**敏感性，**不进主判据**；理由：该验证集与判据面板无重叠，不构成在 1019 上调参。
- 若上界 > 23.0 而 final ≤ 22.2 ⇒ 主判据仍 P1，但结果必须同时披露该上界。

---

## 6. 统计功效与种子声明

- n = 15 场景；历史配对 sd ≈ 0.36 ⇒ **MDE ≈ 0.26 dB**（α=0.05 双侧、power 0.8，z 近似 (1.96+0.842)·σ/√15）。
- 三档阈值间距 ≥ 0.8 dB ≈ 3×MDE ⇒ 档位本身可分辨；但 LightRR 与 RDNet 的真实差值若只有 ~0.5 dB，正贴着 MDE ⇒ **必须报 CI**，不能只报点估计。
- 事后用**实测**配对 sd 重算 MDE 并写进结果。
- **单种子声明**：两臂各一个 ckpt（seed 101），只代表 1 个种子；不得声称稳健；<1 dB 的臂间差不得单独判罪（本项目臂间种子 sd 0.5–1.0 dB）。

---

## 7. 已知混淆与措辞边界（提前声明）

1. **数据配方不同** ⇒ 结论只能写「实际可达到的上限」，**不能**写「LightRR 架构在该域失效」或「轻量网络不可行」。
2. LightRR 迭代预算 100k it（16.7×RDNet 的 6k it），不是欠训；但 batch = 1。
3. 1019 是 development panel（反复开发），任何数字不得称「冻结 / 未见」。
4. **跨论文数字不可比**：LightRR 自报 Real20 25.12 与我们的 1019 面板无关，禁止并列对比。
5. 若微调后超过 23.0 ⇒ 说明轻量也能学会，如实报并改叙事（这正是本实验要防的自我欺骗）。
6. 其官方验证集（Real20）是普通玻璃，**只能看有没有训崩**，不能外推本域。

---

## 8. 执行清单与产物

- [ ] 等训练完成（ETA 11-06 06:30–07:30），只读确认 final ckpt；
- [ ] 推理 3 个 ckpt：`ft_lr1e4` / `ft_lr5e5` / 官方冻结重跑（可借 codex-dev 的 task-1 管线，接口须满足 §4）；
- [ ] 管线验证门（§4）→ 出表 → 配对统计 → 指标扩展；
- [ ] 回填本档 §9（不另开档）；结论交 Lead，台账交 task-7。

产物：
- `runs/pred/real_pair_1019_lightrr_ft_lr1e4|ft_lr5e5|lightrr_official_rerun/`
- `eval/table_final_1019_lightrrft1106.{csv,md,json}`、`eval/real_pair_1019_lightrr_ft_*_summ.json`
- 本档（服务器 + 本地镜像）

---

## 9. 结果与判决（待回填，2026-11-06）

**回填于 2026-11-06 04:56（判据 §5 先于结果写死，未改动）。**

**主表（1019 development panel，n=15 对，uint8 口径；权威表 `eval/table_final_1019_lightrrft1106.{md,csv,json}`）**

| 臂 | PSNR | SSIM | HF-PSNR | 梯度余弦 | LapVar | 洞区 PSNR | 非洞区 |
|---|---|---|---|---|---|---|---|
| lightrrft_lr1e4（`_041_00102500.pt`） | **20.221** | 0.7135 | 33.29 | 0.573 | 47.8 | 20.32 | 20.23 |
| lightrrft_lr5e5（`_041_00102500.pt`） | **20.198** | 0.7133 | 33.30 | 0.572 | 47.5 | 20.18 | 20.21 |
| lightrr_official_rerun（官方冻结 ckpt，同管线） | **19.175** | 0.6804 | 32.34 | 0.594 | 68.8 | 20.89 | 19.12 |

**best-of-2 = 20.221 dB（lr1e4）⇒ 档位 P1（动机成立，≤22.2）。**

**配对 Δ（逐场景，n=15；t_14=2.145）**
- lr1e4 − quiet_s101(full seed101) = **−2.489**，95% CI [−3.100, −1.878]，胜出 0/15，配对 sd 1.103，MDE 0.798
- lr1e4 − oldpool_s101(base seed101) = **−1.770**，95% CI [−2.198, −1.341]，胜出 0/15，MDE 0.560
- lr5e5 − quiet_s101 = −2.512，CI [−3.091, −1.933]；lr5e5 − oldpool_s101 = −1.793，CI [−2.211, −1.374]（均 0/15）
- 官方重跑 − quiet_s101 = −3.534，CI [−4.156, −2.912]，0/15

**管线验证门**：官方冻结 ckpt 同管线 19.175 vs 历史 verify summ 19.1295 ⇒ 差 +0.046（验收门 ±0.05～0.1，**通过**）；identity 18.44 与 r11annot 表 18.437 一致（差 0.003）。

**训练语义披露**：实际训练 41 epoch / 102,500 it（预登记写 40 epoch；`train_ft.py:153` 的 `while engine.epoch <= nEpochs` 边界多 1 轮），最终 ckpt 为 `_041_00102500.pt`。两臂对称，不影响判据与档位。

**ICME 动机措辞建议（可直接用）**：「Even after 100k+ iterations of in-domain fine-tuning under its official recipe, the 9.5M-parameter LightRR reaches only 20.22 dB on the development panel, still 2.27 dB below our RDNet baseline (22.49 dB, 3-seed) and 2.77 dB below the full model — i.e. the gap is not a capacity/under-training artifact.」
（注意：1019 是 development panel；跨论文 Real20 数字不可比。）

**判定：P1 —— LightRR 动机实验成立。** 产物：`/tmp/ri_dev/LIGHTRR_JUDGE_1106.md`、`eval/table_final_1019_lightrrft1106.*`、`runs/pred/real_pair_1019_lightrrft_lr1e4|lr5e5|lightrr_official_rerun/`。
