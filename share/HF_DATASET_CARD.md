---
license: other
license_name: research-use-only
license_link: https://github.com/lime-j/RDNet
tags:
  - reflection-removal
  - single-image-reflection-removal
  - synthetic
  - image-restoration
task_categories:
  - image-to-image
---

# TintedGlass RDT — 贴膜玻璃合成反射数据 + 实拍配对测试集

面向**贴膜玻璃（车窗/建筑膜）**场景的单图反射去除合成数据。核心构造：
`I = α(x)·T + α(x)^p·R + n`，其中 `α(x) ∈ (0,1]` 是空间变化的膜透过率场
（覆盖突变 + 弧状缓变），`p=0` —— **反射强度不随膜衰减**，这是贴膜场景与通用
SIRR 合成集（一般 `p=1`）的关键差别。

## 仓库结构

```text
train/rdt_0915/          合成训练三元组（4000 场景 × 5 档 = 20000 样本, 448²）
  T/{sid}.png            透射层（各档共享）
  R/{sid}.png            反射层（各档共享）
  I/{sid}_l{030,045,060,075,100}.png     合成输入（5 档）
  alpha/{sid}_l{...}.npy 透过率场（float16, 448×448）
  m/{sid}.npy            膜覆盖二值场（float16; alpha = clip(level·g·m + (1-m), 0.01, 1))
  g/{sid}.npy            弧状缓变增益场（float16）
  meta.json              全部生成参数（levels / spatial / noise / r_pool / physics）
test/synth_v21_0913/     12 场景 × 3 档（α=0.3/0.6/1.0）= 36 样本
test/synth_v22_scan/     12 场景 × 8 档（α=0.05…1.0）= 96 样本
test/synth_v23_lowscan/  48 场景 × 5 档（α=0.05…1.0）= 240 样本
test/synth_v21_indomain/ 12 场景 × 3 档，与训练集同源（**结论不可引用**，仅供完整性）
test/real_pair_0917/     13 场景/13 输入，实拍配对（6 月，弱反射）——**仅评测**
test/real_pair_apr_0917/ 6 场景/16 输入，实拍配对（4 月，强反射）——**仅评测**
test/real_pair_may_0917/ 5 场景/14 输入，实拍配对（5 月，局部极强）——**仅评测**
pools/                   131 张反射层源池（合成 R 层的原材料, 38 个 RRW 场景）
  r_train_0915_meta.csv  源池元数据
```

合成测试集目录：`gt/t{XX}.png`（透射 GT）、`r/t{XX}.png`（反射层）、
`levels/alpha_{lv}/t{XX}.png`（各档输入）、`alpha/`（透过率场）、`meta.json`。

实拍配对集目录（两套命名）：

```text
real_pair_0917/{scene}/        I.png（有反射输入） T.png（无反射参考） N.png（近无反射帧） meta.json
real_pair_{apr,may}_0917/{scene}/
    GT.png                    场景无反射参考帧（与 vehicle_ssd 训练数据同源）
    I_{frame:06d}.png         该场景的多个反射输入帧（帧号见文件名）
    meta.json                 gt_frame（GT 帧号）、inputs{帧号: shift/peak_ratio/D_med/...}
```

**注意**：无反射参考帧（`T.png` / `GT.png`）与 `vehicle_ssd` 训练数据**字节级同源**，
因此这三套集**只能用于评测**，进入任何训练/调参流程即污染（各目录内有
`DO_NOT_USE_IN_TRAINING.md`）。

## 生成参数（train/rdt_0915）

| 项 | 值 |
|---|---|
| 场景数 × 档数 | 4000 × 5 = 20000 |
| 尺寸 | 448×448 |
| 档位 | α ∈ {1.0, 0.75, 0.6, 0.45, 0.3} |
| 空间场 | spatial=0.3（覆盖突变 + 弧状缓变） |
| 噪声 | σ=1.5（0–255 域高斯） |
| R 池 | `pools/r_train_0915`（131 张，RRWDataset 时域中值提层后重标定） |
| T 源权重 | SIR²:0.175 / nature:0.125 / scenes:0.2 / RRW-GT:0.5 |
| 物理 | `I = α(x)·T + R_eff + n`（p=0）；5 档共享 m/g/R_eff/噪声 |
| α 场定义 | `g = 1 + s·f_arc`；`α = clip(level·g·m + (1−m), 0.01, 1.0)` |

**泄漏核查**：T 源已经过内容级核查（md5 + 感知哈希/聚类），与 v21/v22 测试集无重叠；
v23 的 R 源曾误入旧训练池（v23 结果已被标记不可引用）。

## 用途与已知结论

这条数据用于 **RDNet（CVPR 2025）的贴膜域微调**：冻 FocalNet 主干训练其余参数，
配套权重见 `zmy1234567890/tintedglass-rdnet-synth-ft`。

**重要负结果**：合成域微调**不能迁移**到实拍（三套实拍配对集 gain-matched PSNR
全部变差 −0.44 ~ −3.87 dB，Wilcoxon p<0.05）。候选原因是合成 T 无噪声/JPEG、
R 来自公开反射视频、α 场人工合成，与实拍域存在系统性差异。用本数据复现微调时
请把它当作"域差研究"的素材，而不是可直接部署的配方。

## 来源与许可

- 合成层（T 源、R 池）来自公开学术数据集（SIR²、RRWDataset 等），仅供研究；
  再分发前请自行核对各上游数据集许可。
- 实拍配对集为拍摄者自有素材（手机拍摄），仅授权研究用途；含车牌/建筑等场景信息，
  请勿用于人脸识别、车牌识别等二次任务。
