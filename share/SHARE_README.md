# TintedGlass_Enhance — 合成数据 + RDNet 域微调（分享包）

本分支（`tintedglass-synth`）分享的是 **贴膜玻璃合成数据管线 + RDNet 合成域微调** 的工作，
与 main 分支的 `vehicle_ssd` 恢复代码互不干扰。

- **数据**（~19G）已传到 Hugging Face：`zmy1234567890/tintedglass-rdt-synth`（公开 dataset repo）
- **微调权重**（4 个 ckpt）已传到：`zmy1234567890/tintedglass-rdnet-synth-ft`（公开 model repo）
- **完整研究仓库**（含 docs/、fig/、逐实验记录）在内网机器上；本分支只放这条线的代码。

---

## 1. 这条线在做什么

贴膜玻璃（车窗膜）场景下，透射成像被窗上倒影污染。我们把问题拆成
**反射去除（RDNet）→ 低光增强（ExpoMamba）** 两级，本分支只涉及第一级：

1. **合成数据管线**：把"真实反射残差层 R"按物理模型 `I = α(x)·T + α(x)^p·R + n` 合成到
   透射图上，其中 `α(x)` 是空间变化的膜透过率场（覆盖突变 + 弧状缓变）。
   p=0（v2.1 口径）表示反射强度不随膜的厚度衰减——这是贴膜场景与通用 SIRR 数据集的关键差别。
2. **RDNet 域微调**：用上述合成三元组（I/T/R）微调官方 RDNet，冻结 FocalNet 主干、
   训练其余参数；核心超参是低 α 样本的**逆衰减加权** `w(α) = clamp((0.3/α)², 1, 10)`。
3. **评测结论（重要，先看）**：合成域上微调有效，但**迁移到实拍反而显著变差**
   （三套实拍配对集 −0.44 ~ −3.87 dB）。详见下文 §6——避免重复踩坑。

## 2. 下载数据

```bash
# 数据集（~19G）
HF_ENDPOINT=https://hf-mirror.com \
  hf download zmy1234567890/tintedglass-rdt-synth --repo-type dataset --local-dir data_hf

# 微调权重（~6G）
HF_ENDPOINT=https://hf-mirror.com \
  hf download zmy1234567890/tintedglass-rdnet-synth-ft --repo-type model --local-dir ckpt_ft
```

（`HF_ENDPOINT=https://hf-mirror.com` 只在 huggingface.co 直连不通的网络下需要。）

## 3. 目录怎么摆

把下载物按下面的相对路径摆进本仓库根目录（脚本里路径就是这么写的）：

```text
TintedGlass_Enhance/
├── data/
│   ├── train_sets/train_triplets_rdt_0915/     ← data_hf/train/rdt_0915/
│   ├── test_sets/synth_rrw_v21_0913/           ← data_hf/test/synth_v21_0913/
│   ├── test_sets/synth_rrw_v21_indomain/       ← data_hf/test/synth_v21_indomain/
│   ├── test_sets/synth_rrw_v22_scan/           ← data_hf/test/synth_v22_scan/
│   ├── test_sets/synth_rrw_v23_lowscan/        ← data_hf/test/synth_v23_lowscan/
│   ├── test_sets/real_pair_0917/               ← data_hf/test/real_pair_0917/
│   ├── test_sets/real_pair_apr_0917/           ← data_hf/test/real_pair_apr_0917/
│   ├── test_sets/real_pair_may_0917/           ← data_hf/test/real_pair_may_0917/
│   ├── pools/r_train_0915/                     ← data_hf/pools/（131 张 png，平铺）
│   └── pools/r_train_0915_meta.csv             ← data_hf/pools/r_train_0915_meta.csv
├── runs/ckpt/rdt_{plain,freeze}*/              ← ckpt_ft/rdt_{plain,freeze}[_s101]/
├── pretrained/                                 （RDNet 官方权重，见 §4）
└── RDNet/                                      （模型代码，见 §4）
```

## 4. 依赖

- Python / torch：本项目用 `torch 2.12.0+cu126`；RDNet 代码目录 `RDNet/` 已在本仓库内
- **RDNet 官方预训练权重**（不在 HF 分享包里，需自行下载后放到 `pretrained/`）：
  来自官方仓库 [lime-j/RDNet](https://github.com/lime-j/RDNet)（CVPR 2025）：
  - `rdnet.pth` → https://checkpoints.mingjia.li/rdnet.pth
  - `cls_model.pth` → https://checkpoints.mingjia.li/cls_model.pth
  - `focal.pth`（FocalNet 骨干）→ https://checkpoints.mingjia.li/focal.pth
- 数据生成脚本：`numpy`、`opencv-python`、`Pillow`（`scipy`/`matplotlib` 仅画图用）

## 5. 关键脚本（按执行顺序）

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 1. 提 R 池 | `scripts/build_rrw_pool.py` | 从 RRWDataset（反射视频）提时域中值反射层；入参 `--src` 指原始数据集 |
| 2. 扩池 | `scripts/build_rrw_pool_expand.py` + `build_r_train_expand_0918.py` | 帧窗切片 + 泄漏安全网清洁（cluster 去重）→ `r_train_expand_0918` |
| 3. 泄漏核查 | `scripts/check_t_src_leak.py` | **新数据必须过**：内容级（md5+感知哈希）核查 T 源与测试集是否重叠 |
| 4. 建训练三元组 | `scripts/build_train_triplets.py` | 物理合成 `I=αT+R+n`；产物 I/T/R/alpha/m/g + meta.json |
| 5. 微调 | `scripts/finetune_rdnet_tint.py`（主臂） / `finetune_rdnet_peft.py`（Mona 适配器） | 30k iter，冻 backbone，逆衰减加权由 `--lam-wlow` 控制 |
| 6. 评测 | `scripts/eval_rdt_tint.py`（合成） / `eval_real3_ft.py`（实拍三集） | PSNR/SSIM/LPIPS，含增益匹配臂 `_gm` |

完整命令链见仓库 `Makefile`（`make help`）。

> **路径提示**：各脚本的 `--src` / `--out` / `--triplets` 默认值写的是原作者的机器路径，
> 在你机器上**显式传参覆盖**即可；脚本内部的相对路径（`PROJ/RDNet`、`pretrained/`）
> 按 §3 摆放后自动成立。

## 6. 结果与结论（先读再跑）

**合成域（v2.1，α=0.3/0.6/1.0 三档）**：官方 RDNet 在贴膜合成集上明显退化；
微调后（冻主干 30k iter）结构指标显著回升。

**实拍域（三套配对集，gain-matched PSNR 中位）**：

| 数据集 | identity | 官方零样本_gm | ft_plain_gm | ft_freeze_gm |
|---|---|---|---|---|
| 6 月（弱反射） | 19.40 | **22.73** | 20.07 | 20.31 |
| 4 月（强反射） | 17.93 | **17.84** | 16.68 | 17.02 |
| 5 月（局部极强） | 20.26 | **24.07** | 19.89 | 20.17 |

**9/9 格为负**（Wilcoxon p<0.05）→ 合成微调**不迁移**到实拍。
候选解释：合成 T 无噪声/JPEG、R 来自公开反射视频、α 场为人工合成——与实拍的真实
噪声/压缩/自反射形态存在系统性域差。**这是一个负结果，请勿据此重复训练。**

## 7. 微调权重（HF model repo）

| ckpt | 说明 |
|---|---|
| `rdt_plain` | 冻主干，`--lam-wlow 0`（无逆衰减加权），seed 20260915 |
| `rdt_freeze` | 冻主干，`--lam-wlow 1.0`（逆衰减加权），seed 20260915 |
| `rdt_plain_s101` / `rdt_freeze_s101` | 同上，仅随机种子 101（种子方差对照） |

载入方式：`torch.load(ckpt)["icnn"]` 即 `FullNet_NLP` 的 state_dict（与官方 `rdnet.pth` 同构，
可直接 `load_state_dict`）。每个 ckpt 结构：`{"icnn":..., "opt":..., "scaler":..., "args":..., "iter":...}`。

## 8. 备注

- 实拍配对集（`test/real_pair_*`）**仅限评测**：其 T 帧与训练数据字节级同源，
  一旦进入训练/调参，该集所有数字作废（每个目录内有 `DO_NOT_USE_IN_TRAINING.md`）。
- `synth_rrw_v21_indomain` 与训练集存在同源簇（r≈0.999），结论已被标记为不可引用，
  分享包保留它只为完整性。
- 合成数据本身无隐私问题；R 层来自公开反射数据集（RRWDataset），仅供研究。
