# REPRO.md —— 从零复现本论文的实验

> 目标读者：拿到本包、有 GPU 的合作者。
> 全部命令在 `better2`（8× RTX 3080 10 GB）上验证过；单卡 10 GB 足够，全量微调需要 ≥16 GB 的另一套配方（本文不用）。

---

## 1. 环境

```bash
conda create -n xreflection python=3.10 -y
conda activate xreflection
# 关键版本（实测可用）
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24 opencv-python==4.8 scipy einops timm==0.9.8
```

```bash
git clone <repo> && cd vehicle-window-restoration   # 本包所在的仓
export TINTED_ROOT=$PWD
export TINTED_PYTHON=$(which python)
```

**目录约定**（训练脚本按此路径找数据）：

```
data/
├── test_sets/real_pair_1019/     # 冻结面板：15 个场景，每场景 I_*.png + GT.png
├── train_sets/real_pairs_1020/   # 我们的真实配对（apr 126 / may 130 / jun 44）
│   ├── I/  T/  R/  omega/  meta.json
├── train_sets/real_pairs_1026/   # apr 重裁 82 对
├── external/VOC2012/VOC2012_s320/            # 合成 T 源（公开）
├── pools/r_final_1029/                       # 合成 R 源（SIR² 系，463 张）
└── silhouettes/lib/                          # S2 形状先验（Penn-Fudan 剪影 420 张）
```

**R 池与剪影库可由脚本重建**（都在 `code/data_build/`）：

```bash
python code/data_build/build_r_pool_final.py --out data/pools/r_final_1029
python code/data_build/build_silhouette_lib.py    # 需要先下 Penn-Fudan
```

---

## 2. 评测面板与必带基线（**任何对比都先跑这一条**）

```bash
# 单臂推理（结果落 runs/pred/real_pair_1019_<name>/）
CUDA_VISIBLE_DEVICES=0 python code/infer/eval_arm_1019.py \
    --ckpt runs/<arm>/ckpt_final.pth --name <tag> --device cuda:0

# 权威评测表（PSNR/SSIM/HF-PSNR/梯度余弦/LapVar）
python code/eval/table_final_1019.py --pred-map <map.json> --tag <tag>
```

`<map.json>` 是 `{"arm 名": "预测目录名"}` 的字典。**必须包含**：

```json
{ "identity": "real_pair_1019_identity", "rdnet_zs": "real_pair_1019_rdnet_zs" }
```

`identity` 给出分数地板，`rdnet_zs` 给出公开模型现状。**缺这两个基线的数字不算结论**（本项目纪律）。

---

## 3. 训练（论文里每个数字对应的命令）

统一配方：6000 iters / bs 6 / lr 1e-4 / Mona adapter / 裁剪课程 224→256→320 / λ_om=3.0。

### 3.1 主结果（5+6 月，22.70 dB）

```bash
CUDA_VISIBLE_DEVICES=0 python code/train/finetune_rdnet_s2.py \
  --arm main --seed 101 --iters 6000 --batch 6 --lr 1e-4 --mona both \
  --lam-r 1.5 --lam-recons 0.2 --lam-excl 0.1 --lam-hinge 1.0 --lam-dists 0.2 \
  --lam-grad 5.0 --lam-lap 2.0 --s-jitter 0.0 --seg-iters 2000 --lam-om 3.0 \
  --omega-dir data/train_sets/real_pairs_1020 --omega-domains may,jun \
  --out runs/main/mayjun_s101
```

三种子：把 `--seed 101` 换成 `202` / `303`，输出目录同名后缀。
结果：**23.07 / 22.87 / 22.20 ⇒ 22.70 ± 0.46**。

### 3.2 对照（4+6 月，20.97 dB）

同上，只改 `--omega-domains apr,jun`。三种子：**20.78 / 21.35 / 20.76 ⇒ 20.97 ± 0.34**。

### 3.3 关掉区域项（论文里最大的单效应：−3.2 / −4.4 dB）

```bash
# 在同一条命令上只改一个参数：--lam-om 0
... --lam-om 0 ... --omega-domains may,jun --out runs/main/mayjun_nomask
```

实测：may+jun **18.33**、apr+jun **17.77**，**均低于 identity 18.44**。

### 3.4 区域"画在哪"的四种对照（82 对 apr 池，`real_pairs_1026`）

| 臂 | 额外参数 | PSNR（3 种子均值） |
|---|---|---|
| 标注掩码 | `--omega-mode real` | 19.41 ± 0.54 |
| 随机盘·每步重画 | `--omega-mode random` | 19.90 ± 0.16 |
| 随机盘·每对固定 | `--omega-mode randomfixed` | **20.25** ± 0.29 |
| 整图（不分区域） | `--omega-mode full`（λ_om 需按面积重标定，用 0.6） | 20.05（单种子） |
| 无区域项 | `--lam-om 0` | 17.76 ± 0.16 |

判读：**固定 − 重画 = +0.35 dB，未达预登记的 0.5 dB 阈值 ⇒ 无显著差异**；
标注掩码既不最好、种子方差又最大 ⇒ 论文因此把"区域身份"降级为不显著。

### 3.5 合成遮挡三臂（2026-10-29 新增）

```bash
# A 合成样本带精确 Ω（软覆盖率 c 作区域权重）
... --synth-occ exact --synth-shape mix --omega-mode real ...
# B 同面积固定随机盘（图像相同，只换监督区域）
... --synth-occ exact --omega-mode randomfixed ...
# C 合成样本不参与 Ω 监督（图像仍注入遮挡）
... --synth-occ none ...
```

合成遮挡的物理口径（与实测对齐，见 `code/train/synth_occlusion_v2.py` 顶部常量）：

| 量 | 目标 |
|---|---|
| Ω 外 `I/T` | 1.045 |
| Ω 内 `I/T` | 0.87 ± 0.10（截断 [0.55, 1.05]）⇒ 输入比目标暗 ~13% |
| 遮挡物自身暗反射 | 占 `T` 的 1.5%–3.0% |
| 形状 | S1 随机闭曲线 / S2 真实行人剪影（**不旋转**、放宽 1.2–1.7×、可只取上半身） |

---

## 4. 复现后应得到的表

跑完 §3.1–3.4 后，用 `table_final_1019.py` 出的表应与下列一致（容差 ±0.05 dB，种子对齐）：

| 方法 | PSNR |
|---|---|
| identity | 18.44 |
| RRW | 20.50 |
| apr+jun（λ=3，3 种子） | 20.97 |
| apr+jun（λ=0） | 17.77 |
| may+jun（λ=3，3 种子） | 22.70 |
| may+jun（λ=0） | 18.33 |
| 标注掩码 / 重画盘 / 固定盘 / 无项（82 对池） | 19.41 / 19.90 / 20.25 / 17.76 |

权威原始表：`results/table_final_1019_final3_1029.md`（四区组合并）、
`results/table_final_1019_all_1029.md`（含 affine 臂）。

---

## 5. 诊断脚本（论文 §4.6 与限制章节用）

```bash
python code/eval/measure_real_darkness_gap.py --pool real_pairs_1026   # Ω 内外落差分布
python code/eval/check_illumination_vs_reflection.py --pool real_pairs_1026  # 光照 vs 反射的分解
python code/eval/region_gain_omega.py                                # identity 参照的区域增益
python code/eval/probe_refl_color_scalar.py                          # 反射层色偏
```

**注意**：`region_gain_omega.py` 会打印"Ω 内增益"，但**该 Ω 是自动判据生成的、不可靠**，
论文里这一类数字已全部删除。脚本保留是为了公开"我们试过并且知道它不行"。

---

## 6. 常见失败

| 症状 | 原因 | 处理 |
|---|---|---|
| `FileNotFoundError: data/train_sets/...` | 没设 `TINTED_ROOT` | `export TINTED_ROOT=$PWD` |
| `KeyError: 'hole_psnr'` | 用了旧版列名 | 现表列名是 `psnr_hole` / `psnr_nonhole` |
| 训练 loss 正常但 PSNR ≈ identity | `--lam-om 0` 或没带 `--mona` | 核对命令行 |
| 评测 PSNR 比表高 3–5 dB | **配对错帧**（同场景多帧，取错 I） | 必须按同一输入 stem 配 `I_*` 与预测 |
| 合成样本训练后无 Ω 监督 | 忘加 `--synth-occ exact` | 该参数默认 `none`（保持旧行为） |

---

## 7. 复现所需显存与时间

| 项 | 值 |
|---|---|
| 单臂训练 | **约 127 min**（6000 it，bs6，单卡 10 GB，占用 ~7 GB） |
| 单臂推理（15 场景） | 约 1.5 min |
| 全套（主结果 3 种子 + 4 组对照 + λ=0 两臂） | 约 20 卡·小时 |
| 论文全部 86 臂 | 约 180 卡·小时（本机 8 卡，约 1 周） |
