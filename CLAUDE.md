# CLAUDE.md

## 项目概述

TintedGlass_Enhance 针对**深色贴膜玻璃场景**的透射成像增强。核心思路：**RDNet（反射去除）→ ExpoMamba（低光增强）** 两阶段联合推理。

- RDNet [CVPR 2025] — 可逆解耦网络，分离透射层 T 和反射层 R
- ExpoMamba [2024] — Mamba 架构低光增强网络，提升暗光图像亮度

## 环境

使用 conda 环境 `xreflection`：

```bash
conda activate xreflection
```

## 目录结构

```
TintedGlass_Enhance/
├── docs/                   # 研究文档（实验记录/评审/调研，md）与早期资料（txt）
├── fig/                    # 实验记录图（结果曲线/判定与核验 sheet/拼图）
├── RDNet/                  # RDNet 代码（反射去除）
├── ExpoMamba/              # ExpoMamba 代码（低光增强）
├── pretrained/             # 预训练模型权重（符号链接）
├── scripts/                # 推理/工具脚本
│   └── pipeline.py         # 联合推理：RDNet → ExpoMamba
├── eval_*.csv              # 评测指标输出（脚本写入项目根）
├── CLAUDE.md               # 本项目说明
└── command.txt             # 常用命令
```

注意：`r_rrw_strong/`、`synth_rrw_*/`、`probe_rscale_0913/` 等实验数据产物目录不入 git（见 .gitignore），可由 `scripts/` 下生成器重建。

## 常用命令

### 联合推理（去反射 + 低光增强）

```bash
python3 scripts/pipeline.py /path/to/images [output_dir]
```

### 单独 RDNet 反射去除

```bash
cd RDNet && python3 inference_simple.py /path/to/images
```

### 单独 ExpoMamba 低光增强

```bash
# 需先配置 ExpoMamba 环境和权重路径
```

### 训练 RDNet

```bash
cd RDNet && python3 train.py
```

## 关键参数

- `--num_subnet`：子网络数量（默认 4）
- `--if_align`：将图像对齐到 32 的倍数
- `--save_memory`：使用可逆模式节省显存

## 注意事项

- RDNet 要求输入尺寸能被 32 整除
- 深色贴膜场景（α << 1）：先 RDNet 去反射，再 ExpoMamba 提亮
- 普通反射场景：RDNet 单独即可
- 模型权重在 `pretrained/` 下（符号链接到原始位置）
- **磁盘（2026-09-12）**：物理盘 `/DataDisk`（`/dev/sda2`）已故障、整盘只读（读正常、写报 EROFS）。本项目所在 `/DataDisk/zmy_code` 与 `/DataDisk/miniconda` 已迁至 NVMe（LVM `ubuntu-vg/workspace`），经 bind mount 保持旧路径可用，故以上所有路径照常使用；**但不要直接读写 `/DataDisk` 上其他（非 zmy_code/miniconda）路径**。详见 `/home/courseliu/2026-09-12_DataDisk坏盘迁移记录.md`
