# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

RDNet (Reversible Decoupling Network) 是一个用于单张图像反射去除（Single Image Reflection Removal）的深度学习模型，发表于 CVPR 2025。核心思想是使用可逆编码器在 forward pass 中灵活解耦传输层和反射层特征，并通过 transmission-rate-aware prompt generator 动态校准特征。

**注意：此代码库已被官方废弃。README 中声称 XReflection 替代，但实际上 XReflection 也是半成品（大量 TODO 未完成），不可用。因此本仓库仍是实际可用的实现。**

claude全部中文答复我


## 环境与依赖

使用 conda 环境 `xreflection`：

```bash
conda activate xreflection
```

依赖包安装：

```bash
pip install torch>=2.0 torchvision
pip install einops ema-pytorch fsspec fvcore huggingface-hub matplotlib numpy opencv-python omegaconf pytorch-msssim scikit-image scikit-learn scipy tensorboard tensorboardx wandb timm dominate visdom
```

## 常用命令

### 推理/测试

```bash
# 在标准 benchmark 上测试（需先修改 test_sirs.py 中的数据集路径）
python3 test_sirs.py --icnn_path <checkpoint路径> --resume

# 在自定义图片上运行推理
python inference_simple.py /path/to/images [output_dir] [model_path] [gpu_id]
```

### 训练

```bash
python3 train.py
```

## 架构概览

### 模型核心 (`models/arch/RDnet_.py`)

- **`FullNet_NLP`** — 主网络。使用 FocalNet（`focalnet_L_384_22k_fl4`）作为 backbone 提取多尺度特征，通过多个 `SubNet` 级联处理，最后用多个 `Decoder` 生成输出。支持 transmission-rate-aware prompt 机制。
- **`SubNet`** — 可逆子网络。包含 4 个 `Level`（对应 4 个尺度），支持普通前向和可逆前向两种模式。可逆模式通过 `ReverseFunction` 实现，在反向传播时重建中间特征以节省显存。
- **`Level`** — 单个尺度的处理单元，由 `Fusion`（特征融合/上下采样） + 多个 `NAFBlock`/`ConvNextBlock` 组成。
- **`Fusion`** — 处理不同尺度间的特征融合，第一列 subnet 只做下采样，后续 subnet 做上采样+下采样的融合。

### 可逆机制 (`models/arch/reverse_function.py`)

`ReverseFunction` 是自定义的 `torch.autograd.Function`，实现可逆传播。前向时保存 RNG state 和中间激活值，反向时通过重新计算（recompute）中间特征来还原，从而用时间换空间。

### 模型封装 (`models/cls_model_eval_nocls_reg.py`)

- **`ClsModel`** (继承 `YTMTNetBase` → `BaseModel`) — 训练/推理的主控制器。管理 `net_i`（主网络 `FullNet_NLP`）、`net_c`（分类器 `PretrainedConvNext`），负责 loss 计算、optimizer 管理和 checkpoint 保存/加载。
- 训练时使用 DINO-ViT perceptual loss (`DINOLoss`)，VGG loss，像素级 loss，梯度 exclusion loss 和 reconstruction loss。
- 模型输出为 6 通道张量（3 通道 reflection + 3 通道 transmission）。

### 数据流 (`data/dataset_sir.py`)

- **`DSRDataset`** — 合成数据集。随机配对透射图(T)和反射图(R)，通过 `ReflectionSynthesis` 合成混合图像 I = T + R - T*R（或 I = T + R）。训练时使用 `FusionDataset` 混合合成数据、真实数据和自然图像数据。
- **`DSRTestDataset`** — 测试数据集。期望目录结构：`{datadir}/transmission_layer/` 和 `{datadir}/blended/`。
- **`RealDataset`** — 仅含输入图像的推理数据集（无 ground truth），用于自定义图片推理。

### 训练流程 (`train.py` + `engine.py`)

1. `train.py` 构建数据集和 dataloader，创建 `Engine` 实例
2. `Engine` 管理训练/评估循环：
   - 每个 epoch 训练后在多个 benchmark（real20, SIR2 SolidObject, Postcard, WildScene）上评估
   - 根据加权 PSNR 总和保存最佳模型
   - 每 `save_epoch_freq` 个 epoch 保存一次
3. 学习率为 1e-4，使用 Adam 优化器，前 20 个 epoch 保持恒定学习率

### 配置系统 (`options/`)

- `base_option.py` — 基础参数解析（name, gpu_ids, checkpoints_dir, resume 等）
- `net_options/base_options.py` — 网络相关参数（inet, icnn_path, init_type 等）
- `net_options/train_options.py` — 训练参数（lr, batchSize, loss 权重, num_subnet, loss_col, drop_path 等）

### 预训练模型

| 模型 | 路径 | 用途 |
|------|------|------|
| rdnet.pth | 主 checkpoint | 推理/测试 |
| cls_model.pth | `pretrained/` 目录 | 分类器（transmission-rate predictor） |
| focal.pth | `pretrained/` 目录 | FocalNet backbone 权重 |

## 关键参数

- `--num_subnet`：子网络数量（默认 4），控制网络深度
- `--loss_col`：计算 loss 的最后几个 subnet 列数（默认 4）
- `--if_align`：是否将图像尺寸对齐到 32 的倍数
- `--save_memory`：在 `FullNet_NLP` 中控制是否使用可逆模式节省显存
- `--dataset`：合成数据集在混合训练中的比例（默认 0.5）

## 注意事项

- 模型要求输入尺寸能被 32 整除（`if_align` 选项会自动处理）
- 训练路径在 `train.py` 中硬编码为 `/opt/datasets/sirs`，需要根据实际数据位置修改
- `test_sirs.py` 中的测试路径是本地路径，使用前需要修改
- `models/__init__.py` 中的 `make_model` 函数忽略 name 参数，始终返回 `ClsModel`
- 原版在单卡 3090（24GB）上运行。12GB 显卡（如 3080 Ti）会爆显存，已对 `forward_eval()` 和 `test()` 添加 `torch.cuda.amp.autocast()` 实现 fp16 推理，显存占用约减半
