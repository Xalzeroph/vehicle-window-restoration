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
- 模型权重在 `pretrained/` 下（符号链接到 `RDNet-main/`；ExpoMamba 权重在 `pretrained/expomamba/`）

## 服务器环境与守则（2026-09-14 迁入共享机）

本项目现运行于实验室共享服务器 `init-SYS-4029GP-TRT`（Ubuntu 24.04，80 核 / 251G 内存，8× RTX 3080 **10GB**，`/home` 下约 14 个用户）。旧机（3080Ti / `/DataDisk` 坏盘 / IERR 崩溃）事实已作废，见 `2026-09-14_项目搬迁方案.md`。

### 环境事实

- 项目路径 `/DataDisk/zmy_code/TintedGlass_Enhance`；`/home/zmy/mycode -> /DataDisk/zmy_code` 软链，两种写法等价
- conda：`/DataDisk/miniconda/envs/xreflection`（torch 2.12.0+cu126，8 卡可见）
- 磁盘：`/DataDisk` = 系统盘 `/dev/sda2`（879G，余 ~600G，项目与 miniconda 都在其上）；`/home` = 独立 4T 盘（余 1.5T，全组共用）；`/storage` 5T **已 100% 满，禁止写入**
- 2026-09-14 搬迁 smoke：`python scripts/pipeline.py` 端到端通过

### 共享机守则

1. **GPU 先到先得**：跑任务前 `nvidia-smi` 查空闲卡，用 `CUDA_VISIBLE_DEVICES=N` 显式钉卡；不占满全部 8 卡，任务结束确认显存释放。显存仅 10GB（旧机 12GB），大 batch 训练先小步验证 OOM。
2. **CPU/内存是公共资源**：其他用户常驻任务（如 MATLAB）会吃满 CPU；重任务设 `OMP_NUM_THREADS` / `num_workers` 上限，别抢满 80 核。
3. **产物落盘**：实验产物/数据集一律放**项目目录内**（延续旧机习惯）；不写 `/storage`；`/DataDisk` 是系统盘，累计产物超 ~200G 时再议转移。
4. **不动他人目录**：`/home/{fangyuan,kyshen,hjl,...}` 属其他用户；sudo 需密码，系统级操作先问用户。
5. 长任务用 `tmux`/`nohup` 跑防 SSH 断开；GPU0 有桌面会话常驻少量显存占用，优先用 1–7 号卡。
