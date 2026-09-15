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
├── scripts/                # 推理/实验脚本（probes_0915/ 为一次性探针）
│   └── pipeline.py         # 联合推理：RDNet → ExpoMamba
├── .claude/skills/         # research-loop 工作流协议（随库入库）
├── data/                   # 实验数据产物（不入 git，可由 scripts/ 重建）
│   ├── pools/              # T/R 源池、matched_pool_*、probe_rscale_*
│   ├── test_sets/          # synth_rrw_v21_0913 / _indomain / v2 / demo
│   ├── train_sets/         # train_triplets_*、train_pairs_*
│   └── rdnet_runs/         # RDNet 训练日志/events（原 checkpoints/、experiment/）
├── runs/                   # 训练产物（不入 git）
│   ├── ckpt/               # 模型权重（原 pilot_ckpt/）
│   └── pred/               # 逐图预测 fac_pred_* / pilot_pred* / synth_rrw_*_pred
├── eval/                   # 评测输出 CSV/JSON（入 git）
├── Makefile                # 实验执行链（make help）
├── CLAUDE.md               # 本项目说明
└── command.txt             # 常用命令
```

（2026-09-15 目录整理：数据/权重的旧路径 `synth_rrw_*`、`train_triplets_*`、`pilot_ckpt/`、`fac_pred_*`、
根下 `eval_*.csv` 分别迁入 `data/`、`runs/`、`eval/`；历史 docs 中的旧路径按当时事实保留。）

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

## 研究工作流（research-loop）

实验轮次走固化流程：**P0 破题(预登记判决门) → P1 方案 → P2 Opus 红队审核(关键轮) →
P3 执行 → P4 分析 → P5 写 `docs/` md 汇报 → P6 确认后 git 提交**。三个必停门：
方案定稿 / 启动 >30min GPU 任务 / git 提交。

- 协议全文：`.claude/skills/research-loop/SKILL.md`（说"开新一轮/走工作流"即触发）
- 执行链：`Makefile`（`make help`；check-pool / triplets / train / eval-all / report / clean-ckpts）
- 每轮护栏：新数据先过 `scripts/check_t_src_leak.py`（内容级泄漏核查）；评测必带
  常数/平凡基线 + v2.1 & indomain 双测试集；引用历史数字前查 `docs/` 修正框

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
