# 预登记 · LightRR 微调工程可行性（2026-11-06 exp-dev，不追求效果）

> 只回答：**LightRR（9.5M，AFM Mamba+小波）能不能在我们的数据上微调起来**。
> 先落判据再跑；**不污染 xreflection**；不追求效果。

## 资产（服务器）
- 代码：`/DataDisk/zmy_code/LightRR-CVPR26`（作者官方 clone；本地也有 `D:\...\reflection_removal\lightrr\LightRR-CVPR26`）
- 权重：`/DataDisk/zmy_code/LightRR-CVPR26/weights/lightrr/lightrr_woNature.pt`（113,742,966 B，epoch=58，9.50 M）
- 现成环境：`/DataDisk/zmy_code/LightRR_env/lightrr/bin/python`（py3.8 + torch 2.4.0+cu121 + **已编译 mamba_ssm 2.2.2 / causal_conv1d 1.4.0**；2026-10-12 搭好，推理已验证）
- 数据：`/DataDisk/zmy_code/TintedGlass_Enhance/data/train_sets/real_pairs_*`（I/T 同名 png，真实配对）
- GPU：8×RTX3080 10GB（共享，他人占用时余量小；训练一律 `CUDA_VISIBLE_DEVICES` 钉卡）

## 四个问题（按序）
1. **依赖**：Mamba 算子是否必须？版本？在 3080 能否编译？纯 PyTorch/Triton 回退？
2. **数据格式**：`data/sirs_dataset_dual.py` / `data/image_folder.py` 要什么结构/字段（`scalar_a/b`、`target_r`、双输入？）→ 写最小适配脚本把 (I,T) 转过去。
3. **冒烟训练（最重要）**：10 步前向+反传，loss 下降、显存、bs、单步耗时 → 估 20–30k 步成本。
4. **推理复现**：官方权重在我们 1019 上零样本，PSNR/结构指标；对照我们记录 19.18 dB（`2026.10.12_LightRR接入…`）。对不上先修口径，微调无意义。

## 判据（一行）
- **①装得上 + ③能训 + ④推理对得上 ⇒ 可行**，报训练成本；
- 任一项卡住 ⇒ **不可行**，报卡在哪 + 绕过办法。

## 纪律 / 止损
- **先冒烟 10 步，绝不长训**；
- 数据适配只做最小（不跑大数据管线）；
- 优先用现有 `LightRR_env`（不新建、不升级、不编译 mamba——除非现有环境坏了）；
- 若必须动 `xreflection`：**先报告再动**（本预检预期不需要）；
- 冒烟若 OOM，先降 bs/裁到 224/320，再报边界；
- 结果落 `/tmp/ri_dev/lightrr_ft_feasibility_1106.md` + 本地 `code/local_tools/`。

## 已知风险
- 3090→3080 (sm86) 不影响已编译的 mamba CUDA 算子（sm80 兼容）；
- 训练需要 `train_S.py` 的 `--data_root` 结构；README 里数据管线可能与推理 `eval_sirs.py` 不同；
- 共享卡占用可能使冒烟 OOM/排队，需重试或等窗口（记录当时卡占用）。
