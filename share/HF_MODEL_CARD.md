---
license: other
license_name: research-use-only
tags:
  - reflection-removal
  - single-image-reflection-removal
  - pytorch
  - rdnet
  - fine-tuned
---

# TintedGlass RDNet — 贴膜玻璃合成域微调权重

本仓库是 **RDNet（CVPR 2025, [lime-j/RDNet](https://github.com/lime-j/RDNet)）** 在
贴膜玻璃合成数据（`zmy1234567890/tintedglass-rdt-synth`，`train/rdt_0915`）上微调的
4 个 checkpoint。官方 `rdnet.pth` 是**必需的底座**（本仓库只存微调后的完整 state_dict，
未打包官方权重）。

## 权重清单

| 文件 | 逆衰减加权 | 随机种子 | 文件大小 |
|---|---|---|---|
| `rdt_plain/ckpt_final.pth` | 关（`--lam-wlow 0`） | 20260915 | ~1.5G |
| `rdt_freeze/ckpt_final.pth` | 开（`w(α)=clamp((0.3/α)²,1,10)`） | 20260915 | ~1.5G |
| `rdt_plain_s101/ckpt_final.pth` | 关 | 101 | ~1.5G |
| `rdt_freeze_s101/ckpt_final.pth` | 开 | 101 | ~1.5G |

每个目录另附 `train_log.csv`（逐 50 iter 的 L_t/L_r/L_recons/total）。

> 命名说明：`freeze/plain` 指**损失里的低 α 加权开关**，不是网络冻结与否——
> 4 个 ckpt 都冻结 FocalNet 主干、训练其余参数，网络结构与官方完全一致。

## 训练配置

| 项 | 值 |
|---|---|
| 数据 | `tintedglass-rdt-synth` / `train/rdt_0915`（4000 场景 × 5 档） |
| 迭代 | 30,000 iter；随机裁剪按 224²(b8) → 256²(b6) → 320²(b4) 轮换（10GB 卡） |
| 优化 | Adam, lr 1e-4, cosine（下限 0.1×）, AMP |
| 冻结 | FocalNet 主干冻结（`baseball.*`），其余参数训练 |
| 损失 | `Σ列 (w_col/Σw)·[Charbonnier(T̂,T)·w(α) + 1.5·Charbonnier(R̂,R)] + 0.2·\|I−T̂−R̂\|` |
| 加权 | `w(α)=clamp((0.3/α)², 1, 10)`（仅 `*_freeze*` 开启） |
| 硬件 | 单卡 RTX 3080 10GB |

## 载入方式

```python
import torch
ck = torch.load("rdt_freeze/ckpt_final.pth", map_location="cpu")
net.load_state_dict(ck["icnn"])      # state_dict 与官方 rdnet.pth 同构
print(ck["iter"], ck["args"])        # 训练元信息
```

配套代码：`tintedglass-synth` 分支的 `scripts/finetune_rdnet_tint.py`（训练）与
`scripts/eval_rdt_tint.py` / `eval_real3_ft.py`（评测）。

## 评测结果

**合成集（v2.1，seed 固定）**：微调显著改善贴膜档结构指标（相对官方零样本）。

**实拍配对集（gain-matched PSNR 中位 dB）**：

| 数据集 | identity | identity_gm | 官方零样本_gm | ft_plain_gm | ft_plain_s101_gm | ft_freeze_gm |
|---|---|---|---|---|---|---|
| 6 月（13 输入） | 19.40 | 22.40 | **22.73** | 20.07 | 20.12 | 20.31 |
| 4 月（16 输入） | 17.93 | 20.01 | **17.84** | 16.68 | 16.80 | 17.02 |
| 5 月（14 输入） | 20.26 | 23.51 | **24.07** | 19.89 | 20.19 | 20.17 |

对官方零样本的场景级 Wilcoxon 配对检验：**9/9 格全部显著为负**（p ≤ 0.029），
即**合成域微调不迁移到实拍**（−0.44 ~ −3.87 dB）。候选解释为训练/测试域的系统性差异：

- 合成 T 无噪声、无 JPEG；实拍含压缩与传感器噪声
- 合成 R 来自公开反射视频（RRWDataset）时域中值提层；实拍反射含拍摄者自反射等形态
- 合成 α 场是人工构造的大尺度场；实拍膜的 α 场统计未对齐

**请把它当作域差研究的素材与可复现的负结果基线，而不是能直接部署去反射的权重。**

## 许可

代码与权重来自 [lime-j/RDNet](https://github.com/lime-j/RDNet)（CVPR 2025）+ 本项目微调，
仅供研究用途；上游数据集许可请自行核对。
