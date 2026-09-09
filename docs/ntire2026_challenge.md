# NTIRE 2026 Challenge on Single Image Reflection Removal in the Wild: Datasets, Results, and Methods

**Organizers:** Jie Cai, Kangning Yang, Zhiyuan Li, Florin-Alexandru Vasluianu, Radu Timofte, Jinlong Li, Jinglin Shen, Zibo Meng (OPPO, Meta, University of Würzburg)  
**Conference:** CVPR 2026 Workshops (NTIRE)  
**Dataset:** [OpenRR-5k](https://huggingface.co/datasets/qiuzhangTiTi/OpenRR-5k)  
**Challenge Homepage:** https://github.com/caijie0620/OpenRR-5k

---

## 目录 / Index

| 页码 | 章节 |
|------|------|
| p.1 | Abstract, §1 Introduction |
| p.2 | Tab. 1 (Related methods overview) |
| p.3 | §2 NTIRE 2026 SIRR Challenge, §2.1 Overview, §2.2 OpenRR-5k Dataset |
| p.4 | §2.2 (continued), Fig. 2 OPPO AI Reflection Remover pipeline |
| p.5 | §2.2-2.5, Evaluation Metrics, Final Ranking |
| p.6 | Fig. 5 Visual comparison |
| p.7 | Fig. 6 Visual comparison |
| p.8 | Table 2 (Objective & Subjective Results) |
| p.9 | Table 3 (Technical Specifications) |
| p.10 | §4.1 RRay – RdNafNet |
| p.11 | §4.2 Xreflect Master – RDNet-XL + Diffusion Distillation |
| p.12 | §4.3 AIIALab, §4.4 VIP Lab, §4.5 YuFans |
| p.13 | §4.6 KLETech-CEVI, §4.7 PSU DUSKAN |
| p.14 | §4.8 SiGMoid, §4.9 NTR TimeDiffiT |
| p.15 | §4.10 refineX SIRR-Net |
| p.16 | §4.11 ACVLAB RDNet+, §5 Comparison with Generative Methods |
| p.17 | Fig. 18 Generative vs. discriminative comparison |
| p.18-19 | Appendix A: Teams and Affiliations |
| p.20-31 | References |

---

## Abstract

**Original:**

In this paper, we review the NTIRE 2026 challenge on single-image reflection removal (SIRR) in the wild. SIRR is a fundamental task in image restoration. Despite progress in academic research, most methods are tested on synthetic images or limited real-world images, creating a gap in real-world applications. In this challenge, we provide participants with the OpenRR-5k dataset. This dataset requires participants to process real-world images covering a range of reflection scenarios and intensities, aiming to generate clean images without reflections. The challenge attracted more than 100 registrations, with eleven of them participating in the final testing phase. The top-ranked methods advanced the state-of-the-art reflection removal performance and earned unanimous recognition from five experts in the field.

**中文:**

本文回顾了 NTIRE 2026 真实世界单幅图像反射去除挑战赛。SIRR 是图像恢复中的一项基础任务。尽管学术研究取得了进展，但大多数方法仅在合成图像或有限的真实图像上进行测试，在真实应用中存在差距。在本挑战中，我们向参与者提供了 OpenRR-5k 数据集，要求参与者处理涵盖各种反射场景和强度的真实世界图像，以生成无反射的干净图像。该挑战吸引了超过 100 个注册，其中 11 个队伍进入了最终测试阶段。排名靠前的方法推动了反射去除性能的最前沿，并获得了该领域五位专家的一致认可。

---

## 1. Introduction

**Original:**

SIRR is a critical task in image restoration, focusing on recovering the transmission layer T from an input image I with reflection contamination R caused by different reflective surfaces (e.g., transparent glass). Over the years, various techniques have been proposed to address the SIRR problem. Traditional methods typically rely on non-learning paradigms to mitigate the ill-posed nature of this problem. However, these methods typically rely on hand-crafted priors to guide the recovery process, which limits their ability to generalize well to diverse real-world scenarios. To address this issue, deep learning-based methods have been used to model the uncertainty of transmission estimation.

Despite deep learning's progress, the scarcity of high-quality data remains a key bottleneck.

**中文:**

SIRR 是一项关键的图像恢复任务，专注于从受到不同反射表面（如透明玻璃）导致的反射污染 R 的输入图像 I 中恢复透射层 T。多年来，研究者提出了各种技术来解决 SIRR 问题。传统方法通常依赖非学习范式来缓解这一问题的不适定性，但这些方法依赖手工设计的先验来引导恢复过程，限制了其在多样真实场景中的泛化能力。为解决这一问题，基于深度学习的方法被用于对透射估计的不确定性进行建模。

尽管深度学习取得了进展，高质量数据的稀缺仍然是一个关键瓶颈。

---

### Table 1. 现有 SIRR 方法概览

**Source:** p.2

论文汇总了现有 SIRR 方法的分类：

| 分类 | 方法 | 方案 |
|------|------|------|
| **单阶段** | Zhang et al. (CVPR 2018), ERRNet (CVPR 2019), RobustSIRR (CVPR 2023), YTMT (NeurIPS 2021), F2T2-HiT (ICIP 2025) | I → [T, R] 或 I → T |
| **两阶段** | CoRRN (TPAMI 2019), DMGN (TIP 2021), RAGNet (2023), CEILNet (ICCV 2017), DSRNet (ICCV 2023), SP-net/BT-net (CVPR 2020), Zheng et al. (CVPR 2021), Zhu et al. (CVPR 2024), Language-Guided (CVPR 2024), Cai et al. (MIPR 2025) | 先提取中间表示，再逐步恢复 |
| **多阶段** | BDN (ECCV 2018), IBCLN (CVPR 2020), Chang et al. (WACV 2021), LANet (ICCV 2021), V-DESIRR (ICCV 2021), L-DiffER (ECCV 2024), **RDNet (CVPR 2025)** | 递归/扩散模型，多步迭代优化 |

**Reading note:** Tab. 1 将现有方法按阶段数分类，并标注了 cross-stage fusion 方式。RDNet (CVPR 2025) 被归类为多阶段可逆架构（Recurrent + Convolutional Fusion）。

---

## 2. NTIRE 2026 SIRR Challenge

### 2.1 Overview

NTIRE 2026 workshop 包含约 30 个挑战赛道。本 SIRR 挑战的三个目标：

1. 为社区提供大规模真实世界 SIRR 基准，涵盖多样场景和客观性能协议
2. 激励探索能够处理真实反射固有复杂性和强度变化的 SIRR 架构
3. 弥合实验室研究与消费级成像解决方案之间的鸿沟

### 2.2 OpenRR-5k Dataset

**核心创新——AI驱动的数据采集协议：**

传统的反射去除数据采集需要物理操作（移开玻璃、用吸光黑布遮挡），存在效率低、对齐不准的问题。

OpenRR-5k 的采集流程（Fig. 1）：

1. **AI 反射去除工具生成初版 GT**：使用 OPPO Find X8 Ultra 手机内置的 AI Reflection Remover 生成初始透射结果
2. **后处理精炼**：使用 Adobe Photoshop、MeituPic 等专业工具进行人工精细调整，消除残留反射

**OPPO AI Reflection Remover 架构**（Fig. 2，2025 年 4 月版本）：

- **Stage 1**：YOLO + YOSO 检测强反射区域 → LaMa [60] Inpainting 填充掩码区域
- **Stage 2**：NAFNet [9] 进行精细反射去除

**数据集规模：**

| 子集 | 数量 | 用途 |
|------|------|------|
| OpenRR-5k~train~ | 5,000 对 | 训练 |
| OpenRR-5k~val~ | 300 对 | 验证 |
| OpenRR-5k~test~ | 100 张（仅输入） | 盲测（无 GT） |

**数据分布**（Fig. 4）：
- **图像主题**：风景 70%，物体 18%，人物 8%，动物 4%
- **光照条件**：白天 61%，室内照明 28%，夜间 11%

**与现有数据集的关键区别：** 包含真正的真实世界反射场景，不依赖人工合成，覆盖更广泛的场景和反射强度。

### 2.3 Evaluation Protocol

**验证集（客观指标）：** PSNR, SSIM, LPIPS, DISTS, NIQE

**测试集（主观评价）：** 5 位专家打分，三个维度加权平均：
- **C** (Cleanliness, 25%)：反射去除干净程度
- **A** (Artifacts, 25%)：伪影控制
- **Q** (Overall Image Quality, 50%)：整体图像质量（颜色、纹理、锐度、细节等）

最终得分：**S = 0.25C + 0.25A + 0.5Q**

**最终排名仅基于测试集主观得分。**

---

### Table 2. 完整结果排名

| 排名 | Team | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | NIQE ↓ | 主观分 ↑ |
|------|------|--------|--------|---------|---------|---------|---------|
| 1 | **RRay** | **36.1688** | 0.9758 | 0.0235 | 0.0135 | 3.7375 | **4.45** |
| 2 | Xreflect Master | 36.0496 | **0.9776** | **0.0210** | **0.0127** | 3.7648 | 4.31 |
| 3 | AIIALab | 35.3799 | 0.9750 | 0.0231 | 0.0155 | 3.7737 | 4.23 |
| 4 | VIP Lab | 34.6872 | 0.9766 | 0.0231 | 0.0148 | **3.7218** | 3.85 |
| 5 | YuFans | 34.9062 | 0.9738 | 0.0257 | 0.0159 | 3.7783 | 3.59 |
| 6 | KLETech-CEVI | 34.5375 | 0.9748 | 0.0242 | 0.0150 | 3.7566 | 3.57 |
| 7 | PSU | 34.5148 | 0.9746 | 0.0282 | 0.0191 | 3.7559 | 3.25 |
| 8 | SiGMoid | 34.2792 | 0.9749 | 0.0289 | 0.0205 | 3.7691 | 3.09 |
| 9 | NTR | 33.9679 | 0.9729 | 0.0323 | 0.0228 | 3.7639 | 3.01 |
| 10 | refineX | 30.5993 | 0.9715 | 0.0378 | 0.0291 | 3.7675 | 2.55 |
| — | OPPO-baseline | 31.6173 | 0.9229 | 0.0566 | 0.0539 | 3.6205 | — |
| — | ACVLAB (late) | 31.6886 | 0.9241 | 0.0541 | 0.0551 | 3.6393 | — |

**Reading note:**
- PSNR 与主观分呈中等相关但不严格单调——YuFans PSNR 高于 VIP Lab 但主观分更低
- 排名前 3 的方法（RRay, Xreflect Master, AIIALab）全部基于 **RDNet** 架构
- OPPO-baseline 使用 YOLO+LaMa+NAFNet 两阶段方案，是手机端部署方案

---

### Table 3. 技术规格与计算复杂度

| Team | 方法 | 训练数据 | 训练时间 | 推理时间 | 参数量 | FLOPs |
|------|------|---------|---------|---------|---------|-------|
| **RRay** | RDNet + NAFNet | OpenRR-5k + Real-Nature + RRW + Synthetic | 48h (A100) | 1.21s (A10) | **322.82M** | **1934.50G** |
| **Xreflect Master** | UNet (RDNet-XL) | OpenRR-5k + RR4K + RRW + DRR + perception-reflection-removal | 180h (8×A100) | 5.6s (A100) | 545.52M | 5175.93G |
| **AIIALab** | RDNet | OpenRR-5k + Nature | 352h (3090) | 1.6s (3090) | 314M | 7340G |
| **VIP Lab** | Park & Sim [47] | OpenRR-5k + RRW + Real-Nature + Synthetic | 71h (RTX 6000) | 0.988s (RTX 6000) | 404M | 2269G |
| **YuFans** | RDNet (FocalNet-L) | **仅 OpenRR-5k** | 12h (H800) | 5.0s (H800) | 522.34M | 671.27G |
| **KLETech-CEVI** | Encoder-Decoder CNN | **仅 OpenRR-5k** | 27h (RTX 6000) | 0.9s (RTX 6000) | **27.4M** | — |
| **PSU** | UNet + KAN | **仅 OpenRR-5k** | 3.46h (A100) | **0.74s (A100)** | 45.6M | 493.36G |
| **SiGMoid** | 4-stage Restormer | **仅 OpenRR-5k** | 180h (RTX 6000) | 2.36s (RTX 6000) | 26.13M | 3389.21G |
| **NTR** | TimeDiffiT (MDAE) | **仅 OpenRR-5k** | — | 6.5s (H200) | 142.5M | — |
| **refineX** | Restormer U-Net | **仅 OpenRR-5k** | 18h (3×RTX 6000) | 1.8s (RTX 6000) | **8.2M** | **120G** |
| **ACVLAB** | RDNet + DINOv2 | **仅 OpenRR-5k** | 35h (A100) | 528.39s (A100) | 267.17M | 5749.15G |

**Reading note:**
- Top-3 方法均使用**外部训练数据**，排名 4-10 中仅 VIP Lab 使用外部数据
- 使用外部数据是获得高主观分的关键因素之一
- YuFans 仅用 OpenRR-5k（无外部数据）通过 fine-tuning 技巧取得第 5 名，效率突出
- KLETech-CEVI (27.4M) 和 refineX (8.2M) 是轻量级方案

---

## 3. Challenge Methods — 各队伍方法详解

### 3.1 第 1 名：RRay — RdNafNet（两阶段级联网络）

**团队：** Intsig Information Co., Ltd.  
**核心思路：** RDNet 擅长去除显著反射但会丢失细节 → 级联 NAFNet 做细节增强

**架构（Fig. 7）：**
- **Stage 1 (RDNet)**：去除输入图像中的可见反射，输出中间结果
- **Stage 2 (NAFNet)**：对中间结果进行精细细节恢复

**训练数据：** OpenRR-5k~train~ + Real-Nature (289 pairs) + RRW (29,771 pairs) + 合成数据 (42,736 images)

**训练策略：**
- 使用 PyTorch + BasicSR 框架，AdamW 优化器
- 先 384×384 patch 训练 100k iter (MSE + gradient loss + VGG perceptual loss)
- 再 768×768 patch 训练 50k iter（加对抗损失）
- TTA：水平/垂直/组合翻转 → 加权融合

**关键 insight：** 两阶段级联（去反射→细节增强）+ 加权融合两个模型的输出。

---

### 3.2 第 2 名：Xreflect Master — RDNet-XL + 扩散蒸馏

**团队：** MiLM Plus, Xiaomi Inc.  
**核心思路：** 放大 RDNet 为 XL 版本 + 扩散模型知识蒸馏

**架构（Fig. 8）：**
- **FocalNet-XL** backbone
- **Decoder-XL** decoder
- **RDNet-XL** 作为主体（4 SubNets × 4 Levels）
- **扩散蒸馏**：使用 WindowSeat [78] 和 DAI [25] 生成伪 GT

**训练策略：**
- 渐进式分辨率：384 → 512 → 768
- 384/512 各 200 epochs，768 100 epochs
- **扩散蒸馏阶段**：768 checkpoint 后，用扩散模型生成 hard-case 伪 GT，额外 20 epochs
- 8×A100, bf16, batch size 2 per GPU
- AdamW, lr=2e-4, 每 50 epochs 减半
- TTA：水平+垂直翻转平均

**关键 insight：** VAE 编解码对齐设计——将反射图像也过一遍扩散 VAE 的 encode-decode 流程，确保输入与扩散生成监督的域对齐。

---

### 3.3 第 3 名：AIIALab — RDNet

**团队：** Harbin Institute of Technology  
**核心思路：** 直接使用 RDNet [82] 架构 + 额外 Nature 数据

**训练细节：**
- 使用 OpenRR-5k + Nature 数据集（合成数据）
- 352h 训练在 RTX 3090 上
- 参数量 314M，FLOPs 7340G（最高之一）

**关键 insight：** 不修改架构，依靠额外训练数据 + 充分训练取得第 3 名。

---

### 3.4 第 4 名：VIP Lab — Park & Sim [47]

**团队：** UNIST, Republic of Korea  
**架构：** 基于 [47] 的 reflection removal 专用架构

**训练数据：** OpenRR-5k + RRW + Real-Nature + Synthetic  
**训练策略：**
- 先 384×384 训练，再 768×768 精炼
- 损失函数：reconstruction loss + gradient loss + feature loss + load balancing loss
- TTA：原始 + 水平翻转 + 垂直翻转 → 平均

---

### 3.5 第 5 名：YuFans — 频率感知 Fine-Tuning

**团队：** National University of Singapore + Zhejiang University  
**核心思路：** RDNet 架构 + 仅用 OpenRR-5k + 无外部数据 + 频率感知微调  

**架构（Fig. 11）：**
- FocalNet-Large backbone (ImageNet-22K pretrained)
- RevCol: 4 SubNets × 4 Levels (64/128/256/512 channels)
- 3 × NAFBlock decoders
- Frozen ConvNext classifier (TAPG)

**两阶段微调策略：**
- **Stage 1**：从 SIRS 预训练权重初始化，用空间损失 (L1 + Gradient + VGG19 Perceptual + SSIM) 微调 20 epochs → 32.37 dB
- **Stage 2**：加入 Focal Frequency Loss (FFL, weight 0.01) 再微调 20 epochs → 提升至 34.90 dB
- **SWA (Stochastic Weight Averaging)**：Top-5 checkpoints 权重平均
- **Geometric TTA**：4 rotations × 2 flips = 8× 增强

**关键 insight：**
- **FFL 的有效性**：残留反射集中在特定频率范围，空间损失无法有效惩罚这些频率
- **数据使用的重要发现**：加入合成 SIRS 数据反而降低性能（32.77 → 31.94 dB），MixUp 也被破坏物理混合关系
- **差分学习率**：backbone 5e-6, decoder 2e-5, classifier 2e-4

**值得注意：** YuFans 是 6 个获奖队伍中**唯一仅使用 OpenRR-5k 训练数据**的队伍，说明优秀的微调策略可以弥补数据不足。

---

### 3.6 第 6 名：KLETech-CEVI

**团队：** KLE Technological University, India  
**架构：** Encoder-Decoder CNN，残差学习

**方法：**
- 编码器下采样 4 级（卷积 + ReLU + BN + MaxPooling）
- 解码器上采样重建
- 残差学习：输出预测反射分量 → ˆT = I − Fθ(I)
- 多尺度特征聚合

**Loss：** L_total = λ1·L_pixel + λ2·L_perc + λ3·L_ssim

**参数效率：** 仅 27.4M 参数，是前三名里效率最高的。

---

### 3.7 第 7 名：PSU — DUSKAN (Dual Spectral Kolmogorov-Arnold Network)

**团队：** Prince Sultan University, Saudi Arabia  
**核心思路：** 双路径架构——全局频域处理 + 局部 KAN 自适应激活

**架构（Fig. 13）：**
- 对称 4 级 U-Net backbone
- **Path A (Spectral-Spatial)**：2D FFT 幅度调制 → 空间位置编码 → SE reweighting，处理全局频率偏移
- **Path B (KAN Adaptive)**：KAN 多项式基激活函数 + 并行加性门控
- **门控融合**：每级可学习 logit α 混合两路径输出

**训练：** 仅用 OpenRR-5k，无外部数据。512×512 patches，标准几何增强。

**关键 insight：** KAN 在低级视觉任务中的应用——多项式基激活函数替代固定激活，处理空间变化的退化。

---

### 3.8 第 8 名：SiGMoid — 4-Stage Two-Head Restormer

**团队：** National Institute of Technology, Oita College + Kyushu Institute of Technology, Japan  
**架构：** 改进的 Restormer

**训练策略：**
- Patch size 256, batch size 1, AdamW, cosine scheduler
- EMA (decay 0.999)
- **Checkpoint Soup**：最佳 checkpoint + 2 个 continuation 快照 → 等权平均
- 轻量增强（水平翻转、小旋转、color jitter、gamma），避免破坏反射模式

**关键 insight：** Checkpoint soup（模型权重平均）是最可靠的提升手段之一，简单有效。

---

### 3.9 第 9 名：NTR — TimeDiffiT (Masked Diffusion Autoencoding)

**团队：** University of Illinois Urbana-Champaign, USA  
**核心思路：** 自监督预训练 + 监督微调

**两阶段策略：**
- **Stage 1: MDAE Pretraining** — 双退化：空间 masking (1%-75% blocks) + VE-SDE noise injection，用 MAE + Corruption2Self loss 无监督预训练
- **Stage 2: D2S-SFT Fine-tuning** — Diffusion-to-Score SFT loss 监督微调

**架构：** U-Net encoder-decoder，channel dims [128,256,512,1024]，时间条件通过 AdaGN 调制

**关键 insight：** 扩散模型的自监督预训练可以用于 SIRR，MDAE 策略在大规模无标签数据上学习通用图像先验。

---

### 3.10 第 10 名：refineX — SIRR-Net (Progressive Restormer U-Net)

**团队：** KLE Technological University, India  
**架构（Fig. 16）：** Restormer 风格的层次化 U-Net，4 个分辨率 level

**核心组件：**
- **MDTA (Multi-Dconv Head Transposed Attention)**：通道注意力，复杂度 O(C²)
- **GDFN (Gated Depthwise FFN)**：SimpleGate 门控机制
- **SE (Squeeze-and-Excite)**：通道重标定
- **Global residual connection**：ˆT = clamp(Dec(Enc(I)) + I, 0, 1)

**参数效率：** 仅 **8.2M** 参数，**120G** FLOPs，是全部方法中最轻量的。

---

### 3.11 ACVLAB (Late Submission) — RDNet + DINOv2

**团队：** National Yang Ming Chiao Tung University + National Cheng Kung University + NVIDIA  
**核心思路：** RDNet + 冻结 DINOv2 ViT-B/14 作为辅助语义先验

**动机：** 在强镜面反射下，FocalNet 特征缺少足够的目标级语义来推断反射去除后的合理外观。

**架构（Fig. 17）：**
- RDNet backbone (FocalNet-L PHE + MCRE + TAPG + HDec)
- 冻结的 DINOv2 ViT-B/14 分支
- 4 个 1×1 Conv adapters → 映射到 MCRE 各 level (64,128,256,512 ch)

**训练：** OpenRR-1k + OpenRR-5k, AdamW, ReduceLROnPlateau

**关键 insight：** 冻结 DINOv2 提供丰富语义先验，训练成本仅限 4 个轻量 adapter，显著减少模糊伪影。

---

## 4. Comparison with Generative Methods (§5)

**评估的生成方法：** gpt-image-1.5, Nano Banana Pro, Nano Banana 2

**主要发现：**
- **细节恢复**：Nano Banana 系列在恢复锐利细节、抑制残留鬼影方面表现更优
- **语义一致性**：重度遮挡区域保持更好的语义完整性，但可能有随机变化
- **零样本泛化**：无需微调即可处理多样的反射场景
- **局限**：可能将强反射误识别为背景结构，导致伪影放大

**Reading note:** 生成方法的定性结果优秀但定量（pixel-wise）未必占优，与判别式方法互补。Fig. 18 显示 Nano Banana Pro 在去反射细节上视觉上优于 RRay 和 OPPO-baseline。

---

## 5. 关键观察与结论

1. **RDNet 占据主导地位**：Top-3 全部基于 RDNet [82]（CVPR 2025），证明可逆解耦架构在 SIRR 中的有效性。

2. **外部数据是关键差异**：排名前 4 的方法均使用外部训练数据，仅用 OpenRR-5k 的队伍最高排名为第 5 名（YuFans）。

3. **主观 vs. 客观指标**：PSNR 与主观分并非严格单调相关——最终排名完全基于主观评价，强调感知质量的重要性。

4. **两阶段/级联设计趋势**：RRay (RDNet+NAFNet) 和 OPPO-baseline (YOLO+LaMa+NAFNet) 均采用级联策略，先强去除再精修。

5. **轻量方案值得关注**：KLETech-CEVI (27.4M) 和 refineX (8.2M) 以极小参数量取得有竞争力的结果，适合边缘部署。

6. **生成方法的潜力**：Nano Banana 等生成模型在重度反射区域的细节恢复上展示出优势，但可靠性仍需改进。

---

## 术语表

| English | 中文 |
|---------|------|
| Transmission layer | 透射层 |
| Reflection layer | 反射层 |
| Subjective evaluation | 主观评价 |
| Objective metrics | 客观指标 |
| Test-time augmentation (TTA) | 测试时增强 |
| Knowledge distillation | 知识蒸馏 |
| Diffusion distillation | 扩散蒸馏 |
| Checkpoint soup / weight averaging | 模型权重平均 |
| Focal Frequency Loss (FFL) | 焦点频率损失 |
| Masked Diffusion Autoencoding (MDAE) | 掩码扩散自编码 |
| Kolmogorov-Arnold Network (KAN) | KAN 网络 |
| Multiscale | 多尺度 |
| Residual learning | 残差学习 |

---

## 阅读提示 / Reading Notes

1. **OpenRR-5k 数据集的意义**：首个大规模（5,300 对）、真实世界、AI辅助生成的反射去除基准。使用手机 AI 去反射工具 + 人工精修替代传统物理遮挡方法，大幅提高了 GT 质量和采集效率。

2. **评价体系的设计**：主观评价权重占 100% 的最终排名 + 三个维度（清洁度 25%、伪影 25%、整体质量 50%）体现了从"数值好"到"看着好"的范式转变。

3. **RDNet 生态**：NTIRE 2026 相当于 RDNet 的"大阅兵"——多个队伍基于 RDNet 改进，验证了该架构的强泛化性和可扩展性。

4. **YuFans 最具启发性的工作**：仅用 5,000 张训练数据、不引入外部数据、通过 Focal Frequency Loss + SWA + 几何 TTA 获得第 5 名，对数据有限的场景最具参考价值。

5. **值得关注的技术方向**：
   - 扩散模型蒸馏（Xreflect Master）
   - KAN 网络用于低级视觉（PSU）
   - 语义先验注入 (DINOv2, ACVLAB)
   - Checkpoint soup / SWA (SiGMoid, YuFans)
   - 频率域损失 (YuFans FFL)

6. **局限**：各方法在 Table 3 中的技术指标（参数量、FLOPs、推理时间）由各队伍自行报告，公平性存疑；Challenges 论文本身没有统一的重现实验。

7. **未来方向**：生成方法在重度反射区域展示了超越判别式方法的潜力，GAN 扩散混合架构可能是下一个突破口。
