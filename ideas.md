# 深色贴膜玻璃透射成像增强 —— 改进方案与文献支撑

> 基于 RDNet [CVPR 2025] 架构改进，针对 α 空间变化且 << 1 的贴膜玻璃场景。
> 
> **2026.6 更新批注**：近期阅读 L-DiffER [ECCV 2024]、DAI/DRR [AAAI 2026]、DSIT [NeurIPS 2024] 后，发现若干可借鉴的思路。特别值得注意的是，**DSIT 与 RDNet 同属天津大学郭晓杰组**（Qiming Hu 横跨 YTMT → DSRNet → DSIT → RDNet 全部四篇），RDNet 是 DSIT 之后的工作。这意味着 DSIT 中被 RDNet **跳过**（而非否定）的小机制——RefMix、梯度排斥 loss、LRPB 分层标记——很可能仍然有效，只是该组优先探索了可逆列架构方向。以下在各方案中逐一标注。

---

## 2026.9.3 路线定版（决策记录）

> 背景：2026.9.3 完成 SIRR 2026 新文献调研（`2026.9.3_反射去除调研.md`：GenSIRR/GFRRN/WindowSeat/PRISM/EvReflection），并确认本机算力约束（8–16 GB 显存，无大模型训练条件）。

- **主线定版：纯"联合方法"论文** —— 车窗场景端到端"反射去除 + 低光增强"（FullTriAxisNet 线，见 timeline_index 2026.6 转向），真实四步法数据为护城河；**不卷通用 SIRR 基准精度**（该基准已被扩散方法占领且仍在上涨，拼不过也不必要）。
- **对比与叙事**：判别式同族（RDNet/DSIT/GFRRN）+ 可运行扩散参照（DAI）在**车窗自有评测集**上对比 + fail-case 佐证；通用基准只作坐标。story 支点 = "通用方法在真实黑膜车窗（低光+强反射耦合退化）上失效，专用联合建模必要"。
- **模型压缩/端侧方向：搁置（2026.9）**。理由：无真实端侧部署场景（无车载平台）、8–16GB 算力撑不起"微调 12B 级扩散"的原始设想、扩散压缩文献竞争激烈。**复活条件**：① 拿到端侧/产品场景；② 需要 GenSIRR 类生成式 teacher 离线蒸馏/造 pseudo-GT 时，以"离线 teacher + 轻量 student"形式重启（GenSIRR 论文自列蒸馏为 future work，此线不过期）。
- **扩散/大模型定位：只借不训** —— 官方权重推理（GenSIRR HF 权重、DAI 权重、WindowSeat HF demo）做车窗 fail-case 与无参考评测参照，不进方案设计。
- 由此 Phase 3 的 G（扩散先验注入）与 I（MDAE 预训练）**降级为背景/探索项**；资源优先投 Phase 1/2 与 FullTriAxisNet 迭代。
- **主攻架构草案（2026.9.3）**：联合"迭代分离—增强—反馈"循环设计（方案 L：双流试分 + 低光增强 + γ_k 系数融合反馈 ×3 轮 + α(x)/多轮一致双轨约束），见 `2026.9.3_联合迭代分离增强_idea.md`。

---

## Overview 总览

| ID | 方向 | 类型 | 来源参考 | 实现成本 | 预期收益 | 优先级 |
|----|------|------|---------|---------|---------|--------|
| A | 退化程度识别 | 架构改进 | RDNet TAPG + SPADE | 低 | 中-高 | ★★★ |
| B | 物理先验注入 | 输入增强 | Retinex + DoG + YTMT | 低 | 高 | ★★★ |
| C | 语义先验辅助 | 架构改进 | DINOv2 + ControlNet | 中 | 中 | ★★ |
| D | 频率域显式监督 | 损失函数 | Focal Frequency Loss | 低 | 中 | ★★★ |
| E | 可逆级联解码 | 架构改进 | RDNet 可逆单元 + Recurrent | 中 | 高 | ★★ |
| F | 条件细节精修 | 后处理模块 | RRay/NTIRE-1st | 低 | 中 | ★★ |
| G | 扩散先验注入 | 架构改进 | Xreflect Master + ControlNet | 高 | 高 | ★ |
| H | Checkpoint Soup | 训练策略 | SiGMoid/NTIRE-8th + SWA | 低 | 低-中 | ★★ |
| I | Masked Diffusion Pre-training | 预训练策略 | NTR TimeDiffiT + MAE | 中 | 中 | ★ |
| J | 贴膜玻璃数据采集 Pipeline | 数据工程 | DAI DRR + 旋转玻璃 | 低-中 | 高 | ★★★ |
| K | DSIT 遗留小机制 | 训练策略/Loss | DSIT + L-DiffER | 极低 | 中 | ★★★ |

---

## 同组演进：YTMT → DSRNet → DSIT → RDNet

全部来自**天津大学郭晓杰组**（Qiming Hu 贯穿四篇），RDNet 是该组最新工作（CVPR 2025）。理解这组在架构选择上的取舍，对 RDNet 改进有直接参考价值。

| 版本 | 会议 | 交互方式 | 全局先验 | 被 RDNet 继承？ |
|------|------|---------|---------|----------------|
| YTMT | NeurIPS 2021 | 激活门控选择 | HyperColumn (VGG) | 交互思想继承，具体机制被替换 |
| DSRNet | ICCV 2023 | MuGI 高效交互 | Feature Pyramid | MuGI 成为 RDNet 列内部模块 |
| **DSIT** | NeurIPS 2024 | **DS-CA 交叉注意力** | **冻结 Swin-L** | **几乎全部丢弃** |
| **RDNet** | CVPR 2025 | **可逆列 (RevCol)** | **ConvNext TAPG** | 当前 baseline |

**关键解读**：

1. **DSIT 的注意力交互被 RDNet 完全替换为可逆列**——该组试过注意力路线后选择了另一条路。这不意味着注意力无效（DSIT 定量结果很强），更可能是注意力交互训练不稳定、计算量大、或该组认为可逆结构更优雅
2. **Swin-L 大模型被降级为 ConvNext TAPG**——大模型冻结注入太重，TAPG 用 6 维向量达到类似效果
3. **DSIT 中被 RDNet 跳过的机制 = 可以回补的机会**：RefMix、梯度排斥 loss、LRPB 分层标记——这些 DSIT 验证有效的设计，在 RDNet 上未做消融，直接加上很可能有收益



---

### K. DSIT 遗留小机制——零成本回补 RDNet

以下三个机制在 DSIT 中验证有效，但 RDNet 未采用（可能被架构大改跳过而非否定），可直接加入 RDNet 训练 pipeline。

**K1. RefMix（反射混合增强）**
- 公式：`I_aug = γ·I + (1-γ)·T`，γ ∈ [0,1]
- DSIT 消融：+0.34dB (SIR2)，+0.34dB (Real20)
- RDNet 适配：训练数据 DRR/合成/Nature 都是对齐的，直接可用。相当于零成本调节反射强度，增加训练多样性
- 实现：DataLoader 中一行

**K2. 多尺度梯度排斥 Loss**
- 公式：`L_excl = mean(tanh(ξ₁|∇T̂|) ⊙ tanh(ξ₂|∇R̂|))²`，多尺度下采样后计算
- DSIT 验证：显式惩罚 T/R 梯度重叠，对 ghosting artifact 抑制有效
- RDNet 适配：RDNet 损失为 `L_cont + L_per + L_TAPG`，L_cont 只约束重建不做梯度解耦。L_excl 可直接加到 L_cont 旁边
- 改动：~15 行

**K3. MVC 物理约束（来自 L-DiffER）**
- 公式：`L_MVC = ||max(T̂ - M, 0)||₁`
- 基于 M = T + R：透射层任意像素不应超过混合图像
- RDNet/DSIT/DSRNet 均未使用此约束
- 改动：~3 行

**K4. LRPB 式列标记（来自 DSIT，思路迁移）**
- DSIT 给 T/R token 加 3D 位置偏置 (x, y, layer) 来区分信息来源
- RDNet 的 MCRE 中 T 列和 R 列通过可逆公式隐式交互，但没有显式的"我是 T/R"标记
- 方案：给 MCRE 的 T/R 列输入各加一个可学习的 column embedding vector
- 参数量增加可忽略不计（~512 维），改动 ~20 行

**参考文献**：
- DSIT [Hu et al., NeurIPS 2024] — K1/K2/K4 的来源
- L-DiffER [Hong et al., ECCV 2024] — K3 MVC 的来源

---

## 方案详述

### A. Spatial TAPG——空间自适应退化程度估计

**思路：** RDNet 的 TAPG 用 ConvNext 输出 6 个全局标量 (α_RGB, β_RGB)，但在 α 空间变化的贴膜玻璃上不够。改为同时输出全局参数 + 空间 α(x) map。

**实现：**
- ConvNext backbone 分叉为两个 head
  - **global head**（保留原结构）：6 维 → MLP → prompt P，用于列嵌入初始调制
  - **spatial head**（新增）：pixel-wise regression → α(x) ∈ ℝ¹×ᴴ×ᵂ
- α(x) 无需额外标注，用自监督重建约束训练：
  `L_alpha = ||I(x) - α(x)T̂(x) - βR̂(x)||₁`，其中 T̂ 和 R̂ 来自 MCRE 粗输出

> **批注 A1 [2026.6]**：DAI [AAAI 2026] 的反射不变微调（RIF）策略可为此方案提供另一种训练信号——若采集"同一场景 + 不同反射"的图像对，可额外约束 α(x) 在不同反射条件下保持一致性：`L_alpha_con = ||α₁(x) - α₂(x)||₂`。这比纯自监督重建约束更强，因为 α(x) 作为透射场景的固有属性，理应与反射变化无关。但前提是需要类似 DRR 的数据采集方式（旋转玻璃），对贴膜玻璃场景需要设计对应的采集 protocol。

**参考文献：**
- RDNet TAPG [CVPR 2025] — 原始全局 TAPG，是此方案的 baseline
- SPADE [Park et al., CVPR 2019] — 空间自适应归一化，spatial condition 的代表性设计范式 [link](https://arxiv.org/abs/1903.07291)
- Deep Parametric Indoor Lighting Estimation [Gardner et al., ICCV 2019] — pixel-wise parameter prediction 的思路
- DSRNet [Hu & Guo, ICCV 2023] — 退化模型 I = αT + βR - T∘R，为自监督 α 估计提供了物理框架 [link](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Single_Image_Reflection_Separation_via_Component_Synergy_ICCV_2023_paper.html)

---

### B. 物理先验通道注入——在输入层提供解耦线索

**思路：** MCRE 只靠 FocalNet 的语义特征来解耦 T 和 R。在 α << 1 场景下语义不可靠时，物理线索是关键补充。在 PHE 输入前将 I 扩展为多通道，附加四个零成本物理先验。

**四个先验通道：**

| 先验 | 物理原理 | 计算方式 | 参考文献 |
|------|---------|---------|---------|
| B1. blur_map | 反射在玻璃表面近焦边缘锐利，透射在后离焦模糊 | 局部 Laplacian 方差 / patch-wise gradient magnitude ratio | Edge-guided Reflection Removal [Chang et al., WACV 2021] |
| B2. ghost_map | 双表面反射呈水平偏移复现 | 局部水平自相关峰值检测 | Reflection removal using ghosting cues [Shih et al., CVPR 2015] |
| B3. saturation_residual | 贴膜色偏使透射区白平衡残差大，反射区光源色温正常 | 经白平衡校正后 per-pixel chromaticity deviation | Gray-world assumption + 修正 |
| B4. edge_ratio | 不同频段边缘形态差异 | Sobel 响应高/低频比值 | CEILNet [Fan et al., ICCV 2017] |

**扩展思路（超越传统 CV）：** 附加通道也可由轻量级网络预测而非硬编码公式——但传统方法的 zero-shot 可控性是其优势（训练集外场景不退化）。

> **批注 B1 [2026.6]**：L-DiffER [ECCV 2024] 的双条件分离设计（颜色条件 Ic 控制色调 + 结构条件 Is = ∇M 控制边缘纹理）与本方案的"物理先验通道注入"异曲同工，但它把条件注入从输入层搬到了 diffusion latent space。可借鉴的点：① 将 B3（saturation_residual）作为"颜色条件"的增强版单独编码，B2/B4（ghost/edge）作为"结构条件"的增强版单独编码，两条路径分别进入 PHE 的不同 head；② L-DiffER 的 MVC（最大像素值约束）可作为一种训练时的物理约束 loss 加入到贴膜场景中：`L_MVC = ||max(T̂ - M, 0)||₁`，强制网络不生成超越混合图像的透射像素。

**参考文献：**
- CEILNet [Fan et al., ICCV 2017] — 用 edge map 作为显式引导输入到网络 [link](https://openaccess.thecvf.com/content_ICCV_2017/papers/Fan_CEILNet_A_Generic_ICCV_2017_paper.pdf)
- Edge-guided Reflection Removal [Chang et al., WACV 2021] — 利用边缘信息辅助反射检测 [link](https://openaccess.thecvf.com/content/WACV2021/papers/Chang_Single_Image_Reflection_Removal_With_Edge_Guidance_Reflection_Classifier_and_WACV_2021_paper.pdf)
- Reflection removal using ghosting cues [Shih et al., CVPR 2015] — 利用双反射鬼影线索 [link](https://ieeexplore.ieee.org/document/7298960)
- Zhu et al. [CVPR 2024] — 显式检测反射边缘作为 mask 用于 guided restoration [link](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Edge-guided_Reflection_Removal_Using_Reflection_Edge_Detection_Mask_CVPR_2024_paper.html)

---

### C. 语义先验辅助注入——DINOv2 Adapter

**思路：** 在强反射完全淹没透射内容时，物理先验也无法恢复——需要高级语义先验来"推测"被遮住的部分。参考 ACVLAB 的方案：将 DINOv2 [ViT-B/14] 作为辅助语义分支注入 MCRE。但 ACVLAB 是 element-wise addition 融合，可改为更灵活的 cross-attention 注入。

**实现：**
- 输入 resize 到 518×518 → 冻结 DINOv2 → patch tokens
- 4 个 1×1 Conv adapters → 映射到 MCRE 各 level (64/128/256/512)
- **改进点**：将 element-wise addition 改为 cross-attention 注入（DINOv2 token → MCRE feature as cross-attention KV），让 MCRE 自适应地从语义先验中"提取"需要的信息，而不是被强灌

**参考文献：**
- DINOv2 [Oquab et al., 2023] — 自监督视觉特征，适合零样本语义先验 [link](https://arxiv.org/abs/2304.07193)
- ACVLAB, RDNet + DINOv2 [NTIRE 2026] — 将 DINOv2 注入 RDNet 的已有验证 [link](https://github.com/wuw2135/RDNet-with-Frozen-DINOv2-Semantic-Prior-for-Reflection-Removal)
- ControlNet [Zhang et al., ICCV 2023] — 冻结 backbone + 可训练 adapter 的范式，可作为改进接入方式的参考 [link](https://arxiv.org/abs/2302.05543)
- GLOM [Hinton, 2021] — part-whole hierarchy 的原始概念框架，与 MCRE 的列结构同源 [link](https://arxiv.org/abs/2102.12627)

> **批注 C1 [2026.6]**：方案 C 的"DINOv2 → MCRE cross-attention"范式与 DSIT 的 GPE（冻结 Swin-L → 双流 CNN）逻辑一致。但需要注意：**该组在 RDNet 中主动放弃了 Swin-L 大模型，换成了更轻量的 ConvNext TAPG**。这意味着冻结大模型做 adapter 注入在 SIRR 中可能"有效但不划算"——DSIT 验证了可行方向，但该组在后来的 RDNet 中选择了更轻的方案。方案 C 若走 DINOv2 路线，需要在"收益 vs 推理成本"上做严格控制（如仅对极低 α 区域激活 DINOv2 分支）。
> 
> **批注 C2 [2026.6]**：一个值得尝试的折中：不用 DINOv2 ViT-B/14 的全量 patch tokens，仅用其 [CLS] token 或 register tokens 作为全局语义向量，通过类似 TAPG 的 MLP 调制方式注入 MCRE。这样保留了语义先验的信息量，同时避免了 patch token 的计算膨胀。

---

> **批注 C3 [2026.9]**：GFRRN [CVPR 2026] 在 SIRR 上验证了同一范式——给冻结 Swin 骨干插可学习 adapter（Mona-tuning）值 **+0.63dB**，且全量微调因数据量不足反而次优；该文还承认 RDNet 注意到同类问题（TAPG 改用可训 ConvNext）。→ 方案 C 的"DINOv2 adapter"路线在 SIRR 任务上有了直接先例背书；C2 的"仅 CLS token + 调制注入"折中仍是弱卡首选。注意 GFRRN 训练协议偏重（A6000 48GB、bs=1、fp32、384²）——只借鉴思想，不整网复现。

### D. 频率域显式监督——Focal Frequency Loss + Gradient Mask

**思路：** 反射残留集中在特定频率范围（中低频 ghost + 锐利边缘），标准 L1/L2 对所有频率均等惩罚。借用 YuFans 的 FFL 思路，并扩展为**频率加权 + 空间掩码**的联合损失。

**实现：**
```
L_FFL = (1/N) · Σ_uv |F(T̂) - F(T)|² · w(u,v)
```
其中 w(u,v) 在中低频权重放大（针对反射残留的高发频段）

扩展：结合 α(x) map 做空间加权
```
L_final = L_cont + w₁·L_per + w₂·L_FFL + w₃·L_alpha
         + η·Σ_x (1-α(x)) · L_per(x)    ← 严重衰减区放感知损失权重
```

**参考文献：**
- Focal Frequency Loss [Jiang et al., ICCV 2021] — 自适应频率加权 loss 的原始方法 [link](https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.html)
- YuFans [NTIRE 2026, 5th] — 将 FFL 用于 RDNet 微调并验证了效果（34.90 dB），仅凭 OpenRR-5k 取得第 5 [link](https://github.com/caijie0620/OpenRR-5k)
- Perceptual Loss [Johnson et al., ECCV 2016] — VGG-based perceptual loss [link](https://arxiv.org/abs/1603.08155)

> **批注 D1 [2026.6]**：DAI [AAAI 2026] 的 Cross-Latent Decoder（CLD）为"缓解模糊"提供了另一种非 loss 层面的解法——在解码器端通过跳跃连接注入编码器特征来保留高频细节，而非仅在 loss 层面对频率加权。两者可互补：FFL 在训练时引导频率关注，CLD 式的 skip connection 在架构层面保留细节通道。若在 HDec 中加入类似 DAI 的 latent-to-pixel skip connection，预期能与 FFL 产生叠加收益。

---

> **批注 D2 [2026.9]**：GFRRN 的 G-AFLB（解码器内高斯掩码自适应频率学习）消融 **+0.25dB**、DAA 动态注意力 **+0.42dB**——频域处理也可在**架构级**做（不限于 FFL 的 loss 级）：若 FFL 提分不明显，可在解码器加轻量频域块试。另外：GFRRN 的 unified label（反射标签取 (I−T) 的低频）对四步法数据**意义有限**——我们有干净真实反射 GT，不存在合成/真实标签打架；仅在将来混入合成数据时才需启用。

### E. 可逆迭代级联——同一 MCRE 多轮循环

**思路：** RRay 将 RDNet + NAFNet 级联是"两套权重先后跑"，成本高（322M + NAFNet 的额外参数）。改为**同一套 MCRE 权重多轮迭代**——借鉴 RDNet 可逆单元本身就是循环结构这一特性（前向公式 T̂₂ = T̂₁ + F(R̂₁) 天然可展开）。

**实现：**
```
Iter 1: T₁, R₁ = MCRE(I | α₁, β₁)
Iter 2: T₂, R₂ = MCRE(I | α₂(T₁), β₂(T₁))  ← TAPG 条件基于上一轮输出
Iter 3: T₃, R₃ = MCRE(I | α₃(T₂), β₃(T₂))
```

- MCRE 权重跨轮共享，只 TAPG 的 MLP 在迭代间接收新条件
- 训练时只反向传播最后一轮或使用截断 BPTT（与 RevCol 原始训练一致）
- TAPG 输入从 I 替换为 concat(I, T_{k-1})——让网络修正自己上一轮的遗漏

**参考文献：**
- RDNet [Zhao et al., CVPR 2025] — 可逆单元的数学形式天然支持多轮展开 [link](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Reversible_Decoupling_Network_for_Single_Image_Reflection_Removal_CVPR_2025_paper.html)
- RevCol [Du et al., NeurIPS 2022] — 可逆列网络，RDNet 的基础架构 [link](https://arxiv.org/abs/2210.11610)
- RRay [NTIRE 2026, 1st] — RDNet + NAFNet 级联方案，验证了"先粗去后精修"路径有效 [link](https://github.com/caijie0620/OpenRR-5k)
- IBCLN [Li et al., CVPR 2020] — 迭代式反射去除的早期代表（LSTM-like refinement）[link](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Single_Image_Reflection_Removal_Through_a_Cascaded_Refinement_With_CVPR_2020_paper.html)

> **批注 E1 [2026.6]**：DAI [AAAI 2026] 的三阶段渐进训练（Foundation → RIF → CLD）是一种"分阶段收敛"的策略，与本方案的"多轮迭代 MCRE"可结合：训练时先训单轮 MCRE（等价于 DAI Stage 1），再引入迭代展开（等价于将 DAI Stage 2 的 RIF 一致性思想从数据层面搬到架构层面），最后训解码器精修。这种"先基础后展开"的训练顺序比端到端训练多轮迭代更稳定——DAI 的消融实验也证明了分阶段 > 一步到位。
> 
> **批注 E2 [2026.6]**：L-DiffER 的迭代条件精炼（ICR）与本方案的"多轮迭代 MCRE"在形式上有共通之处——都是利用当前输出来修正下一轮条件。区别在于：ICR 在 diffusion latent space 中操作（timestep 维度），而本方案在 MCRE forward pass 中操作（iteration 维度）。可考虑将 ICR 的时变插值系数（βt, γt）引入 MCRE 迭代：第 k 轮的条件 = γk · T_{k-1} + (1-γk) · I，早期迭代多依赖原图，后期多依赖上一轮输出。
> 
> **批注 E3 [2026.6，同组演进的启示]**：RDNet 的可逆单元本身就来自同组 DSIT 之后的新探索。该组在 DSIT（注意力交互）和 RDNet（可逆交互）之间选择了后者作为最新方向。但可逆更新公式 `T₂ = T₁ + F(R₁)` 的信息传递是**单向且固定的**——没有 DSIT DS-CA 那种"自适应评估对方需要什么信息"的能力。一个可能的 hybrid 方案：在可逆单元的 F(·) 中加一个轻量通道注意力门控，让网络自己学习"从 R 列给 T 列传多少信息"——保留可逆结构的稳定性，同时引入注意力的自适应性。改动量约 30-50 行。

---

### F. 条件细节精修模块（Conditional Detail Refinement Block）

**思路：** 在 MCRE + HDec 之后加一个轻量级精修网络（≤ 5M 参数），以 α(x) map 为条件做空间自适应增强。不同于 RRay 的 NAFNet 级联（NAFNet 是不分区域的全局处理），CDRB 只在高衰退区域激活更多计算。

**实现：**
```
T_raw ← HDec(MCRE(I))

# 区域条件门控
gate = σ(conv(α(x)))             ← soft region selection
T_enhance = conv(T_raw)          ← 细节增强支路
T_refine = (1-gate)·T_raw + gate·T_enhance
```

- 类似 Per-Pixel Adaptive Normalization [SPADE] 的思路，但用更轻量的 gating
- **整体参数估计**：~3-5M, ≈ NAFNet 的 10-15%

**参考文献：**
- RRay RdNafNet [NTIRE 2026, 1st] — 级联设计验证，提供了 baseline 对比 [link](https://github.com/caijie0620/OpenRR-5k)
- NAFNet [Chen et al., ECCV 2022] — Simple Baselines for Image Restoration 中的奠基性工作 [link](https://arxiv.org/abs/2204.07396)
- Spatial Feature Transform [Wang et al., NeurIPS 2018] — conditional feature modulation 的经典方法 [link](https://arxiv.org/abs/1806.00830)
- SPADE [Park et al., CVPR 2019] — 空间自适应归一化，适合 spatial condition 的调制 [link](https://arxiv.org/abs/1903.07291)

---

### G. 扩散先验注入——Lightweight Diffusion Prior Adapter

**思路：** 在 MCRE bottleneck（Level-4, End Level）附加扩散先验分支，专门负责 α(x) 低值区域的语义修复。参考 Xreflect Master 的蒸馏 + ControlNet 的 condition adapter 范式，但植入 RDNet 内部而非外部级联。

**实现：**
```
Level-4 feature f₄
    ├→ 常规通路 → HDec → T_raw
    └→ diffusion prior adapter (conditioned on f₄)
            └→ one-step diffusion → inpainting residual → skip connection 回 HDec
```

- 扩散分支训练时只更新 adapter（如 ControlNet 的 zero convolution），扩散 backbone 冻结
- 推理时**按需激活**——仅 α(x) < 阈值（如 0.2）的区域触发扩散采样，其他区域直通
- 这控制了推理成本（全图跑扩散在 Table 3 里看到是昂贵的，如 ACVLAB 的 528s 单张）

> **批注 G1 [2026.6，重要更新]**：DAI [AAAI 2026] 证明**单步扩散（one-step diffusion）在 SIRR 中完全可行**，且效果优于 50 步 DDIM 的 L-DiffER。这意味着之前的"扩散 = 慢"假设需要修正。具体而言：
> - 原方案中的 DDIM sampling 可替换为 DAI 的单步去噪公式 `z₀ = (zt - √(1-ᾱt) · ϵθ) / √ᾱt`，推理从 ~50 步 → 1 步，速度提升 50×
> - DAI 在 RTX 3090 上 768 分辨率约 1 秒/图——若仅对 α(x) < 0.2 的 masked 区域运行，总延迟可能在 200-400ms 级别，实用性大幅提升
> - **架构适配建议**：DAI 使用 Stable Diffusion v2.1 的 U-Net + ControlNet 作为去噪网络。若要植入 RDNet，可将 RDNet 的 MCRE 输出（f₄）作为 ControlNet 的条件输入（类似 DAI 用混合图像 M 作为 ControlNet 条件），而非从零设计扩散分支
> 
> **批注 G2 [2026.6]**：L-DiffER 的 ICR（迭代条件精炼）也可用于改进扩散 adapter 的条件输入——在采样过程中，条件从"MCRE 粗输出"逐渐过渡到"扩散自身估计的伪透射层"，解决 MCRE 在极低 α 区域输出不可靠的问题。
> 
> **批注 G3 [2026.6]**：DAI 的 Cross-Latent Decoder 设计表明，扩散模型的 vanilla decoder 容易产生模糊输出。若在扩散 adapter 中也加入 skip connection（从 f₄ 到 decoder 各层），预期能显著改善扩散分支的细节保真度。

**参考文献：**
- Xreflect Master [NTIRE 2026, 2nd] — 扩散蒸馏 + VAE 对齐，大模型的蒸馏先验 [link](https://github.com/BrewieRain/MS-RDNet)
- ControlNet [Zhang et al., ICCV 2023] — Adapter 范式，可用作注入方式的模板 [link](https://arxiv.org/abs/2302.05543)
- L-DiffER [Hong et al., ECCV 2024] — 将扩散模型用于 SIRR 的早期工作 [link](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/13057_ECCV_2024_paper.php)
- DAI [Hu et al., AAAI 2026] — "Dereflection Any Image with Diffusion Priors"，SIRR 领域最新扩散模型方法 [link](https://ojs.aaai.org/index.php/AAAI/article/view/32958)
- WindowSeat [Yamashita et al., CVPR 2024] — 扩散反射去除的另一代表 [link](https://openaccess.thecvf.com/content/CVPR2024/html/Yamashita_WindowSeat_Reflection_Removal_with_Diffusion_Priors_CVPR_2024_paper.html)

---

> **批注 G4 [2026.9，定位修正]**：按文首路线定版，G 降级为背景/探索项。若未来重启，2026 有两个低成本入口：(a) GenSIRR 的 **DEBS**——深度引导的免训练多候选选路，先当"无参考质量筛选"工具用（只需单目深度网络，可对现有模型输出做车窗评测）；(b) 以 DAI/GenSIRR 官方权重**离线**为车窗难样本生成 pseudo-GT 后蒸馏入轻量 student（不动 12B 级训练，8–16GB 卡可执行）。

### H. Checkpoint Soup + SWA —— 低成本集成

**思路：** 训练结束后，无需改变架构，通过权重平均获得一致性的免费增益。参考 SiGMoid 和 YuFans 在 NTIRE 中的实践。

**实现：**
- **SWA (Stochastic Weight Averaging)**：训练最后 N 个 epoch，每个 epoch 以固定学习率循环，取 average weight
- **Checkpoint Soup**：选取 validation 最佳的几个 checkpoint + 稳定末期的快照，等权平均
- 可叠加 TTA（几何翻转）和提分

**参考文献：**
- SWA [Izmailov et al., UAI 2018] — Stochastic Weight Averaging，理论保证找到更宽的最优点 [link](https://arxiv.org/abs/1803.05407)
- Model Soups [Wortsman et al., ICML 2022] — 独立微调多个 checkpoint 后权重平均，不减推理速度 [link](https://arxiv.org/abs/2203.05482)
- SiGMoid [NTIRE 2026, 8th] — 验证了 checkpoint soup 在 SIRR 中的实用性 [link](https://github.com/caijie0620/OpenRR-5k)
- YuFans [NTIRE 2026, 5th] — Top-5 checkpoints SWA + geometric TTA，仅 OpenRR-5k 获第 5 [link](https://github.com/caijie0620/OpenRR-5k)

---

### I. Masked Diffusion Autoencoding 预训练

**思路：** 在贴膜玻璃专用数据集有限时，先用大规模通用数据预训练模型基础能力，再在目标域上微调。NTR (TimeDiffiT) 的 MDAE 策略——空间 masking + VE-SDE noise injection + Corruption2Self loss——可以适配到 RDNet/MCRE 的预训练。

**实现：**
- 将 MCRE 的 encoder 部分用作 MDAE 训练的 backbone
- 在 ImageNet / Places2 等大规模数据上用 MDAE loss 预训练
- 预训练好的 encoder 参数用于初始化 MCRE 各列的 encoder
- 再在贴膜玻璃数据上全模型微调

**参考文献：**
- MAE [He et al., CVPR 2022] — Masked Autoencoding 的原始工作 [link](https://arxiv.org/abs/2111.06377)
- TimeDiffiT / NTR [NTIRE 2026, 9th] — MDAE + D2S-SFT 的双阶段策略 [link](https://github.com/caijie0620/OpenRR-5k)
- Score-based Self-supervised MRI Denoising [Tu et al., 2025] — Corruption2Self loss 的来源，TimeDiffiT 的预训练框架 [link](https://arxiv.org/abs/2505.05631)

---

### J. 贴膜玻璃专用数据采集 Pipeline（受 DRR 启发）

**思路：** DAI [AAAI 2026] 的 DRR 数据集通过**旋转玻璃角度**高效获得"同一场景 + 多种反射"的图像对。这种采集方式天然适用于贴膜玻璃场景——只需将普通玻璃替换为贴膜玻璃，即可采集"同一透射场景 + 不同贴膜反射"的训练数据。

**实现方案：**
- **设备**：固定相机 + 固定场景，玻璃架可旋转（0°/15°/30°/45° 等角度）
- **贴膜类型**：覆盖不同透光率（5%/15%/30%/50%）和不同贴膜颜色（深灰/镜面/蓝紫/渐变）
- **光照条件**：室内灯光 / 窗外自然光 / 混合光，每种光下拍摄不同角度
- **GT 采集**：移除贴膜玻璃直接拍摄透射场景（与 DRR 对齐流程一致：SIFT + RANSAC 对齐）
- **预计规模**：50-100 场景 × 3 贴膜类型 × 4 角度 × 3 光照 ≈ 1800-3600 对，远超现有贴膜玻璃数据的规模

**额外收益——支持 RIF 式一致性训练：**
- 同一透射场景 + 不同贴膜反射角度的多对数据可直接用于 DAI 式的反射不变微调（RIF）
- 损失：`L_RIF = ||f(M₁) - f(M₂)||₂`，其中 M₁/M₂ 是同一场景不同玻璃角度的混合图像
- 这强制网络学习透射场景的不变特征，对贴膜导致的强衰减场景特别有效

**参考文献：**
- DAI / DRR [Hu et al., AAAI 2026] — 数据采集 pipeline 原版，旋转普通玻璃 [link](https://ojs.aaai.org/index.php/AAAI/article/view/32958)
- RRW [CVPR 2024] — 已有最大规模 SIRR 真实数据集，可作对比 baseline [link](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Single_Image_Reflection_Removal_With_Reflection_Robust_Temporal_Consistency_CVPR_2024_paper.html)

> **批注 J1 [2026.6]**：该数据采集方案成本低（仅需可旋转玻璃架 + 不同贴膜），采集速度快（固定机位免去重复对焦），是本文所有方案中"投入产出比"最高的基础设施工作。建议在 Phase 1 就启动数据采集，因为 A/B/D/H 等方案都需要贴膜玻璃数据做验证——目前贴膜玻璃场景缺乏标准 benchmark。

---

## 推荐实施路线

### 2026.9 主线重排（定版后）

- **第一优先：FullTriAxisNet 端到端联合模型迭代**——真实验证平台为 6 月 3000 对四步法数据 + 车窗评测集；本文件以下 RDNet 改进项均以"可作为联合模型组件/训练技巧"为前提，不单独刷通用 SIRR 榜。
- **第二优先（Phase 1 内）**：K（DSIT 小机制）、D（FFL）、H（SWA/Soup）——改动浅、收益稳，直接落进联合模型训练管线；J 主体（四步法采集）已落地，RIF 式多角度扩展按需启用。
- **按需组件化（Phase 2 内）**：C（DINOv2 adapter，有 GFRRN 先例背书）、E（迭代级联）、F（条件精修）。
- **降级/暂停**：G、I（见文首路线定版）；压缩/端侧整体搁置（见文首复活条件）。

### Phase 1（最短路径，快速出效果）
```
J (贴膜数据采集) + K (DSIT 小机制) + A (Spatial TAPG) + B (物理先验注入) + D (FFL + 空间加权) + H (SWA + Soup)
```
- J 优先启动——数据是一切验证的基础，采集成本低、可并行进行
- 改动 RDNet 约 300-400 行代码，不引入新网络
- B 完全零训练成本，A 和 D 改动浅
- 预期收益：贴膜玻璃场景 PSNR +1~2 dB
- **2026.6 更新**：若 J 采集时覆盖同一场景多角度（仿 DRR），可在 Phase 1 就直接引入 DAI 式的 RIF 一致性 loss，收益可能更大

### Phase 2（扩展到架构层面）
```
C (DINOv2 Adapter) + F (Conditional Detail Refinement) + E (可逆级联)
```
- 逐步增加复杂度，每个模块可独立验证/合并

### Phase 3（高成本探索）
```
G (扩散先验注入，改用单步扩散) + I (MDAE 预训练)
```
- 需要额外硬件和数据资源
- **2026.6 更新**：DAI 的单步扩散证明了扩散路线在 SIRR 中可以是实时的（~1s），Phase 3 的 G 方案应将原 DDIM 采样改为单步扩散，大幅降低计算成本
- 潜在突破点：单步扩散 + 按区域激活（仅 α(x) < 阈值）→ 实用化贴膜玻璃反射去除系统

---

## 关键实验设计

### 验证 A（Spatial TAPG）
```
实验组: RDNet + spatial TAPG (global + α(x))
对照组: RDNet (原始 TAPG)
数据集: 贴膜玻璃数据（5%/15%/30% 透光率 + 不同光照）
指标: PSNR↑, α(x) 图可视化（预期亮点在贴膜划痕处）
```

### 验证 B（物理先验注入）
```
实验组: RDNet + 4 物理先验通道输入
消融组1: RDNet + blur_map 单独
消融组2: RDNet + ghost_map 单独
消融组3: RDNet + saturation_residual 单独
消融组4: RDNet + edge_ratio 单独
任务: 预期模糊度先验（blur_map）贡献最大，因为贴膜玻璃场景中焦距差异最可靠
```

### 验证 K（DSIT 遗留小机制）
```
实验组1: RDNet + RefMix（K1）
实验组2: RDNet + 梯度排斥 Loss（K2）
实验组3: RDNet + MVC Loss（K3）
实验组4: RDNet + Column Embedding（K4）
实验组5: RDNet + K1+K2+K3（全组合）
对照组: RDNet（原始）
预期: K1、K2 分别有 ~0.2-0.3dB 收益，K3 在极低 α 区域有肉眼可见的 artifact 抑制
```

### 验证 D（FFL + 空间加权）
```
实验组: RDNet + L1 + L_per + L_FFL + η·(1-α(x))·L_per
对照组: RDNet + L1 + L_per  (RDNet 原始损失)
观察: 预期 FFL 对 ghosting artifact 抑制最明显
```

---

## 新增 baseline 对比（2026.6 更新）

贴膜玻璃场景的相关方法已快速演进，实验设计时需将以下方法纳入对比：

| 方法 | 会议 | 为什么需要对比 |
|------|------|---------------|
| **DAI** [Hu et al.] | AAAI 2026 | 当前 SIRR 最强 SOTA，单步扩散路线，Nature PSNR 27.05 |
| **L-DiffER** [Hong et al.] | ECCV 2024 | 首个扩散 SIRR 方法，语言引导路线，可作为扩散路线的 baseline |
| **DSIT** | NeurIPS 2024 | DAI 论文中最强的非扩散 baseline（Nature 26.25 dB） |
| **RDNet** [Zhao et al.] | CVPR 2025 | 本方案的 backbone，提供改进的起点 |
| **RRay** | NTIRE 2026 | RDNet + NAFNet 级联，NTIRE 2026 第 1 名 |
| **GenSIRR** [Li et al.] | CVPR 2026 | 现通用 SIRR 最强（Nature 27.30+，天大郭组"判别式→扩散"转向作）——"通用方法上限"参照；HF 有权重，可离线跑车窗测试 |
| **GFRRN** [Chen et al.] | CVPR 2026 | 判别式新顶（五集均值 27.33/0.929），结构同 DSIT/RDNet 族 + PEFT 语义调优 —— 判别式对比的关键参照（本地 PDF 在手） |
| **WindowSeat** [Zakarin et al.] | CVPRW 2026 | DiT LoRA + PBR 合成路线；权重未开源，先记坐标 |
| EvReflection | ICML 2026 | 事件路线，与"单帧"约束冲突 —— 明确不进对比表 |

> **关键问题**：DAI 在标准 benchmark 上大幅领先（Nature +1.8 dB vs RDNet），但贴膜玻璃场景不在其测试范围。若本文改进的 RDNet 在贴膜玻璃场景上能显著超越 DAI，则证明"通用方法不够，专用改进必要"——这是 paper 的 story 支点。
>
> **2026.9 修正（路线定版后）**：通用基准坐标已更新（GenSIRR/GFRRN 均高于 DAI），DAI 降为"可运行扩散参照"；对比主战场改为**车窗评测集**（真实贴膜 + 低光耦合退化）。扩散类方法权重在手者用**官方权重推理**评测并注明来源，不在车窗数据上重训；story 支点从"超 DAI"改为"超通用方法（判别式 GFRRN / 扩散 DAI·GenSIRR）在车窗耦合退化上的表现"。
