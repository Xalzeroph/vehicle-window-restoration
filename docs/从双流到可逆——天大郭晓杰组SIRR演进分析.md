# 从双流到可逆：天津大学郭晓杰组 SIRR 研究演进分析

> 2026-06-08 基于论文阅读 + GitHub 项目分析的整理

---

## 1. 论文阅读清单

今天阅读了以下论文：

| 论文 | 会议 | 一作 | 核心贡献 |
|------|------|------|---------|
| **L-DiffER** | ECCV 2024 | Hong YC (北大) | 首个语言引导扩散 SIRR 模型，ICR + MVC/SSC |
| **DSIT** | NeurIPS 2024 | Hu QM (天大) | 双流交叉注意力交互，Swin-L 全局先验 |
| **DAI / DRR** | AAAI 2026 | Hu JC (上交) | 单步扩散 + DRR 数据集 + RIF 一致性训练 |

---

## 2. 关键发现：YTMT → DSRNet → DSIT → RDNet → DIRS 全属于同一组

通过 GitHub (mingcv) 和论文作者核实，**天津大学郭晓杰组**（Qiming Hu 贯穿全部五篇）在 SIRR 方向的完整演进：

```
YTMT (NeurIPS 2021)     DSRNet (ICCV 2023)      DSIT (NeurIPS 2024)
Hu QM, Guo XJ           Hu QM, Guo XJ           Hu QM, Wang HN, Guo XJ
激活门控交互              MuGI + 非线性残差         DS-CA 注意力 + Swin-L
═══════════════════════════════════════════════════════════════════
                        RDNet (CVPR 2025)
                        Zhao H, Li MJ, Hu QM, Guo XJ
                        可逆列 (RevCol) + TAPG
                        ═══════════════════════
                        DIRS (最新)
                        Hu QM, Li MJ, Li YT, Guo XJ
                        LORS 统一物理模型 + 三种交互封装
```

### 2.1 Mingjia Li 的桥梁角色

Mingjia Li 同时出现在 RDNet 和 DIRS 的作者列表中，是组内连接架构工作和物理建模工作的关键人物。

### 2.2 各阶段的核心机制

| 版本 | 交互机制 | 物理模型 | 全局先验 | 组内评价 |
|------|---------|---------|---------|---------|
| YTMT | 激活值门控选择 | I = T + R | HyperColumn (VGG) | 首次引入双流交互 |
| DSRNet | MuGI 互相门控 | I = T + R + C(T,R) | Feature Pyramid | 交互效率提升 |
| DSIT | DS-CA 交叉注意力 | 同 DSRNet | 冻结 Swin-L | 显式建模层间相关性 |
| RDNet | 可逆列 (RevCol) | 渐变参数 α,β (TAPG) | ConvNext TAPG | 交互从显式→隐式 |
| DIRS | YTMT/MuGI/PAIR 三合一 | LORS: I=T+R+Φ+Ψ | — | 框架统一 |

---

## 3. DSIT → RDNet：同组为何放弃注意力交互？

DSIT 在定量上很强（Nature 26.77dB），但该组在随后的 RDNet 中**几乎完全放弃了 DSIT 的设计**：

| DSIT 设计 | RDNet 中的命运 | 可能的放弃原因 |
|-----------|---------------|---------------|
| DS-CA 交叉注意力 | **替换为可逆列** | 注意力训练不稳定、计算量大、对 batch size 敏感 |
| Swin-L 全局先验 | **降级为 ConvNext 6维 TAPG** | 冻结大模型太重，推理成本高 |
| DSLP Block (MuGI) | **保留为列内部模块** | MuGI 局部交互被验证有效 |
| RefMix | **跳过（非否定）** | 可能被更复杂的数据策略替代 |
| 梯度排斥 Loss | **跳过（非否定）** | RDNet 选择在架构层面解耦而非 loss 层面 |
| LRPB 分层位置偏置 | **因可逆设计而不再需要** | 可逆公式天然区分 T/R 流 |

### 3.1 被 RDNet 跳过的 = 可以回补

DSIT 中验证有效的 RefMix (+0.34dB)、梯度排斥 loss、MVC 约束，在 RDNet 上**未被消融过**。它们很可能是 RDNet 的免费提分项——改动量 < 50 行，可直接加入 Phase 1。

---

## 4. DIRS：该组的"收网"之作

DIRS 是目前该组在 SIRR 方向的最新工作。核心是两个贡献：

### 4.1 LORS（Learnable Offset-Residual Superposition）

物理模型从 `I=T+R` → `I=T+R+C(T,R)` → `I=T+R+Φ(T,R)+Ψ`：

- **Φ**（非线性残差）：可学习，替代 DSRNet 中固定形式的 C(T,R)
- **Ψ**（零阶偏移）：描述全局色偏/亮度偏移

这对**贴膜玻璃场景**天然适配——贴膜引入的色偏 = Ψ，非线性衰减 = Φ。

### 4.2 三种交互机制的统一封装

| 变体 | 机制 | 对应前身 | 适用场景 |
|------|------|---------|---------|
| DIRS-YTMT | 激活选择性交换 | YTMT (2021) | 资源受限、轻量部署 |
| DIRS-MuGI | 互相门控 | DSRNet (2023) | 通用、鲁棒性好 |
| DIRS-PAIR | 双流联合注意力 | DSIT (2024) | 高难样本、追求极致质量 |

### 4.3 DIRS 对 RDNet 改进的启示

- **RDNet 的可逆交互可能是"第四种"**：比 MuGI 更结构化，比 PAIR 更轻量。可以在 DIRS 框架下做公平对比
- **LORS 物理模型可以直接引入 RDNet**：让 TAPG 输出不仅包含 α/β，还包含 Φ 和 Ψ 的参数
- **该组已不再争论"哪种交互最好"**——说明 RDNet 改进不应执着于发明新交互，而应针对贴膜玻璃的**特定物理退化**做定制

---

## 5. mingcv 的其他研究方向

通过 GitHub 主页分析，Qiming Hu 有三条研究线：

### 5.1 反射分离主线（5 个项目）
YTMT → DSRNet → DSIT → RDNet → DIRS，上文已详述。

### 5.2 低光增强：Bread (⭐ 111, IJCV)
将低光增强拆为三个子问题：
- IAN（光照调节网络）
- ANSN（自适应去噪网络）
- CAN（色彩调整网络）

**启示**：Bread 展示了"拆解 ill-posed 问题为可控子问题"的工程哲学。对贴膜玻璃场景，可以识别出独立子问题（色偏、α 空间衰减、贴膜纹理残留），各自用小模块处理。

### 5.3 文字感知超分：TADiSR (⭐ 20)
扩散模型 + 文字分割联合解码。信号：胡启明已开始做扩散模型在 low-level vision 中的应用，但用法克制（分割引导而非直接生成）。

---

## 6. 外部论文与本组工作的关系

### 6.1 DAI [AAAI 2026, 上交+华为]

| 维度 | DAI（上交） | 郭组路线 |
|------|-----------|---------|
| 交互机制 | 无显式交互（单流扩散） | 双流显式交互是核心 |
| 扩散步数 | 1 步 | DSIT 之后转向非扩散路线 |
| 物理模型 | I = T + R | I = T + R + Φ + Ψ (LORS) |
| 数据策略 | DRR 旋转玻璃 + RIF | 传统合成+真实数据混合 |
| Nature PSNR | **27.05** | DSIT 26.77, RDNet 25.94 |

DAI 在定量上全面领先，但两者的路线完全不同。**如果改进后的 RDNet 在贴膜玻璃场景上能超越 DAI，则证明"通用方法不够，场景专用改进必要"。**

### 6.2 L-DiffER [ECCV 2024, 北大]

- 语言引导 + 多步扩散 SIRR 的开创者
- 被 DAI 在速度（1步 vs 50步）和效果（27.05 vs 23.95 Nature）上全面超越
- 但 ICR 和 MVC/SSC 的设计思路是跨路线的——可作为训练技巧迁移

---

## 7. 对 RDNet 改进的核心启示

### 7.1 物理建模优先于架构创新
郭组的演进表明，每代物理模型的升级（I=T+R → +C(T,R) → +Φ+Ψ）都至少值一篇顶会。贴膜玻璃场景的物理模型（色偏 + 多表面 + 非均匀衰减 + 偏振相关）尚未被充分描述。

### 7.2 被跳过的不等于被否定
DSIT 中的 RefMix、梯度排斥 loss、MVC 约束在 RDNet 上未被消融——但 DSIT 验证了它们有效。这些都是 < 50 行改动的免费提分。

### 7.3 小模块协同 > 单一大模型
Bread 三个子网、DIRS 三种变体、DSRNet 三组件——该组的核心工程哲学是模块化拆解。改进 RDNet 不应是"加一个大模块"，而是"加几个针对性小模块"。

### 7.4 同组的最新注意力在物理模型上
DIRS 表明该组的关注点已从"如何交互"转向"如何建模"。跟进这个方向比跟进注意力 vs 可逆的争论更有价值。

---

## 8. 相关文件索引

| 文件 | 说明 |
|------|------|
| `ideas.md` | RDNet 改进方案汇总（含今日批注） |
| `L-DiffER_阅读笔记.md` | ECCV 2024，语言引导扩散 |
| `DSIT_阅读笔记.md` | NeurIPS 2024，双流注意力交互 |
| `Dereflection_Any_Image_阅读笔记.md` | AAAI 2026，单步扩散 + DRR |
| `Hu_Dereflection_Any_Image_with_Diffusion_Priors_and_Diversified_Data_AAAI_2026_paper.pdf` | DAI 原文 |
| `L-DiffER Single Image Reflection Removal.pdf` | L-DiffER 原文 |
| `Hu_Single_Image_Reflection_Separation_via_Dual-Stream_Interactive_Transformers_NeurIPS_2024_paper.pdf` | DSIT 原文 |
| `Zhao_Reversible_Decoupling_Network_for_Single_Image_Reflection_Removal_CVPR_2025_paper.pdf` | RDNet 原文 |
