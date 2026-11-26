# CCDD-RDNet: 跨列差分解耦的单图像反射去除

## 方法

模型沿用 RDNet (CVPR 2025) 的 FocalNet-L 主干、四列可逆 SubNet、多尺度 Decoder，以及 FocalModulationBlock 内的 Mona 适配器。CCDD 使用连续可逆列的表征变化来分离透射和反射，而不是增加一个独立编码器。

第 k 列、第 l 个尺度的特征为 C[k,l]，列间变化 D[k,l] = C[k,l] - C[k-1,l]。输入为 I，上一列的恢复结果为 T[k-1] 和 R[k-1]。反馈是对 [I - T[k-1], R[k-1]] 停止梯度后的六通道图像。

Z[k,l] = a[l] * C[k,l] + b[l] * D[k,l] + W[l] Pool_l(feedback)
M[k,l] = M[k-1,l] + D[k,l] * tanh(Z[k,l])

a,b 是每尺度逐通道参数，W 是 6 -> C_l 的 1×1 投影。所有可逆列共享这组尺度参数。第一列的 M 为零，所有门控参数零初始化，初始所有阶段与原 RDNet 逐元素相同。

透射读出为 C+M，反射读出为 C-M。两组读出直接使用原 Decoder 的对应参数化解码路径，最终输出顺序为 [R_hat, T_hat]（前 3、后 3 通道）。不新增 Decoder 或修改可逆 SubNet。为控制训练显存及避免额外的跨列图像监督依赖，反馈信号停止梯度。

## 三项统一技术创新

- 跨列差分记忆：显式编码每一次可逆列对多尺度表征的修订。
- 互补双层读出：只使用一套累积变化表征，对 T 和 R 施加相反方向的特征修订。
- 逐阶段恢复反馈：上一列的反射预测及透射修正量影响下一列的差分选择。

新增 8,640 个可训练参数，理论新增运算复杂度为 O(K sum_l H_l W_l C_l)。主干 FocalNet、Mona、SubNet 及 Decoder 原始参数和运算形式不变。

## 论文定位

RDNet 具有多列可逆编码（CVPR 2025）；GFRRN 已经采用 Mona 参数高效适配（CVPR 2026）；ReflexSplit 已经研究双流差分注意力及跨尺度门控（CVPR 2026）。CCDD 的区别是跨可逆列的状态变化记忆，以及恢复结果反馈下的互补读出。不得把 Mona、双流或差分本身描述为首次提出。

## 真实数据训练协议

使用 Hugging Face 上 HigherHu/S2R-Bench 的 S2R-Ref 实拍配对视频，在 2026-11-26 完成下载。该数据是实拍玻璃反射输入和对应无反射透射层，不是合成反射图。每个视频以每秒一帧、960×540 尺寸等时间戳抽取输入与 GT，按视频标识 SHA-256 固定划分，50 个视频 273 对图像训练，10 个视频 56 对图像作内部验证。全部产物保存在 HDD3 的 `datasets/real_sirr_hf/s2r_real_frame_pairs` 中。训练监督为直接配对的真实透射层 T 和图像差值 I-T；后者是残差代理，不宣称为真实反射物理层的独立观测。

研究限制：S2R-Ref 原本是公开 benchmark，这些视频经用于训练后，不能再把同一视频作为官方独立 S2R 测试结果。内部验证必须按视频分组，公开泛化评价应使用无交叉的 OpenRR 数据。S2R-Ref 本身来源于 DRR 训练划分，未来与 DRR 合并时需要执行源场景去重。

当前小规模真实数据训练入口为 `scripts/finetune_rdnet_peft.py`，参数：`--train-manifest .../s2r_real_frame_pairs/train/manifest.json --mona both --gpu 3 --iters 2000 --batch 1 --lr 1e-4 --seed 101 --seg-iters 600 --lam-wlow 0`。训练从官方 RDNet 权重初始化，不复用此前合成数据 checkpoint。每 200 步保存 checkpoint。训练完成后使用同一模型构造入口，在 `validation/manifest.json` 的 10 个独立视频上评测 PSNR／SSIM。

OpenRR-5k 的完整 5000 对真实训练包仍在下载；完成后应采用其官方训练划分作为正式主训练数据，并保留官方验证集完整用于最终对比。此前的 `tintedglass_hf_rebuild_20261029` 为合成数据，不用于本协议。
