# DARL-RDNet: 延迟归属的可逆层分离

## 方法

DARL（Deferred Attribution via Reversible Lifting）在 RDNet 的四列可逆特征提取器与原双分支 Decoder 之间，使用唯一的一套跨列三状态归属更新。保留预训练 FocalNet-L、Mona、SubNet 和两条原 Decoder 参数，不建立另一套网络。

针对第 \(k\) 列、第 \(l\) 个尺度的 RDNet 特征 \(C_{k,l}\)，维护三个同形状特征 \(T_{k,l},R_{k,l},U_{k,l}\)；它们分别表示透射归属、反射归属、尚未归属的特征，不是三个独立物理图层。始终保持

\[
T_{k,l}+R_{k,l}+U_{k,l}=C_{k,l}.
\]

第一列初始化 \(T=R=0,U=C\)。进入下一列时，先把主干特征更新量添加到未决状态：

\[
U_{k,l}^{\mathrm{pre}}=U_{k-1,l}+C_{k,l}-C_{k-1,l}.
\]

其他两种状态保持不变，因此三状态总和始终等于当前主干特征。

### 条件可逆归属

对每个尺度，从固定的 \(C_{k,l}\) 计算归属概率

\[
(p_T,p_R,p_U)=\operatorname{softmax}(h_l(\operatorname{GN}(C_{k,l}))).
\]

共享尺度参数跨四列复用。用于重新分配信息的单步提升变换作用于任意两状态 \(X,Y\)：

\[
S=X+Y,\quad D=X-Y,\quad D'=\exp(s)\odot D+t,\quad
X'=(S+D')/2,\quad Y'=(S-D')/2.
\]

其中 \(s=p_X\cdot 0.5\tanh(s_l(C))\)，\(t=p_X\cdot t_l(C)\)；函数只依赖整个归属过程中不变的 \(C\)，不依赖正在更新的 \(D\)。先执行 \(T\leftrightarrow U\)，再执行 \(R\leftrightarrow U\)，以此实现可重分配的未决归属。

每步保持 \(X'+Y'=X+Y\)。由于 \(\exp(s)>0\)，知道 \(C\) 即可重算 \(s,t\)，逆映射为 \(D=\exp(-s)(D'-t)\)。两步的复合仍然可逆。这里证明的是特征分配过程在固定条件下的可逆性与状态总和守恒，并不保证从单张图像中唯一识别物理图层。

为复用原 Decoder，两条读出分别为

\[
C^T_{k,l}=U_{k,l}+T_{k,l},\qquad
C^R_{k,l}=U_{k,l}+R_{k,l}.
\]

它们允许尚未确定归属的信息暂时参与两路预测。原 Decoder 逐阶段输出 \(\hat R_k,\hat T_k\)，不新建解码头。所有归属门控和提升卷积权重零初始化，初始读出等于 \(C_{k,l}\)，与官方 RDNet 输出一致。归属门控为每尺度的 \(1\times1\) 卷积，提升参数为深度可分组 \(3\times3\) 卷积，附加计算对总特征元素数为线性量级。四尺度共享归属模块合计增加 41,292 个参数（通道维度 512/256/128/64）。

### 训练目标

训练使用真实拍摄反射输入 \(I\) 与其配对的无反射透射层 \(T^*\)。由于没有独立拍摄的反射物理层，以 \(I-T^*\) 作为反射残差代理。延用四列递增权重的 Charbonnier 透射损失、残差损失和最后一列的重建一致性损失。

为了避免第三状态退化为无语义的额外支路，引入可计算的弱归属监督：分别计算 \(T^*\) 与 \(I-T^*\) 的水平、垂直 RGB 梯度强度 \(e_T,e_R\)，建立三类空间软标签

\[
q_T=\frac{[e_T-e_R]_+}{e_T+e_R+\epsilon},\quad
q_R=\frac{[e_R-e_T]_+}{e_T+e_R+\epsilon},\quad
q_U=\frac{2\min(e_T,e_R)+\epsilon}{e_T+e_R+\epsilon}.
\]

这三个标签非负且总和为 1；它们表示局部纹理证据中的优势与不可区分部分，并不声称精确恢复了物理归属。对软标签按尺度平均池化，与最后一列归属预测做交叉熵。总目标

\[
\mathcal L = \mathcal L_T+\lambda_R\mathcal L_R+
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}+
\lambda_{\mathrm{attr}}\mathcal L_{\mathrm{attr}}.
\]

先前 CCDD 的跨列差分记忆和恢复反馈不保留。整条方法只有唯一的 DARL 归属实现与统一训练/评测入口。

## 数据与评测

真实配对训练集由 Hugging Face \`qiuzhangTiTi/OpenRR-5k\` 的官方训练 5,000 对与 \`zmy1234567890/tintedglass-rdt-synth\` 仓库的真实子集 713 对组成（\`rdt_1020\` 300 对、\`rdt_1019split\` 331 对、\`rdt_1026\` 82 对）。合并样本数 5,713 对，其中 Zhou 的数据来自 16 个采集场景，\`rdt_1026\` 含同一采集帧的重裁版本，不能按独立场景计数。严格排除该仓库的 \`rdt_0915\` 合成部分及 \`test/*\` 评测部分。全部数据在 HDD3 上按原路径引用，不合成图像、不复制出第二套图片。

联合训练 manifest：\`datasets/real_sirr_hf/openrr5k_plus_zhou_real/train/manifest.json\`。模型先在 OpenRR 5,000 对真实图像上训练至第 10,000 步，再从完整 checkpoint 连续恢复网络、优化器和 AMP scaler，并使用联合 manifest 完成余下步数，不重新初始化。
验证 manifest：\`datasets/real_sirr_hf/qiuzhangTiTi/OpenRR-5k/unpacked/validation_input/manifest.json\`。

训练器：\`scripts/finetune_rdnet_peft.py\`，教师模型构造唯一入口 \`build()\`；评测器：\`scripts/eval_arm_1019.py\`，统一以有重叠 Hann 窗的平铺推理处理 1024px 图像，避免与其他 GPU 作业争用显存。

主训练配置：Mona both、冻结主干、batch 1、Adam 初始学习率 \(10^{-4}\)、30,000 updates、224×224 训练裁剪、seed 101、透射与反射损失系数继承原训练器、归属损失系数 0.02。本次在 HDD3 使用 RTX 4090 GPU 3，batch 1、30,000 步、每 1,000 步保存 checkpoint，训练完成后使用相同评测入口和 overlap=64 的 384px 分块评估官方验证集。其余独立 seed 与无 DARL 的 RDNet+Mona 对照须保持同一数据、预算、评测协议。

S2R-Ref 实拍视频曾被前一版 CCDD 用于训练，因此不能将其作为未见过的官方对比基准；OpenRR 的官方验证集保持完全独立。主指标为 PSNR、SSIM，归属机制必须进一步通过匹配参数量的二流与三流非可逆门控对照、移除 U、移除归属监督、禁止跨列重分配等消融验证。

## 新颖性边界

RDNet 已有可逆多列编码；IBCLN 等方法已有迭代双层恢复；双流差分、特征互补、可逆仿射耦合均已有先例。DARL 候选研究贡献仅限于**未决状态的可逆归属与跨列修订如何提高真实反射去除成功率**，不把已有基本算子重新称为原创。论文新颖性及效果需要正式文献对照和同协议实测支持。
