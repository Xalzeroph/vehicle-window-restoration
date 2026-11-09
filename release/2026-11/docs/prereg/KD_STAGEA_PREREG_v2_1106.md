# 预登记：task-14 Stage A —— SFAM-Net v2 四臂 + Phase0 容量基线（2026-11-06，先于 C/D1/D2/D3 启动）

> 状态：**开跑前写定**（本文件生成时 SFAM v2 四臂均未启动、只有 20-it 冒烟）。启动后不再改判据。
> 背景：Lead m02512（2026-11-06 11:17）要求：先报 Phase0 A/A-2/B；启动 Stage A 前必须对齐 Astra 数学审阅
> `2026.11.06_Astra_数学审阅与伪代码_SFAM-Net.md`（11:01）；Stage A 改为 4 个 SFAM 臂 + A/A2/B。

## §1 运行矩阵（1 seed、12k it、每臂 1 卡并行）

| 臂 | 网络 | 监督 |
|---|---|---|
| A | plain NAFNet-26（11.339M，T̂=I+f(I)） | GT-only（Phase0 已跑） |
| A-2 | 同上，复跑 | GT-only（确定性验收；与 A 应 bit-identical） |
| B | 同上 | GT + 30% 真实 output-KD（Phase0 已跑） |
| C | **SFAM-Net v2**（width40，26 NAFBlock + 三层 Haar 频支 + 三尺度软门 + 有界增益头） | GT-only |
| D1 | SFAM-Net v2 | GT + output-KD（λ_o*=0.1，5% warmup） |
| D2 | SFAM-Net v2 | GT + 前级 feature-KD（λ_f*=0.01，10% warmup） |
| D3 | SFAM-Net v2 | GT + output-KD + feature-KD |

共同协议：12000 it（crop 224/256/320 各 4000）、bs4、AdamW lr 2e-4 wd0 betas(.9,.999) eps1e-8、
warmup 500 + cosine→2e-6、AMP fp16、grad clip 1.0、seed 101、det_seed=1106、data 确定性修复（同 Phase0）。
共享 init 副本 `/tmp/ri_dev/kd_sfam_init_v2_s101.pth`（四臂完全相同，各 run 目录另存 init_copy.pth）。
每 2000 it 存 ckpt；终评固定 12000 it（无 best-of）。

## §2 实现契约（Astra 11:01 审阅 + Lead m02512；与 v1 的差异）

1. **门控相加不相乘**：`G_j = 0.5 + σ(a_ch + a_sp + a_band)`，整门 ∈(0.5,1.5)。
   a_ch 来自 E_j 的 GAP-MLP；a_sp 来自四频带能量图 3×3 卷积；a_band 来自四频带能量 MLP（按频带重复到通道）。
   三者末层零初始化 ⇒ 初始 G≡1。γ_j（1×cX×1×1）零初始化，注入点在 E_j 侧。
2. **增益头**：`u = Head_g(L_3)`，`g = 1 + 0.2·tanh(1.0·Resize(u))`（α=0.2，C=1.0 常数，见 §5 偏差）。
   输出头 r（Head_r 末层）与 g 头末层均零初始化 ⇒ 初始 T̂=I。训练前向不硬裁剪；评测裁剪所有方法一致。
3. **主体**：F_j = Concat_b φ_jb(S_jb)，每带 c'_j/4 通道，c'=(40,80,160)；每级 2 NAFBlock。
   X_j = E_j + γ_j ⊙ Π_j(G_j ⊙ F_j)（Π_j: c'→cX）。Z_3=Bottleneck(X_3)，Z_{j-1}=Dec(Concat[Up Z_j, X_{j-1}])。
   输入 pad 到 8 倍数（reflect），输出裁回；Haar 用契约矩阵 A（AᵀA=I），只递归分解 LL。
4. **损失**：L_GT = L_char(ε=1e-3) + 0.2·L_SSIM(11×11 高斯 σ=1.5 valid) + 0.1·L_Haar
   （L3 + 3 层 LH/HL/HH = 10 子带等权）；L_KD-T = ⟨(T̂−sg T̂_t)²⟩；仅 real（meta.src==0，30%）。
   前级 KD：`L_KD-F = (1/3)Σ_j ⟨(A_j X_j − V_j)²⟩ / (sg⟨V_j²⟩+1e-6)`，
   V_j = Resize(sg Q_j, h_j, w_j)；Q_j = 教师 FocalNet 主干 x_base[0..2]（stride 2/4/8，pre-adapter），
   **不用 stage 4**；A_j 训练期可学 1×1，推理删除。教师冻结、stop-grad。
   λ_k(t)=λ*_k·min(1, t/(w_k·S))；λ_o*=0.1, w_o=0.05；λ_f*=0.01, w_f=0.10；S=12000。
5. **输出 KD 的教师目标**：复用 Phase0 已缓存的全分辨率教师伪标签
   `data/pools/kd_teacher_full2s101_1106/`（pair_hash b4eb0f81…，teacher sha acb2dd06…），按与 (I,T) 完全相同的
   draw plan/crop/flip 取块（与 Phase0 B 臂完全同源，保证 D1/D3 与 B 的可比性）；**不逐 iter 在线跑教师输出头**。
6. 教师加载：冻结 `runs/quiet_abl_1106/full2_s101/ckpt_final.pth`（sha acb2dd06…），mona=both 完整加载；
   教师参数量 321.16M，学生 SFAM 实测约 12.2–12.5M（以自检 json 为准）。

## §3 判据（写死；单 seed pilot，不作显著性证据）

- **主门**（Lead m02109/m02512）：D3−B 与 D3−A 的场景轴配对 Δ+95% CI；
  **观测门 = Δ ≥ +0.15 dB 且 CI 下界 > 0**；若 D3−B < +0.05 或 CI 跨 0 ⇒ **停结构主张、不追加 seed、正稿维持 Plan B**。
- **辅助（回答“KD 是否只是重复监督”）**：D1−C（output-KD）、D2−C（feature-KD）、D1−D2、D3−D1、D3−D2。
  不单独立门，但需与 D3−B 一并同报。
- **守门**：D3 相对 B 的 PSNR_Ω 与 HF-PSNR 降幅 ≤0.10 dB、SSIM 降幅 ≤0.002；β、动作比、LapVar 同报。
- **确定性门**：A 与 A-2 在 15-scene 面板上的 PSNR 差应 bit-identical（同一确定性数据流 + cudnn deterministic）；
  若 >0.2 dB，先修 RNG/算子在判，不得用当前数字下结论。
- 单 seed 的 MDE（历史配对 sd≈0.36，n=15）≈0.26 dB ⇒ +0.15 门低于 MDE，**只作工程继续/停止门**。

## §4 评测

15-scene `real_pair_1019`（development panel），annot_1031 手标 Ω，uint8 PNG 口径，
pred 目录 `runs/pred/real_pair_1019_kd0{_C,_D1,_D2,_D3}/`，五/七臂一张表 tag `kd0all`（含已跑的 A/A2/B）。
指标：PSNR、PSNR_Ω、PSNR_nonΩ、SSIM、HF-PSNR、β、动作比、LapVar；效率：params/FLOPs（320² b1）/延迟（FP32 b1/b4）/峰值显存。

## §5 已知偏差与风险（必须写进报告）

1. **C=1.0 常数**：契约公式的 C 未给数值；取 1.0。两输出头零初始化时 C 不影响起点，只影响 tanh 幅值尺度。
2. **wd=0**：沿用 Phase0 配方（数学审阅建议 1e-4）；为与 A/B 对比保持 wd0。
3. **A/B 的 GT 损失与 Stage A 不同**：Phase0 A/B 用 char + 0.1·grad（仅非真实）；Stage A 按契约用
   char + 0.2·SSIM + 0.1·Haar ⇒ **D−B 的信道含损失配方差异**，报告中必须披露；D−A 同样。
4. **输出 KD 用全分辨率缓存伪标签裁剪**（非教师看 crop 的在线输出）；与 Phase0 B 同源，跨臂一致。
5. 教师在与训练同池的 174 对近记忆化（teacher vs GT 25.96 dB），B−A / D3−B 结构性趋 0 的风险照 Phase0 §6 披露。
6. 学生 SFAM 参数量/FLOPs 若超设计预算（12.2–12.5M / 66–72G），如实报，不调宽度凑数。
   **实测 v2（2026-11-06 11:32，`/tmp/ri_dev/kd_sfam_test_v2_1106.json`）**：
   params **14,431,735（14.43M）**，FLOPs **80.11 GFLOPs@320² b1**，
   FP32 延迟 320² b1 **40.7 ms**（median 38.8）/ b4 **104.7 ms**（median 103.0），峰值显存 0.30 / 0.95 GB；
   Haar 往返误差 1.19e-7（<1e-6 验收门），常数图细节带为 0，初始 |T̂−I|=0.0（两输出头零初始化+γ=0 生效），门范围 [0.5,1.5]。
   ⇒ **超设计预算约 +15% 参数、+11–21% FLOPs**（设计估 11.55M 主干偏低）；Stage A 按实测数字报告，
   Stage B 再做参数匹配（Astra 设计已列宽度搜索 41/42 通道）。
7. GPU 共享机：只对 <1500MiB 空卡起跑；不抢他人卡；被抢占/中断的臂重跑并记录。

## §6 预算与时间

4 arms × 1 GPU × ~1.2–1.5 h ≈ 5–6 GPU·h（Stage A 预算 150 GPU·h 内）。
顺序纪律：Phase0 A/A-2/B 先 FINAL → 立即评测并报 Lead → 再启动 C/D1/D2/D3（本预登记先于这四臂）。

## §7 结果（待填；启动前不得填写）

- Phase0：待填
- Stage A：待填
- 判决：待填
