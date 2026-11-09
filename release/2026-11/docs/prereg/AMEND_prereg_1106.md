---

## §11.9 修正案 A1–A4（2026-11-06 03:40 · Lead 03:35 批复；写于 base2/full2 尚无任何结果时）

> 写入时刻证据：base2_s101 训练至 it=5400/6000（尚无 ckpt_final、无表行）；base2_s202 it=1000；base2_s303 it=400；full2_s101 it=200；full2_s202 it=1；full2_s303/identm_* 未起。下表无 base2/full2 任何 PSNR 行。此后结果不得改动本节判据，只能追加结果。

### A1 provenance（回归门 FAIL ⇒ 情形 B；历史臂仅参照，不进判决）

- **历史 base 臂启动证据** `/tmp/attrib.sh`（2026-11-01 01:33 起，`runs/attrib1031/oldpool_s101`）：
  ```
  CUDA_VISIBLE_DEVICES=6 $PY -u scripts/finetune_rdnet_s2.py \
    --arm main --iters 6000 --batch 4 --lr 1e-4 --mona both --seg-iters 2000 \
    --save-every 6000 --log-every 200 --s-jitter 0.0 --seed 101 --gpu 0 \
    --omega-dir .../real_pairs_1020 --omega-domains may,jun \
    --lam-om 3.0 --lam-hinge 1.0 --lam-grad 5.0 --lam-lap 2.0 --out runs/attrib1031/oldpool_s101
  ```
  脚本 `scripts/finetune_rdnet_s2.py` sha256 `e0ec30dfcac839b500294fd1fa1dfded4e6c21167552b6f1aa86e495a832b7ac`；文件 mtime **2026-10-31 21:54**（早于历史运行，此后未改）⇒ 历史 base 臂所用脚本内容 = 当前工作树内容。
- **历史 full 臂启动证据** `/tmp/quiet_launch.sh`（`runs/attrib1031/quiet_s101`）：同一命令行 + `--lam-quiet 1.0 --quiet-tau 0.05`，脚本 `scripts/finetune_rdnet_s2_quiet_1031.py`。该脚本 mtime **2026-11-05 22:11**（晚于历史运行）⇒ 历史 run 用的是 **patch 前版本**，其 sha256 已不可考；可考的只有当前 patch 后版本 sha256 `c10e24f466ac4a5e89a7fee30ef432e9f0f8ca7bd5dd78179c53c5e94e30e86a`。
- **回归门（§6.3）**：`regr_full_s101 = 23.079` vs 22.709（Δ **+0.369 > 0.26**）；`regr_base_s101 = 22.999` vs 21.990（Δ **+1.009**）⇒ **FAIL** ⇒ 情形 B。
- **最高优先级核查（2026-11-06 03:30–03:33，回应 Lead 03:28 追问；GPU2 ≈150 s）** —— 用**当前评测管线**重评历史 ckpt：
  - 命令：`CUDA_VISIBLE_DEVICES=2 $PY -u scripts/eval_arm_1019.py --ckpt <hist ckpt> --name recheck_hist{base,full}1106 --device cuda:0`，再 `scripts/table_final_1019_annot_1031.py --pred-map /tmp/ri_dev/predmap_recheck1106.json --tag recheck1106 --hole-mask-dir data/annot_1019/omega_masks_1030`（15 场景 uint8，手标 Ω 口径）。
  - `runs/attrib1031/oldpool_s101/ckpt_final.pth`（sha256 `7281794c4480720609a5280f551d5ef1911b77005f339b2ac4d105781739c9de`，1602019889 B，iter=6000）⇒ **PSNR = 21.9903**（历史 21.99，逐位复现）。
  - `runs/attrib1031/quiet_s101/ckpt_final.pth`（sha256 `47c154110d535072b8e0f20090e000ce5cc6e84af3f70f624019e270d9dcb156`，iter=6000）⇒ **PSNR = 22.7093**（历史 22.71，逐位复现）。
  - 旧权威表 `eval/table_final_1019_r11annot.json` 中这两臂 n=15 mean 分别为 21.990 / 22.709，与历史数字一致；历史 pred 目录完好（16 项 = 15 场景 + `arm_info.json`）。
  - **旧表脚本 `scripts/table_final_1019.py`（v13 Otsu 洞口径）对同一批新预测重评：PSNR 亦为 21.9903 / 22.7093**（仅「洞区/非洞区」分区列不同：annot 口径 oldpool 20.91/22.11、quiet 22.28/22.76；Otsu 口径 21.54/22.06、20.90/22.97）⇒ **总 PSNR 差异不是评测口径造成的**。
  - ⇒ **结论：历史数字在其 ckpt 上完全可复现；21.99→23.00 的 ~1 dB 差异在训练侧，不在评测侧。**
  - 训练侧证据：两脚本命令行参数逐项相同（见 §11.10）；`train_log.csv` 轨迹 iter1 完全一致（total 3.56024），iter200 起出现 1e-3 级微差并逐级放大，iter6000 total = 2.05930（旧）/ 2.08040（新）。两 ckpt 均为 iter=6000 最终态。**历史 base 脚本内容可确认（与当前一致），历史 full 脚本版本不可考；因此 1 dB 的来源在「脚本版本差 + 运行级非确定性」之间尚未分离。**
  - **分离实验（已在跑，无需额外成本）**：`base2_s101` 与 `regr_base_s101` 为**同脚本、同参数、同 seed 101** 的两次独立运行；`full2_s101` 与 `regr_full_s101` 同理。二者 PSNR 差 = run-to-run 漂移的直接估计，写入判决 memo §A1。
- **最终判决只用统一脚本臂**：base2/full2/identity/identity_m/hard/shuffled 全部由当前 `scripts/finetune_rdnet_s2_quiet_1031.py`（sha256 `c10e24f4…`，mtime 2026-11-05 22:11:48）训练；配方 args 逐字段一致（§11.10）。历史 `oldpool_*/quiet_*/quiet5_*`（旧脚本）仅作参照，不进判决。

### A2 fairness（identity 主对照改为 λ 幅值匹配）

- primary contrast 仍 **full2 − identity**（§4.1 不变）；但 identity 的**主对照改为 identity_m（λ=0.085065，loss 幅值匹配）**，λ=0.409194 版作 sensitivity 同报。
- 理由：λ_id=0.409194 由 E[w]=0.409194（τ=0.05，400 样本欧式域校准）得到，只匹配**平均权重**（λ·E[w]=1）；loss 幅值匹配应为 λ = E[w·|T̂−I|]/E|T̂−I| = 0.0048075/0.0565165 = **0.085065**。λ=0.409 使 identity 臂正则项幅值约为 quiet 项的 4.8 倍 ⇒ 对 identity 有利、对 full 不利，不能作为公平对照。
- 时间戳：2026-11-06 02:15 exp-dev 提出（send_message），03:35 Lead 书面批准。
- 判决要求：**full2−identity_m 若 95% CI 覆盖 0 或 |Δ|<MDE(0.26) ⇒ 不得写「显著优于 identity」**；full2−identity（λ=0.409）显著但 identity_m 不显著 ⇒ 按 §4.1 降级 J2。

### A3 full2 − base2（必须同报）

- 必须同报 Δ、95% CI、胜出场景数、配对 sd、MDE。按 §4.2 为**描述性**证据，**不作新颖性证据**。
- 它是论文「+0.50 dB」声明的**唯一公平检验**；若 CI 覆盖 0 或 <MDE，由 Lead 决定叙事处置（exp-dev 不自行改写论文文件）。

### A4 失败/重跑臂记录

- 无崩溃/失败臂。重跑/追加原因：`regr_base_s101`、`regr_full_s101`：provenance 回归门 FAIL（仅作参照）；`base2_s101/s202/s303`、`full2_s101/s202/s303`：情形 B 统一脚本重跑（进判决）；`identity_m_s101/s202/s303`：λ 幅值匹配公平性对照（A2）；`identity/hard/shuffled_s101/s202/s303`：原设计五臂之三（进 J1 secondary）；历史 `oldpool_s101/s202/s303`、`quiet_s101/s202/s303`、`quiet5_*`：旧脚本，仅参照。
- 冻结参数：τ=0.05 全臂固定；λ_h=0.645802（E[w]/E[1]）；shuffled λ=1.0（同 quiet），置换 Generator seed=2026；identity_m λ=0.085065。
- 运行级重复性：base2_s101 / regr_base_s101（同脚本同参数同 seed 的两次独立运行）PSNR 差 = run-to-run 漂移；写入 memo §A1。

### §11.10 provenance 清单（manifest 复核，2026-11-06 03:35）

- 当前脚本 sha256：`c10e24f466ac4a5e89a7fee30ef432e9f0f8ca7bd5dd78179c53c5e94e30e86a`（`scripts/finetune_rdnet_s2_quiet_1031.py`）；旧脚本 sha256：`e0ec30dfcac839b500294fd1fa1dfded4e6c21167552b6f1aa86e495a832b7ac`（`scripts/finetune_rdnet_s2.py`）。
- `runs/quiet_abl_1106/*/manifest.json` 的 args 逐字段对比：base2_s101 vs full2_s101 **仅 `manifest`/`out`（逐臂路径）、`reg_mode`、`reg_scale` 四个键不同**；其余配方键（arm/iters/batch/lr/mona/seg_iters/save_every/log_every/s_jitter/omega_dir/omega_domains/lam_om/lam_hinge/lam_grad/lam_lap/lam_quiet/quiet_tau）完全一致。seed 逐臂 101/202/303。
- 旧/新脚本 diff：82 行、6 处 hunk，全部为 reg 相关新增（`reg_loss` 函数、`--reg-mode/--reg-scale/--lam-hold` 参数、`L_reg` 日志列、manifest 的 `reg_mode/reg_scale` 字段）；数据增强/翻转行（第 269 行 `x[:, :, ::-1]`）两脚本逐字节相同 ⇒ **无 flip 修复差异**。
- 结论：新臂之间 provenance 可比；历史臂与新臂不可比（脚本版本不同），故判决只用新臂。

### §11.9b A1 补充证据（2026-11-06 03:55，仍在 base2/full2 出结果之前）

> 触发：Lead 03:41 追查「历史 11-01 quiet 臂用的 quiet loss 与今日 reg_loss(quiet) 是否同一实现」。结论：**同一实现，历史 quiet 脚本可逐字节重建；+1.0 dB 不是脚本版本问题，而是训练运行级不可复现**。

- **历史 quiet 脚本可重建**：`scripts/finetune_rdnet_s2_hold_1031.py`（sha256 `9ad099cc7f7c5cd64b24f00e2208e0ed4848cd7f5fae54aa384a90df157da306`，mtime 2026-11-01 20:26）+ `/tmp/patch_quiet.py`（11-01 20:40 QUIET1031 补丁，SRC=hold_1031 → DST=quiet_1031）⇒ 重建脚本 sha256 `18500831cefe831e2e7fde4270ef0eda59b88dede06ba81f0f3c7921e4a42b67`，与 `/tmp/ri_dev/finetune_rdnet_s2_quiet_1031.py.bak_1106`（2026-11-06 reg-mode 改造前备份）**byte-identical**。
- **quiet loss 实现逐字一致**：历史 quiet 块 `need=(I-T).abs().mean(1,keepdim=True); wq=clamp(1-need/max(tau,1e-6),min=0); L=lam*(wq*(T_hat-I).abs()).mean()` 与今日 `reg_loss(mode='quiet')` 相同 ⇒ 11-05 的重构（hold→quiet→reg-mode）**未改 quiet 数学路径**。
- **历史 base 脚本内容确认**：`scripts/finetune_rdnet_s2.py` sha256 `e0ec30df…`，mtime 2026-10-31 21:54（早于 11-01 01:33 的历史运行，此后未改）。
- **环境/数据在 11-01 后未变**：torch 2.12.0+cu126、conda env 无更新（`conda-meta/history` mtime 2026-05-18）、`scripts/dataset_real_pairs.py` mtime 10-21、`data/train_sets/real_pairs_1020` 与 `real_pairs_1027jun` mtime 10-27。
- **数值不可复现（同配置跨日期）**：hist_base_s101 21.990 vs 新跑 regr_base_s101 22.998（Δ=+1.008）；hist_full_s101 22.709 vs 新跑 regr_full_s101 23.079（Δ=+0.370）⇒ 历史 +0.50 落在同配置重复性噪声（~0.4–1.0 dB）之内，**不可确立**。
- **A1 定稿表述（三句）**：① 历史 ckpt 今日重评 = 21.9903/22.7093，与历史表精确一致 ⇒ 评分口径无问题；② 两历史臂训练脚本内容均可重建、quiet loss 实现一致、环境/数据未变，但同配置独立重跑数值不可复现 ⇒ 训练侧 +1.0 dB 差异**归因不可判**（历史 full 的 22.71 同样不可数值复现）；③ 唯一公平口径 = 全部用统一脚本重跑（情形 B），历史 +0.50 只作「未经统一口径的旧观察」，不进入 claim。

