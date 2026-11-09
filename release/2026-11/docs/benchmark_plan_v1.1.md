# Benchmark v1.1 设计草案（两轨：A=ICME 现有数据 / B=ICCV 新采集；2026-11-08，规划性，不训练、不改论文）

0. **定位（按 OpenRR 反例改）**：本 benchmark 的卖点 = 「贴膜玻璃 + 弱反射 + 局部内容不可辨识」的组合，**不是**「弱反射只有我们失败」。证据：普通数据同模型对照中，OpenRR 弱反射上零样本 6 模型 ΔPSNR vs identity 全为负（0/6 Δ>0；identity 29.95 dB 本身很强），SIR2 混合（4/6 Δ>0）、Real20 强反射 5/6 Δ>0 ⇒ 弱反射本身不是贴膜独有；差异必须落在「贴膜装置的反射结构 + 局部内容不可恢复」上。不可辨识度（multi-GT 方差、buried 代理、同场景跨帧一致性）为**一等指标**。

1. **两轨与互不阻塞**
- **轨道 A（ICME 2026-12-30，不承诺新采集）**：只用现有 1019（户外强反射 15 场景）+ nov02（室内弱反射 15 sessions / 5 唯一内容，诊断用）；benchmark 部分只写**评测协议 / 诊断口径**（identity 平衡基线、ΔPSNR vs identity 配对、β/动作比/affine-oracle、buried 代理、无手标 Ω ⇒ 禁 PSNR_Ω、两口径与 cluster-aware 统计）。数据即现有冻结面板；不新增采集。
- **轨道 B（ICCV 2027，预计 2027-03 截稿）**：新采集最小 **40 场景（室内 20 / 户外 20）×≥3 GT**；节奏 4 周采集 + 2 周 QC + 2 周基线与写作。两轨独立：A 不等待 B，B 不因 A 延期而改变口径。

2. **数据 schema（每 pair 一行 JSONL；轨道 A 只填可得字段，B 全字段）**
- 键：`scene_id / sequence_id / pair_id / frame_idx / gt_idx / split`
- 采集：`date, site_id, indoor_outdoor, scene_type, illumination(lux 估计/白天-夜晚-逆光), weather`
- 相机：`camera_model, lens, focal_mm, exposure, iso, wb, device_id`
- 装置（新增）：`glass_size, film_model, film_batch, film_transmittance_est, tint_level, mounting, angle_of_incidence, dist_to_glass`
- 反射源（新增）：`reflection_type(车灯/行人/天空/室内灯/混合), reflector_type, reflection_strength(估计)`
- GT：`gt_type(黑布/暗屏/其他), n_gt_per_scene, gt_frame_distance`
- 锁定（新增）：`focus_locked, wb_locked, exposure_locked`
- 光度/几何：`g=median(I/T), shift_px, alignment_method, qc_status, qc_flags`
- 不可辨识：`bury_frac_local(§7.9.8 代理), identifiability_proxy(同场景多 GT 方差), kframe_std(同场景跨帧)`
- 隐私/许可（新增）：`privacy_regions(人脸/车牌/住宅 bbox), privacy_flag, license, consent`
- 划分：`scene-level split`（训练/验证/测试同场景不得跨 split）

3. **采集协议（轨道 B；SOP 另见 1 页中文版）**
- 每场景 **≥3 张 GT**（不同 I 帧配对），用于不可辨识度统计（nov02 现为 1 张/场景，已知限制）
- 同场景多光照（白天/夜晚/逆光）；室内外成对（同装置）
- 多膜/多装置分层：≥3 种膜型（深浅）× ≥3 装置位置
- 每场景 ≥100 帧；GT 与输入帧间隔 ≥30 帧；黑布 GT 为主，暗屏 GT 子集
- 现场 QC：`shift≤2px`、`|g−1|≤0.15`、无过曝/欠曝、反射可见；锁定曝光/白平衡/对焦后整段拍摄
- 元数据当日填写；隐私像素模糊化并逐帧登记 `privacy_regions`

4. **评测协议（两轨共用）**
- 主指标：全图 PSNR/SSIM/HF-PSNR/LapVar；光度轴 β、动作比、identity / affine-oracle PSNR；局部可恢复性（§7.9.8 buried 面积与带通残差）；**identity 平衡基线必报**
- 禁止：PSNR_Ω（无手标 Ω 不报）、按模型挑帧/挑 GT、跨 split 调参
- 统计：场景级配对 `ΔPSNR vs identity`（95% CI、MDE、胜场/场景数）；多 GT 场景报 multi-GT 方差；cluster-aware（内容近重复用 union-find 冻结簇图，effective n 披露）
- 不可辨识度（一等指标）：`multi_gt_var`、`kframe_std`、`buried_frac`、`best_of_models_gap`、`affine_oracle_gap`
- **公平性规则**：所有 baseline 同调参预算（报告写明 GPU·h 与迭代数）；hidden test 不公开 label；leaderboard 单模型单提交；公开训练/微调配方与划分
- 报告模板：主表 + 两口径（全场景 / 排除 dev）+ 不可辨识度小节 + 效率（GMAC/延迟/显存）

5. **规模建议**
- 轨道 A（ICME）：现有 1019 + nov02（诊断）+ 6 基线；不承诺新数据
- ICCV 2027（轨道 B 最小）：≥40 场景（室内 20/户外 20）、≥3 GT/场景、≥3 膜型、≥5 光照组合、10+ 基线
- NeurIPS 2027 D&B（远期）：≥100 场景、多装置/多膜批次、公开 train/test、完整元数据与 QC 报告

6. **复用与重采**
- 直接复用：1019 15 场景（户外强反射，主评测已冻结）；nov02 15 sessions（室内弱反射，诊断）；评测管线/指标代码/基线封装（`eval_arm_1019`、`table_final_1019`、affine-oracle、§7.9 分析脚本）
- 需重采/补采：每场景多 GT（现有 1 张）；多光照/多膜/夜晚；元数据补全（lens/film/illum/angle/dist）；隐私标注；训练集扩展
- 规则：dev 4 场景保留内部校验；测试 label 不公开；A 轨可复用 B 轨产出，但 B 轨不反向依赖 A 轨

7. **采集排期模板（轨道 B，4 周采集 + 2 周 QC + 2 周基线与写作）**
- W1：装置/膜型定型、SOP、元数据与隐私模板、试点 10 场景
- W2–W3：室内 20 场景 + 户外 20 场景（每场景 ≥3 GT、多光照）
- W4：补采与现场复核；W5–W6：QC、GT 复核、元数据补齐、schema 冻结；W7–W8：6–10 基线跑通 + 报告/论文写作
- 每周验收：场景数 / QC 通过率 / 每场景 GT 数 / 元数据完整率

8. **风险与声明**
- 弱反射占多数（nov02 证据）⇒ 必须按强/弱分层报告；OpenRR 反例说明弱反射不是贴膜独有，主张只限定「贴膜+弱反射+局部不可辨识」组合
- GT 是黑布图像而非物理真 T，T 与照度不可完全分离 ⇒ 主张限定
- 单帧信息不足是任务属性（§7.9.15 / §7.9.8 证据）而非模型缺陷，benchmark 把不可辨识度作为一等指标
