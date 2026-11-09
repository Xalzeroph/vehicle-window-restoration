# KD_PROBES_PREREG_1106 — Stage B-mini 之后三探针预登记 addendum

- 时间：2026-11-06 17:35（写于 P1 任何训练启动之前；此时 P1 池/cache 尚未生成、无 P1 ckpt）
- 依据：Lead m04047（三探针）与 Lead m04096（P1 定级为「多帧监督」、P2 等 P1 结果再定、P3 挂起）
- 状态：**P1 立即执行**；P2/P3 挂起（P2 等 P1 出数 + Lead 指示；P3 等 Stage B-main 三种子出数且 D2 24k 均值 ≥20.3）

## §1 P1：数据扩容探针（定级：同场景多帧监督，不是多场景）

### §1.1 池与唯一内容（判读口径必须写在报告开头）
- may130：`data/train_sets/real_pairs_1020` 中 domain=may 且 I 存在，共 130 对 / 5 场景。
- jun1093：`data/train_sets/real_pairs_1027jun` 全部 1093 对 / 13 场景。
- **排除**旧 `real_pairs_1020` 的 jun44（2 场景）——Lead m04096 指定。
- 合并池 **1223 对**，但**唯一内容仅 18 种**（may 5 + jun 13；`scripts/check_t_src_leak.py` 池去重统计）。⇒ P1 的期望增益上限≈同场景多帧正则化，**不是数据多样性**；报告不得写成「更多数据」。
- 泄漏核查（Lead m04006 前序要求）：metadata 层与 1019 面板无场景名/VID 重叠；内容级 `check_t_src_leak.py` 两池均 rc=0（字节 md5、缩放派生 MSE、感知近重复 r>0.9 均 0/15；裁剪级 max r=0.735/0.754 < 阈值）。结论照写「未发现泄漏（但池有内容重复，不可按文件名做排除/留出）」。
- 池构建：symlink 合并 I/T/R/omega + 合并 meta.json（keys 已断言不冲突）；排序 case 列表冻结 → `pair_list.json` + `pair_hash`（sha256 of sorted cases）写入 merged meta 与 manifest。
- teacher 输出标签：full2_s101 `runs/quiet_abl_1106/full2_s101/ckpt_final.pth`（sha256 acb2dd06…），全分辨率 uint8 RGB PNG，与训练同预处理（I/T/标签同 crop + 同 flip）；**断点续跑**、每文件 sha256 清单、缓存 meta 记 teacher sha/pair_hash/文件数/字节数/耗时。
  - 注：P1a 训练配方=D2=**feature-KD only**（输出标签不进 D2 损失）；cache 仍按 Lead m04096 生成，供 P2/P3 与可选副臂 P1b 使用。

### §1.2 训练臂（单变量 = 池）
- **P1a（主）= D2 配方 + 扩容池**：`kd_sfam_train_v2_1106.py --arm D2 --iters 12000 --batch 4 --seed 101 --det-seed 1106 --init /tmp/ri_dev/kd_sfam_init_v2_s101.pth --pool <merged pool> --domains may,jun --pseudo-dir <merged teacher cache>`；crop 224/256/320 各 4000、AdamW 2e-4 wd0、warmup500+cos→2e-6、AMP clip1、ckpt 每 2000；与 D2 唯一差别=池（外加 D2 无输出 KD，故 pseudo-dir 只作 manifest 记录）。
- **P1b（可选副臂，卡空余才跑）**：同池同 seed，`--arm D3`（feature+output KD，output 标签用上述 cache），回答「扩容池上 output-KD 是否增益」；不作主判据。
- 对照：**D2 12k = 20.1663**（15-scene uint8 annot_1031，`eval/table_final_1019_kd0all.json`），同 init 副本 sha 9d2cad5e…、同 seed/det_seed/配方，仅池不同。

### §1.3 判据
- 主：**P1a − D2** 的场景轴配对 Δ、95% CI（n=15，t14）、胜场数、MDE。
- 判定：按 Lead m04096——**预期 ≤+0.2 dB 只作描述、不作主张**；若 Δ>+0.2 且 CI 下界>0，记为「候选」，是否升级为 P2 配置由 Lead 拍板。P1 不单独构成论文主张（单 seed、12k）。
- P1b − D2 同样描述性；P1b − P1a 回答 output-KD 在扩容池上的作用。

### §1.4 预算与资源
- P1a 1 卡 ≈1.5 h；P1b 1 卡 ≈1.5 h（可选）；teacher cache ≈0.7 h（GPU5，1223 中新增 jun1093 张，PNG 估 ≈1.7–1.9 GB）。
- 总 ≤4 GPU·h；不抢他人卡（mem<1500MiB 门）；B-main/S24 优先占用现有卡。

### §1.5 披露
- 单 seed、12k it；唯一 T 内容 18 种；教师对 may130 近记忆化（teacher_psnr_vs_GT≈26 dB），对 jun1093 **未见**（输出标签为真实推断，非同池记忆）；
- 池从 174→1223 同时**去掉旧 jun44（2 场景）**（按 Lead 指令）——主对照 D2 仍含 jun44，故 P1−D2 含该差异，报告须写明；
- 确定性：det_seed=1106，DataLoader 固定 generator+worker seed（Stage A 已验收 A/A2 bit-identical）。

## §2 P2：KD 接口加宽（挂起，等 P1 出数 + Lead 指示后再写死）
- 候选臂（仅候选，未启动）：(a) 输出 KD λ_o 0.1→0.2 + feature λ_f 0.01→0.02（其余同 P1 最优）；(b) 增加 RDNet decoder 1/4、1/8 层特征 KD（归一化 MSE，权重 0.005/0.01）。
- 判据：P2−P1 描述性配对 Δ+CI；何时启动由 Lead m04096 决定（「等 P1 出结果再定，别一次铺太多」）。

## §3 P3：90k 长训（挂起）
- 触发条件：Stage B-main 三种子出数，且 **D2 24k（3-seed 均值）≥20.3** 才考虑烧 8–10 GPU·h；否则不启动、维持 Plan B 收线。
- 若启动：12k 间隔 ckpt、12k/24k/48k/90k 曲线、对照 LightRR best-of-2 20.221；细节届时补充登记。

## §4 产物与状态（写定时）
- 预计产物：`/tmp/ri_dev/SFAM_P1_1106.md`、`eval/table_final_1019_kd0p1.{md,json,csv}`、merged pool/cache 目录及其 meta/manifest/sha256。
- 写定时状态：Stage B-main 6 臂 17:24:54 已起跑（GPU0/1/3/4/6/7）、S24 训练中（GPU2）、P1 池/cache 未生成、无 P1 ckpt；Stage A/bis/scaling 判据与结论不变。
