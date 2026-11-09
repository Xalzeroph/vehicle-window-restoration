# 数据侧待办清单（DATA_SIDE_TODO，2026-11-08；Lead 指令 C）

> 用途：把当前所有**数据侧**待办集中登记，标 owner / 依赖 / 状态，每完成一项更新状态。**不开新实验、不改论文、不 commit。**
> 口径：owner 里「用户」= 采集/参数决策；「synth-dev」= 数据协议与建集；「exp-dev」= 本人（统计/评测/文档）；「lit-survey」= 文献/投稿规则。

## P0 用户提供膜/玻璃规格与采集资源参数（阻塞一切正式采集）
- 需要：膜型号/批次/透射光谱或 `film_transmittance_est`；玻璃尺寸/厚度/安装方式；可用采集场地清单（室内/户外/车窗）、车辆资源（3–5 辆）；可投入的采集天数与人员；隐私/同意书模板是否就绪；预算（色卡/黑绒布/转台/量角器）。
- owner：用户（Lead 汇总）；依赖：无；状态：**待用户提供**。
- 影响：决定 benchmark 轨道 B 的最小规模（40 场景）与排期；未定前只能做现有 1019/nov02 的诊断，不启动新采集。

## P1 SOP v1.1 与车窗采样协议
- SOP v1.1 已出：`code/local_tools/SOP_COLLECTION_v1_1_1108.md`（=服务器 `/tmp/ri_dev/`），新增 ① `angle_deg/angle_measure_method/angle_estimated/angle_reference_mark`；② `background_id` + `illumination_label`；③ 附录 A 车窗对照采样（3–5 辆、1 页流程）。
- 车窗采样协议（synth-dev task-17 `2026.11.08_车窗对照采样协议_1108.md`）与 validity 预登记（exp-dev `2026.11.08_车窗对照_validity预登记.md`，#sha 76e79b68…）已就绪。
- owner：exp-dev（SOP）/ synth-dev（车窗协议）；依赖：P0 的膜型与资源参数；状态：**SOP v1.1 已交付，等 Lead/用户确认后冻结**；车窗协议待与 validity 预登记对齐。

## P2 nov02 报告 v3（角度扫描口径）
- 已完成：`NOV02_TEST_1108.md` v3 36 行 sha c15e592c…（同日/同地点/同一玻璃/绕光轴角度扫描；angle labels not recorded；5 簇不当作角度分组；三条 Limitations；探索性角度分析）；verify OK。
- 待办（数据侧）：**用户补录 15 个 session 的实际角度值/采集时间**（`angle_deg` 或估计值与方式）→ 写入 nov02 meta → 重跑角度探索（`nov02_angle_explore_1108.py`）与报告「角度扫描」节；若角度标签可得，可做「角度 vs ΔPSNR」正式趋势（仍描述性）。
- owner：用户（角度值）/ exp-dev（回填与重跑）；依赖：用户补录；状态：**报告 v3 已交付；角度数值待补录（angle labels not recorded）**。

## P3 benchmark 轨道 B 的 8 周排期（等用户参数）
- 已交付草案：`/tmp/ri_dev/BENCHMARK_V1_1_PLAN_1108.md`（57 行 sha 450d264f…；两轨：A=ICME 不承诺新采集，B=ICCV 2027 最小 40 场景 ×≥3 GT；schema 9 新字段；评测协议 identity 基线/禁 PSNR_Ω/cluster-aware/不可辨识度一等指标；4 周采集+2 周 QC+2 周基线与写作）。
- 待办：等 P0 参数（场地/车辆/人员/天数）后把 8 周排期落到具体周次与 owner；写采集排期模板（按周，含每日 target 场景数与 QC 回流节点）；确认轨道 A 的 ICME 只写现有 1019+nov02 口径（不承诺新采集）。
- owner：Lead + 用户；依赖：P0；状态：**草案已交付，排期待参数**。

## P4 validity 统计（等车窗数据）
- 已就绪：预登记 `2026.11.08_车窗对照_validity预登记.md`（sha 76e79b68…；判据 IQR 重叠+中位差≤0.10；only descriptive）+ 脚本 `code/local_tools/carwind_validity_1108.py`（服务器 `/tmp/ri_dev/`，dry-run 输出 PENDING rc=2，数据目录 `data/validity/carwind_1108/` 尚未存在）。
- 待办：数据到达 → 登记 vehicle/group ids 与 n → `python3 carwind_validity_1108.py --prereg-sha 76e79b68…` → 出 `eval/table_carwind_validity_1108.md` + 分布图 → `--verify` 复算 → 回 Lead 表路径/sha/判定/一句英文 claim。
- owner：用户（拍摄）/ synth-dev（协议）/ exp-dev（统计）；依赖：车窗数据交付（P1 附录 A）；状态：**脚本+预登记就绪，等数据**。

## P5 正式 A/B 数据 intake → QC → build → lock → verify（正式数据集链路）
- 需要（Lead 拟定的五步验收）：① intake 清单（`batch_id`、文件数、sha256、MANIFEST.json；rsync 只传一次不覆盖）；② 几何/光度 QC（shift≤2 px、|g−1|≤0.15、GT 目检、元数据 100%、角度/背景/光照标签一致）；③ build（按场景/序列建 {I,GT} 对、写 pairs manifest 与 I 帧选择规则——**先写规则后选帧，不按模型分数挑**）；④ lock（冻结文件与 sha256、写 split：train/val/test 与 dev 披露清单）；⑤ verify（内容级泄漏核查 `check_t_src_leak.py` 退出码 1 即停；平凡基线 identity + 官方冻结 ckpt 必测；预登记判据与统计功效先写）。
- 待办：待 P0/P1 完成后写 intake 模板与 QC 脚本（可复用 1019/nov02 的 `nov02_provenance_1108.py`、`nov02_build_set_1108.py`、`verify_*`）；正式数据到后按 ①→⑤ 执行并把每步状态回填本文件。
- owner：synth-dev（建集/QC）/ exp-dev（评测/verify）/ Lead（lock 决策）；依赖：P0 + 正式采集（P3）；状态：**待启动（等正式数据）**。

## 状态更新约定
- 每完成一项，在本文件对应条目末尾追加「YYYY-MM-DD 状态：…（产物路径/sha）」；不删除历史状态行。
- 只有 P0/P3/P4/P5 属于「数据侧」阻塞项；P1/P2 为已交付文档/报告回填，不阻塞推理。
