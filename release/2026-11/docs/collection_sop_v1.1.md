# 贴膜玻璃 SIRR 数据采集 SOP（v1.1，2026-11-08；面向采集执行人员，直接执行）

> 目标：为「贴膜玻璃单帧反射去除」benchmark 采集可复现的 {I, T} 配对数据。核心原则：**同一场景、同一装置、同一构图下，只改变反射/遮挡条件**；GT（黑布）与 I 帧必须同构图同参数可对齐。
> v1.1 相对 v1 增补：① 每个 session 记录玻璃旋转角度；② `background_id` 与 `illumination_label`；③ 附录 A 车窗对照采样（可选）。主文仍控制在 1 页内，附录不计。

## 1. 场景与装置布置
1. 固定相机（三脚架），固定贴膜玻璃（支架）；记录相机镜头到玻璃的距离 `dist_to_glass`、入射角 `angle_of_incidence`。
2. 相机正对玻璃中心；画面保留玻璃边缘/窗框作为对齐参考。
3. 每换一个场景/装置/膜型，先拍 1 张标定照（含色卡或白纸），进 `CALIB/`，记录 `film_model/film_batch/film_transmittance_est`。
4. 采集前清洁玻璃指纹/灰尘；记录室内外、天气、估计照度（手机 lux App 即可）。

## 2. 光照条件与强弱反射（每场景每个光照条件都要覆盖）
1. 光照清单：白天自然光、夜晚室内灯（暖/冷各一）、逆光/强点光源（车灯或手电）。
2. **每个场景的每个光照条件至少各含 1 个强反射 + 1 个弱反射**（用反射源远近/角度或灯光强度控制）；强弱标签写入元数据（`reflection_strength` 分档 strong/weak），否则无法做强弱分层报告。
3. 记录 `illumination` 与 `reflection_type`（车灯/行人/天空/室内灯/混合）。
4. **`illumination_label`**：给每个光照条件 block 起固定标签（如 `day_cloudy_1` / `night_indoor_warm_1` / `backlight_1`），同一 block 内光照不变才可改参数；逐 pair 写入该字段。
5. **`background_id`**：给玻璃后（透射侧）每一组不同的场景/物体布置起固定标签（如 `bg_office_1`、`bg_street_2`）；同 `background_id` 内不得更换透射景物/大幅移动背景物体，变化必须换 id。

## 3. 黑布 GT 拍法（方向与时间逻辑写死）
1. 材质：**哑光黑绒布**，尺寸 ≥ 玻璃可视面；不反光、无褶皱亮斑。
2. 遮法：黑布遮**玻璃外侧（反射源一侧）整面**，贴近玻璃、不留缝、不反光；**相机侧不动、不碰三脚架**。
3. 遮完立即确认画面构图与 I 帧一致（拍前后帧对比/目检）；构图不一致必须重拍。
4. 时间逻辑：**GT 与对应 I 的拍摄时间差 ≤10 s**；若用视频模式，GT 前后各保留 **≥30 帧稳定段**；**帧号差值不作为要求**（旧规则「帧号间隔≥30 帧」作废）。
5. 每个场景每个光照条件拍 **≥3 组交替的 {I, GT}**（I→GT→I→GT…），可选加拍暗屏 GT 子集并标注 `gt_type`。

## 4. 角度记录（v1.1 新增；若做「角度扫描」实验必须执行）
1. 在玻璃支架/转台/窗框上做**基准标记**（`angle_reference_mark`，如初始朝向对齐窗框边）；定义 `angle_deg` = 玻璃绕**光轴（z 轴）相对基准的旋转角**。
2. 测量方式：转台刻度优先；否则数字量角器/手机水平仪贴玻璃面；把方法写入 `angle_measure_method`（`turntable` / `digital_protractor` / `phone_level`）。
3. 每天/每 block 开始前校零一次；角度设定值写入元数据 `angle_deg`；**实际值与设定值有偏差时以实测为准**。
4. 无法直接测量时允许估计，但必须把 `angle_estimated` 置 `true` 并在 `angle_measure_method` 写估计依据（如“按转台等分，±2° 估计”）；**不得编造精确角度**。
5. 每个 session 的 `angle_deg` 与 `background_id`、`illumination_label` 一起构成该 session 的三个独立变量标签。

## 5. 参数锁定（按光照 block）
1. 以「光照条件 block」为单位：每个 block 内**曝光/ISO、白平衡、对焦全部锁死**（`exposure_locked/wb_locked/focus_locked=1`）。
2. block 之间允许按光照重新设定参数，但**必须把该 block 的曝光/ISO/WB/焦距写进元数据**（每个 pair 行都带）。
3. 禁止自动曝光/自动 WB/自动对焦/HDR/美颜/降噪等任何自动处理。

## 6. 文件命名与目录（batch_id 为第一级）
```
<batch_id>/<site>/<scene_id>/<sequence_id>/
  I/frame_<6位帧号>.jpg            # 输入帧（含反射）
  GT/<seq>_gt<k>.png               # 第 k 个黑布 GT（k=1..≥3）
  CALIB/                           # 标定照 + 色卡
  meta/<seq>.jsonl                 # 每行一个 pair，含 pair_id
  privacy/<seq>_regions.json       # 隐私 bbox（原始帧仅内部保存）
```
- 文件系统路径与文件名**只允许 ASCII 字母/数字/下划线**；`batch_id`、`scene_id` 规则：`<日期>_<in/out>_<膜型>_<序号>`，如 `20261201_in_filmA_003`。

## 7. 元数据模板（JSONL，每行=一个 pair，字段名英文）
`batch_id, scene_id, sequence_id, pair_id, frame_idx, gt_idx, split, date, site_id, indoor_outdoor, scene_type, illumination, illumination_label, weather, background_id, camera_model, device_id, lens, focal_mm, exposure, iso, wb, glass_size, film_model, film_batch, film_transmittance_est, tint_level, mounting, angle_of_incidence, angle_deg, angle_measure_method, angle_estimated, angle_reference_mark, dist_to_glass, reflection_type, reflector_type, reflection_strength, gt_type, n_gt_per_scene, gt_frame_distance, focus_locked, wb_locked, exposure_locked, g, shift_px, alignment_method, qc_status, qc_flags, privacy_regions, license, consent`
（`g`、`shift_px` 由后处理填；现场只填可观测字段；缺失字段填 `null`。）

## 8. 现场 QC 通过标准（不通过就补拍）
1. 几何：I 帧与 GT 帧配准误差 `shift ≤ 2 px`。
2. 光度：全局增益 `g = median(I/T)`，`|g − 1| ≤ 0.15`。
3. 图像：无过曝/欠曝（>99% 像素不在 0/255 饱和）、无运动模糊、无遮挡物入镜。
4. 配对：I 与 GT 同构图同参数、时间差 ≤10 s；每场景 ≥3 组有效交替 {I,GT}。
5. **GT 目检**：黑布区域亮度均匀、**无反射残留、无窗外景物可见**。
6. 元数据完整率 100%（缺一项即打回）；每批交付前**随机 5 场景由 owner（用户）目检签字**。
7. **角度/背景/光照标签一致性**：同 block 内 `illumination_label` 不变；同 `background_id` 内背景物体/构图不变；`angle_deg` 与 `angle_measure_method` 非空（估计时 `angle_estimated` 必为 true）。

## 9. 隐私与许可
1. 原始帧（含人脸/车牌）**只在内部保存**；交付/发布一律用**模糊版**。
2. 逐帧记录隐私 bbox 到 `privacy_regions`，模糊版及 bbox JSON 放 `privacy/`。
3. 采集需现场同意：交付含**同意书模板**与 `license`、`consent` 字段；测试集 label 不公开。
4. 不在私人住宅内未授权拍摄；不拍可识别人脸特写。

## 10. 交付清单（每批次）
- 原始 I/GT 帧 + `meta/*.jsonl`（含 `angle_deg`/`background_id`/`illumination_label`）；
- `CALIB/`（标定照 + 色卡）、`privacy/`（模糊版 + bbox JSON）、`QC_REPORT.md`（每场景通过率/GT 数/|Δ| 分布/角度范围）、`MANIFEST.json`（全部文件 + sha256）；
- 随机 5 场景目检签字表；
- 上传：`rsync` **只传一次、不覆盖**（新批次用新 `batch_id` 目录）。

## 附录 A（可选，1 页流程）：车窗对照采样（validity study）
**目的**：用 3–5 辆真实车窗的 {I,GT} 与我们贴膜玻璃做**光度统计对照**（仅描述性，写论文 validity 句；不做方法评测）。
1. **对象**：3–5 辆普通乘用车，侧窗/后窗各选 1 面；记录 `vehicle_id`、`window_position`、是否已贴膜 `film_installed`。
2. **布置**：相机固定于车内/窗外正对玻璃中心（三脚架），画面含车窗框作对齐参考；`dist_to_glass`、`angle_of_incidence` 同主 SOP。
3. **GT**：黑布从**玻璃外侧（反射源一侧）**整面遮严、贴近、不留缝；相机侧不动；GT 与 I 时间差 ≤10 s。
4. **采样**：每辆 1–2 组 `{I_000000.png, GT.png}`；每组至少 3 次交替 I→GT→I→GT，保留最优一组有效对；记录 `illumination_label`、`background_id`、`angle_deg`（若转车角度可测则记，否则 `angle_estimated=true`）。
5. **交付**：`data/validity/carwind_1108/<vehicle_id>/<group>/{I_000000.png,GT.png}` + `meta.jsonl`（字段同主模板 + `vehicle_id, window_position, film_installed`）。
6. **禁用**：不拍可识别车牌/人脸（模糊化同主 SOP）；不做任何显著性/性能主张；该数据只进 validity 统计表。
