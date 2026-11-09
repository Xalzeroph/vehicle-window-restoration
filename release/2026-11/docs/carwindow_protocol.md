# 2026.11.08 真实贴膜车窗对照采样协议（validity study，v1.0）

> 目的：把「我们的贴膜玻璃与真实车窗性质相近」从经验观察升级为**可测证据**：用与受控装置相同的黑布配对协议，在 3–5 辆真实贴膜车上采 {I, 黑布 GT} 小样本。
> 用途：只作 **validity study / 分布对照**（论文 Data protocol 一节）；`usage="validity-study-only"`、`not_for_benchmark=true`，**绝不进 benchmark 训练/测试集，也不与 benchmark 分数混排**。
> 负责：synth-dev（数据管线）。数据回传后由我跑 `validity_carwind_1108.py` 生成 JSONL + MANIFEST(sha256) + 报告。

## 一、要拍什么（先选车）
- **3–5 辆真实贴膜车**，尽量覆盖不同膜型/颜色/深浅（染色 / 金属 / 陶瓷；深色与浅色 VLT 各至少 1 辆）。
- 每辆车选 **1–2 个侧窗或后挡**（尽量平面、少窗框，避免强脏污/气泡）；同一辆车的同一车窗为一"组"。
- 相机、时段、天气尽量统一：同一台手机/相机、同一天内（或同为阴天散射光），避免正午直射与夜间混合光。
- 不需要真车行驶、不需要多帧视频；**每个车窗 1–2 组静态 {I, GT} 即可**。

## 二、每组怎么拍（SOP v1 黑布法，逐条照做）
1. **锁死曝光**：长按取景框锁 AE/AF；专业模式连白平衡一起锁；关闭 HDR / 夜景 / 美颜 / 自动追焦；全程不变焦、不关灯、不开补光。
2. 先拍 I（输入帧）：相机/人退到反射区外（不要让拍摄者身影入画），保持与车窗相对角度固定，连拍/静拍 ≥1 张；记录角度估计（如 20°）。
3. 铺黑布 GT：哑光吸光黑布**完全遮满可见玻璃并外扩 20–30 cm**，贴平不起褶；布置好**静止 ≥3 秒**再拍 GT（消除手抖/移动），拍 1 张即可。
4. 拍完撤布；同一车窗换角度/换组时重复 2–4 步。每组 I 与 GT 之间**相机位置/焦距/曝光不变**。
5. 每辆车额外补 1 张**车窗全景说明照**（能看清车窗位置、膜色、是否有窗框），编号 `overview.png`（不参与指标计算）。

## 三、回传目录与命名（放到 `data/validity/carwind_1108/`）
```
data/validity/carwind_1108/
  carwind_<car_id>/meta.json
  carwind_<car_id>/pair_01/I_000001.png   # 输入帧（可多张，用 I_000002... 区分）
  carwind_<car_id>/pair_01/GT.png         # 黑布遮满后的目标帧（每组一张）
  carwind_<car_id>/pair_01/overview.png   # 可选：全景说明照
  carwind_<car_id>/pair_02/...
```
- 图片用原始格式（PNG/JPG 均可，不要压缩到看不清）；<car_id> 用简单英文/数字（如 `carA`、`car01`），不要用中文路径。
- 允许车目录下直接放图片（一个 I 配一个 `GT.png`），但**推荐按 `pair_<k>/` 子目录**，one pair = one I + one GT。

## 四、每车 `meta.json`（必填，照下面抄）
```json
{
  "car_id": "carA",
  "captured_at": "2026-11-08 15:30",
  "operator": "你的名字",
  "license": "research-only",
  "glass_desc": "左前侧窗，深色，曲面，有窗框",
  "lighting": "阴天户外散射光",
  "camera": {"model": "iPhone 15", "locked": true, "iso": 100, "wb": "manual"},
  "angle_estimate_deg": 20,
  "film_glass_spec": {
    "film_brand": "3M", "film_model": "", "film_type": "dyed",
    "vlt_pct": 35, "ir_rejection_pct": 45, "uv_rejection_pct": 99,
    "film_age_months": 24, "glass_color": "green", "glass_thickness_mm": 5.0,
    "is_curved": true, "is_automotive_grade": true, "source": "installer_record"
  }
}
```
`film_glass_spec` 中文对照：**膜品牌**=film_brand（可空）、**型号**=film_model（可空）、**膜类型**=film_type（`染色=dyed`/`金属=metallic`/`陶瓷=ceramic`/`未知=unknown`）、
**VLT%**=vlt_pct、**IR 阻隔%**=ir_rejection_pct、**UV 阻隔%**=uv_rejection_pct、**贴膜时长(月)**=film_age_months、**玻璃颜色**=glass_color、**厚度(mm)**=glass_thickness_mm、
**是否曲面**=is_curved、**是否汽车级**=is_automotive_grade、**来源**=source（spec_sheet/installer_record/visual_estimate/vin_decode/unknown）。
参数不知道就写 `null` 或 `"unknown"`，**不要编造**；可向车主/贴膜店要参数。

## 五、自查（拍完每组当场看，不合格重拍）
- [ ] 对齐：I 与 GT 视角基本一致（不要移动相机；脚本会算 shift，>2 px 提示）
- [ ] 曝光：AE/AF/WB 已锁定，I/GT 亮度接近（脚本会算 g_affine，偏离 1 太多提示）
- [ ] GT 干净：黑布无褶皱/卷边/反光，**GT 里不要露出玻璃边缘或窗框**
- [ ] I 干净：黑布已撤走，拍摄者/手机不进入反射区，无遮挡物
- [ ] 内容：车窗在画面中占主要面积，不糊、不过曝、不过暗
- [ ] 记录：meta.json 填完，角度与时间写下

## 六、隐私与授权（重要）
- 这是**内部研究用**数据：避免拍到清晰可识别的**人脸/车牌正面**；不可避免地拍到人像/车牌时，回传时说明，**公开前必须模糊化**。
- 确认车主/数据所有者同意用于内部研究（`license: research-only`）；不要上传到公开网盘。

## 七、需要用户提供的信息清单（回传时一并告诉我）
1. 车辆数与每车 `car_id`（3–5 辆，尽量不同膜型/深浅）
2. 每车窗的膜品牌/型号（可空）
3. 膜类型（染色/金属/陶瓷/未知）
4. VLT%（不知道写 unknown，可查贴膜店）
5. IR 阻隔% / UV 阻隔%（若知道）
6. 贴膜时长、玻璃颜色、厚度、是否曲面、是否汽车级（能填多少填多少）
7. 拍摄设备型号与锁定参数（iso/快门/WB 或截图）
8. 拍摄时间/天气/地点（内部记录即可）
9. 是否含可识别人像/车牌（回传时说明，内部保留、公开前脱敏）
10. 数据许可：确认 `research-only`（或其他约定）并同意内部研究使用
