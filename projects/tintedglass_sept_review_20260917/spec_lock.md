<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: zh-CN
- audience: 课题组组会现场：导师与同组研究生（含做视觉/底层任务的同学）。熟悉 RDNet→ExpoMamba 路线与 PSNR/gm、常数基线、结构相关等口径，见过早期"零样本崩塌"结论，但不知道 9 月后半段的数据卫生事故与出口判决细节
- objective: 按时间线把 9 月进展汇报清楚——基准与起点、三个困难、主结果与两个负结果及其归因、下一阶段两条并行方向的判断与节奏，让听众能复述三条主结论并认可风险已如实标注（不向听众征求技术方向）
- core_message: 论文主线押"域适配 + 可辨识性地图"：一个 +6.4 dB 的主结果，加上把可辨识性边界量化成图；方法增量是上行空间而非前提；本周并行验证特征侧与数据侧两条腿
- consumption_mode: balanced

## mode
- mode: custom
- mode_references: briefing, narrative, pyramid
- mode_behavior: 以 briefing 为骨——主题式页标题、兄弟论点等重、可扫描的表格与数字带、页脚固定的口径与来源行；开场三页借 narrative 的落差（官方权重塌到 12.05 dB）建立张力；两页判断（域适配成功、我的判断）转 pyramid 式结论句标题，证据在其下支撑。全片不设目录页，改用章节色带与页脚章节名导航。

## visual_style
- visual_style: custom
- visual_style_references: data-journalism
- visual_style_behavior: 出版级数据版式：图表是页面骨架而非被装在卡盒里的插画，多栏网格承载图表与数字，1px 发丝规则线（divider 色）分隔区块，侧栏收纳口径与来源注释，页脚常驻"来源 · 口径"行。卡片让步于规则线与留白；除以 secondary_bg 铺底的数据面板外不画装饰性容器。标题用中性无衬线（投影可读性优先），"出版感"由 Consolas 数字角色与严格规则线承载。

## colors
- background: #FFFFFF
- secondary_bg: #F1F4F6
- primary: #123A5C
- accent: #0E7C86
- secondary_accent: #C2410C
- body_text: #17202A
- secondary_text: #5A6672
- divider: #D5DCE2

## typography
- font_family: Microsoft YaHei, Arial, sans-serif
- title_family: Microsoft YaHei, Arial, sans-serif
- body_family: Microsoft YaHei, Arial, sans-serif
- data_family: Consolas, Microsoft YaHei, monospace
- body: 24
- title: 42
- subtitle: 32
- lead: 30
- data: 24
- display: 64
- annotation: 18
- footnote: 16

## icons
- library: tabler-outline
- stroke_width: 2
- inventory: tabler-outline/chart-line, tabler-outline/alert-triangle, tabler-outline/checklist, tabler-outline/database, tabler-outline/adjustments, tabler-outline/scale, tabler-outline/route, tabler-outline/wall

## images
- cover-band: images/rdt_s1_v21_t_alpha03.png | source=user | crop=no-crop
- testset-sheet: images/testset_v21_sheet_0915.png | source=user | crop=no-crop
- e3-error: images/E3_error_vs_alpha.png | source=user | crop=no-crop
- e3-structure: images/E3_structure.png | source=user | crop=no-crop
- alpha-curves: images/alpha_sat_curves.png | source=user | crop=no-crop
- alpha-maps: images/alpha_sat_maps.png | source=user | crop=no-crop
- leak-check: images/cluster_leak_check_0915.png | source=user | crop=no-crop
- r-pool-preview: images/r_sources_preview_0915.png | source=user | crop=no-crop
- delta-alpha: images/fig_delta_alpha.png | source=user | crop=no-crop
- boundary-map: images/fig_boundary_map.png | source=user | crop=no-crop
- exit-counterfactual: images/fig_exit_counterfactual.png | source=user | crop=no-crop
- error-budget: images/fig_error_budget.png | source=user | crop=no-crop
- r-pool: images/fig_r_pool.png | source=user | crop=no-crop

## page_rhythm
- P01: anchor
- P02: dense
- P03: dense
- P04: dense
- P05: breathing
- P06: dense
- P07: dense
- P08: dense
- P09: dense
- P10: dense
- P11: dense
- P12: dense
- P13: dense
- P14: dense
- P15: dense
- P16: breathing
- P17: anchor

## pptx_structure
- mode: flat

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
- Mixing icon libraries
