<!-- ppt-master-schema: design-spec/v1 -->
# TintedGlass Sept Review - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | tintedglass_sept_review_20260917 |
| Canvas Format | ppt169 (1280 × 720) |
| Page Count | 17 |
| Primary Language | zh-CN |
| Target Audience | 课题组组会现场：导师与同组研究生（含做视觉/底层任务的同学）。熟悉 RDNet→ExpoMamba 路线与 PSNR/gm、常数基线、结构相关等口径，见过早期"零样本崩塌"结论，但不知道 9 月后半段的数据卫生事故与出口判决细节 |
| Communication Intent | 按时间线汇报 9 月进展：先给基准与起点，再如实讲清三个困难，然后给主结果与两个负结果及其归因，最后落到下一阶段两条并行方向的判断与节奏。口径是"讲已做的决策与依据"，不向听众征求技术方向 |
| Desired Audience Outcome | 能复述三条主结论（域适配 12.05→18.5 dB、结构相关 0.26→0.63；w 重加权与物理除法出口的净效应只是工作点搬移；R̂ 质量是低 α 档的全局瓶颈），并认可结论带了口径保护、已知风险已如实标注 |
| Core Message / Ask / Action | 论文主线押"域适配 + 可辨识性地图"：一个 +6.4 dB 的主结果，加上把可辨识性边界量化成图；方法增量是上行空间而非前提；本周并行验证特征侧与数据侧两条腿 |
| Delivery Context | 主要是有人主讲、现场投影的 15–20 分钟组会汇报（会留提问）；次要为会后自行翻阅的进展归档，因此关键页需自带必要上下文与数字口径 |
| Artifact Afterlife | 9 月进展归档，兼作后续周会汇报、论文写作与实拍 sanity 排期的素材依据 |
| Reading Mode | balanced |
| Content Strategy | 放开重构：可按汇报节奏重新组织章节、合并或拆分页面，大纲只作素材。事实与数字一律取自实验记录与 eval/ 产物，不得引入源外数字 |
| Design Style | 实验台账（A 方向）：出版级数据版式 + 中性完整报告骨架 |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — 最终 Stage-2 主动值 |
| Custom Animations | disabled — 工作流默认 |
| Narration Audio | disabled — 工作流默认 |
| Created Date | 2026-09-17 |

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | ppt169 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 左右 64 / 上 56 / 下 48 |
| Content Area | x 64–1216 (1152 宽) · y 56–672 (616 高) |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode References**: briefing, narrative, pyramid
- **Mode Behavior**: 以 briefing 为骨——主题式页标题、兄弟论点等重、可扫描的卡片/表格/仪表结构、页脚固定的口径与来源行；开场三页借 narrative 的落差（官方权重塌到 12.05 dB）建立张力；两页判断（域适配成功、我的判断）转 pyramid 式结论句标题，证据在其下支撑。全片不设目录页，改用章节色带与页脚章节名导航。
- **Visual style**: custom
- **Visual Style References**: data-journalism
- **Visual Style Behavior**: 出版级数据版式：图表是页面骨架而非被装在卡盒里的插画，多栏网格承载图表与数字，发丝规则线（1px，divider 色）分隔区块，侧栏收纳口径与来源注释，页脚常驻"来源 · 口径"行。卡片让步于规则线与留白；除少量以 secondary_bg 铺底的数据面板外不画装饰性容器。标题较 data-journalism 的衬线惯例改用中性无衬线（投影可读性优先），"出版感"由 Consolas 数字角色与严格规则线承载。
- **Theme**: 实验台账——把一个月的工作按"证据链"摊开：每个数字旁边站着它的口径，每个结论下方压着它的基线
- **Tone**: 克制、精确、不辩解；负结果与正结果同等分量地被陈列

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FFFFFF | 页面底色，出版纸 |
| Secondary background | #F1F4F6 | 数据面板铺底、表头、代码/公式块 |
| Primary | #123A5C | 标题、结论句、关键结构线 |
| Accent | #0E7C86 | 主结果与关键数字（正向）；图表第一序列 |
| Secondary accent | #C2410C | 负结果、代价、风险、常数基线警戒线；图表第二序列 |
| Body text | #17202A | 正文段落与列表 |
| Secondary text | #5A6672 | 口径注释、图注、侧栏文字 |
| Divider | #D5DCE2 | 发丝规则线、表格边框、网格线 |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | 中性无衬线，字重对比建立层级 | Microsoft YaHei | Arial | sans-serif |
| Body | 中性无衬线，投影可读 | Microsoft YaHei | Arial | sans-serif |
| Data | 等宽，数字对齐与"仪表感" | Microsoft YaHei | Consolas | monospace |

- **Role rationale**: `data` 为新增 recurring role——本片大量关键数字（PSNR、Δ、CI、α 档位）需要等宽对齐与可扫读的仪表感，是 data-journalism 性格的主要承载者；中文单位与标签仍走 Microsoft YaHei。
- **Title stack**: `Microsoft YaHei, Arial, sans-serif`
- **Body stack**: `Microsoft YaHei, Arial, sans-serif`
- **Data stack**: `Consolas, Microsoft YaHei, monospace`

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 24 |
| Title | 42 |
| Subtitle | 32 |
| Lead | 30 |
| Data | 24 |
| Display | 64 |
| Annotation | 18 |
| Footnote | 16 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: 页标题 → 一行判断/主张 → 图表或数字主体 → 侧栏口径与页脚来源；眼睛自上而下、由主张到证据
- **Composition tendency**: 上下分带 + 左右分栏的数据版式：顶带放标题与判断句（下压一条发丝规则线），主体按内容切 2–3 栏或满宽横带，右或下侧栏收纳口径注释；章节色带随 Part 变化，位置恒定
- **Cross-page continuity**: 顶带规则线 + 页脚"来源 · 口径 · 页码"三件套恒定；`data` 数字角色与 accent/secondary-accent 的语义（青=正向主结果，橙=代价与风险）全片一致
- **Spacing posture**: 密排但有网格——除封面、时间线、判断页三处留白页外，正文页偏 dense
- **Spacing anchors**: 页边距 64 · 块间距 24 · 栏间距 28 · 圆角 0 · 正文行距 36

## VI. Icon Usage Specification

- **Primary bundled library**: tabler-outline
- **Stroke Width**: 2

| Icon Path | Suitable Scenarios |
| --- | --- |
| tabler-outline/chart-line | 进展与曲线类页面（低 α 专支、可辨识性地图） |
| tabler-outline/alert-triangle | 困难与负结果页的风险标记 |
| tabler-outline/checklist | 基准口径与"每轮固定三件事" |
| tabler-outline/database | 数据池、扩池、数据卫生事故 |
| tabler-outline/adjustments | 适配器、超参、配置类页面 |
| tabler-outline/scale | 边界、天花板、可辨识性上限 |
| tabler-outline/route | 路线、计划、时间线 |
| tabler-outline/wall | 瓶颈、阻塞类页面（R̂ 全局瓶颈） |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rdt_s1_v21_t_alpha03.png | 2474x790 | 3.13 | 六列对照拼图（输入 / GT / 零样本 / pilot / 域适配），月度主证据 | Photo | 满宽横带置于页面下部，标题与判断句居上；六列等分本身就是"逐列读"的阅读路径，不加框、不加角标 | no-crop | user | Existing | fig/rdt_s1_v21_t_alpha03.png | none | local |
| testset_v21_sheet_0915.png | 1280x1988 | 0.64 | 合成测试集逐场景总览（竖版 sheet） | Photo | 左栏竖长面板，高度吃满主体区，右侧留给两张分析小图 | no-crop | user | Existing | fig/testset_v21_sheet_0915.png | none | local |
| E3_error_vs_alpha.png | 910x585 | 1.56 | 零样本误差随 α 下降急剧上升 | Chart | 右栏上半，与下方结构相关图叠放，两图共用一条竖直对齐线 | no-crop | user | Existing | fig/E3_error_vs_alpha.png | none | local |
| E3_structure.png | 1686x190 | 8.87 | 结构相关随 α 的变化（超宽条） | Chart | 右栏下半，作为一条极扁的证据条，与上方误差图形成"高矮对照" | no-crop | user | Existing | fig/E3_structure.png | none | local |
| alpha_sat_curves.png | 1950x546 | 3.57 | 四个独立估计器的 α̂ 饱和曲线（全部撞 0.7 平台） | Chart | 页面上半满宽条带，四条曲线同框压在一条 0.7 参考线上 | no-crop | user | Existing | fig/alpha_sat_curves.png | none | local |
| alpha_sat_maps.png | 2200x660 | 3.33 | α 估计图与真值的空间对照 | Photo | 页面下半满宽条带，与上方曲线共用左右边距，形成"曲线→地图"的证据递进 | no-crop | user | Existing | fig/alpha_sat_maps.png | none | local |
| cluster_leak_check_0915.png | 974x932 | 1.05 | 孪生簇泄漏核查可视化 | Chart | 右侧方图，左侧留文字说明事故与处置；图与说明等高对齐 | no-crop | user | Existing | fig/cluster_leak_check_0915.png | none | local |
| r_sources_preview_0915.png | 2710x720 | 3.76 | 现有 R 训练池逐张预览 | Photo | 页面下半满宽横带，未采样的条目视觉上直接支撑"池子很小" | no-crop | user | Existing | fig/r_sources_preview_0915.png | none | local |
| fig_delta_alpha.png | 1073x692 | 1.55 | 低 α 专支增益随 α 变化，v2.1 与扩增集两套符号对照 | Chart | 左栏主图，右栏放数字表；图与表等高，读图读表互为验证 | no-crop | user | Existing | scripts/make_ppt_figs_0917.py | none | local |
| fig_boundary_map.png | 1088x745 | 1.46 | 分档能力地图：常数基线 / 当前成绩 / R̂ 天花板 / 真值天花板 | Chart | 左栏主图，右栏陈列四条水平含义与读法 | no-crop | user | Existing | scripts/make_ppt_figs_0917.py | none | local |
| fig_exit_counterfactual.png | 1088x779 | 1.40 | 物理除法出口反事实：同 β 下 α̃ 学习版 vs 真值版 | Chart | 右栏主图，左侧先给 E≡G 与判决结论 | no-crop | user | Existing | scripts/make_ppt_figs_0917.py | none | local |
| fig_error_budget.png | 1038x743 | 1.40 | 用 R̂ 替换真 R 后的落差（1/α 放大） | Chart | 右栏主图，左侧说明"低 α 被 1/α≈2.4× 放大"的机制 | no-crop | user | Existing | scripts/make_ppt_figs_0917.py | none | local |
| fig_r_pool.png | 1487x846 | 1.76 | R 候选池 601→131 的剔除构成与来源族 | Chart | 左栏主图，右栏放扩池计划数字；图讲"为什么小"，文字讲"怎么变大" | no-crop | user | Existing | scripts/make_ppt_figs_0917.py | none | local |

## IX. Content Outline

### Part 1: 起点

#### Slide 01 - 封面

- **Audience move**: 尚未进入状态 → 一眼抓到本月最硬的一个数字，知道这次汇报的落点在哪
- **Relationships**: 主题词总领；副标题三项议程并列；钩子数字（12.05 → 18.5）作为主题的量化注脚，与议程并列而非从属
- **Composition**: 上半幅放标题块与钩子数字，下半幅压一条满宽证据横带；留白页密度
- **Title**: 深色贴膜玻璃透射成像增强：9 月进展
- **Core message**: 这个月把一个塌掉的任务做活了——ᾱ=0.3 档从 12.05 提到 18.5 dB
- **Content**: 主标题「深色贴膜玻璃透射成像增强：9 月进展」 · 副标题「域适配、可辨识性边界、与下一步」 · 钩子数字 12.05 → 18.5 dB（ᾱ=0.3，raw PSNR，plain 臂 2 seed 均值） · 页脚 2026-09-17 组会
- **Images**: 下半幅满宽横带 `rdt_s1_v21_t_alpha03.png`

#### Slide 02 - 任务与基准

- **Audience move**: 大致记得背景 → 明确本次评测的口径，以及"每轮固定三件事"的护栏
- **Relationships**: 成像模型 → 两阶段路线 → 基准与口径，构成一条任务定义链；三件护栏彼此并列
- **Composition**: 左栏定义与公式，右栏三件护栏清单；页面下部满宽横带铺测试集预览
- **Title**: 任务是把 T 从 I = α(x)·T + R 里抠出来
- **Core message**: 单帧深色贴膜（α≪1）下，基准 v2.1 用 3 档 × 12 场景，并以 raw/gm PSNR + 常数基线 + 结构相关为口径
- **Content**: 场景 = 深色车膜 α≪1 的单帧成像，目标恢复 T · 路线 = RDNet（去反射）→ ExpoMamba（低光增强） · 基准 v2.1 = 3 档 ᾱ（1.0/0.6/0.3）× 12 场景，口径 raw/gm PSNR + 常数基线 + 结构相关 · 每轮固定三件事：常数基线必须同框 / v2.1 与扩增集双测 / 配对 + 场景聚类 CI · 一句人话：车膜把光吃掉七成再叠上反射，我们要把"车里长什么样"还原出来
- **Mathematical content**: I = \alpha(x) \cdot T + R
- **Images**: 无（该页为定义页，视觉由两阶段路线示意与护栏图标承担）

#### Slide 03 - 网络架构（魔改版 RDNet）

- **Audience move**: 只知道"拿 RDNet 去反射" → 清楚哪些部分是官方权重、哪些是我们加的，改动落在哪一层
- **Relationships**: 输入 I 分出两条支路（主干 FocalNet-L 与 α 支路 ConvNeXt-s），二者在 α 门控处汇合，之后单线进入 SubNet→Decoder→残差输出；Mona 与 α 门控是并列的两处改动，冻结/可训是横跨全部模块的第三层关系
- **Composition**: 上半幅横向数据流（主链一行 + α 支路一行 + 图例一行），下半幅左右两个注解块——左为 Mona 插入点、右为参数量与训练状态表
- **Title**: 魔改后的 RDNet：α 门控与 Mona 适配器
- **Core message**: 冻结 205.7M 的 FocalNet-L 出多尺度特征；冻结的 ConvNeXt-s 出 6 维 code，经 MLP 变成 64 维通道门控、只乘到 stem 上；Mona 挂在主干每个 block 内；两者相加后域适配只训 60.7M / 266.4M
- **Content**: 主链 = 输入 I（32 倍数）→ FocalNet-L（冻结 205.7M，含 Mona）→ 1×1 投影 → α 门控 ⊗（仅调 stem）→ SubNet × 4（可逆残差）→ Decoder（唯一被调用）→ 残差式 R̂ / T̂ · α 支路 = 缩到 224² → 冻结 ConvNeXt-s → 6 维 code → MLP 6→512→64 · Mona 插点 = FocalModulationBlock 内调制残差后 mona1、FFN 残差后 mona2，24 block × 2 = 48 个，+5.27M · 参数量与状态 = 主干 205.7M 冻结 / 1×1 投影 + α 门控 1.1M 可训 / SubNet × 4 + Decoder 59.6M 可训 / Mona +5.3M 可训
- **Fact IDs**: F03-arch-frozen, F03-arch-mona
- **Visualization**: 数据流框图 + 参数量表（均为自定义绘制，非数据对象，不设 key）

#### Slide 04 - 起点：零样本崩塌

- **Audience move**: 以为官方权重跑一下就行 → 看到"还不如一张平坦灰图"，承认这是内容级损伤而非亮度问题
- **Relationships**: 三个指标（PSNR、与常数基线的差、结构相关）各自独立地指向同一结论；错误随 α 下降的走势是同一结论的机制性证据（对比关系）
- **Composition**: 左栏竖长测试集 sheet 建立"数据长什么样"，右栏上下叠放两张分析图，结论句压在顶部横贯
- **Title**: 官方权重直接跑，还不如一张平坦灰图
- **Core message**: 失败是内容级损伤——同一档上把 GT 均值涂成一片平坦图能拿 17.68 dB，零样本反而低 5.6 dB，结构相关仅 0.26
- **Content**: RDNet 官方权重在贴膜域 ᾱ=0.3 档 = 12.05 dB（gm 12.36） · 同档"平坦图常数基线"= 17.68 dB → 零样本比平凡解低 5.6 dB · 结构相关仅 0.26 · 误差随 α 下降急剧上升 → 内容是坏的，不是暗的
- **Images**: 左栏 `testset_v21_sheet_0915.png`；右栏上 `E3_error_vs_alpha.png`、下 `E3_structure.png`

### Part 2: 这个月走过的路

#### Slide 05 - 九月时间线

- **Audience move**: 只记得零散几次讨论 → 对这一个月的推进节奏与转折点形成一条完整的时间轴
- **Relationships**: 六个里程碑按时间顺序推进；搬迁与数据卫生事故是两个节奏断点
- **Composition**: 单页横向时间轴，六个节点等距，节点上方标里程碑、下方标该阶段的主产出；留白页密度
- **Title**: 九月这条路：从定基准到把边界量化
- **Core message**: 9.9 预演与实拍协议 → 9.13 基准定版 + pilot + α 平台 → 9.14 搬迁 → 9.15 数据卫生重建 → 9.16 S1 域适配判决 → 9.17 出口判决 + 锚点探针 + 新方向
- **Content**: 六个里程碑连成一条轴；标注两个断点（搬迁、数据卫生事故）各自的代价与恢复方式
- **Visualization**: 页面级时间轴（自定义绘制，非数据对象，不设 key）

#### Slide 06 - 困难①：α 的单帧可辨识性平台

- **Audience move**: 以为 α 头没调好 → 认识到 0.7 是单帧可辨识性的上限，不是调参问题
- **Relationships**: 四个独立估计器（不同特征源与架构）落在同一平台，构成"互相独立却同结论"的并列关系；理论上的乘积歧义是成因
- **Composition**: 上半带左为 α 头饱和曲线、右为四个估计器在无膜区的原生读数块（两者等高）；下半带左为 α 估计图、右为读法
- **Title**: 四个独立估计器全部撞在 0.7：这不是调参问题
- **Core message**: 在 I=αT+R 下 α→s·α、T→T/s 观测不变，单帧只能定乘积；无膜区（真值 1.0）四个独立估计器全部落在 0.605–0.741，远低于 0.9 目标，与 Retinex/去雾/阴影同属"先验 + 锚点"消歧问题
- **Content**: α 头 = 挂在冻结特征上的可训练小模块（0.711）· ridge-ALL = 逐像素线性回归基线、31 维手工统计、零训练（0.741）· GLOBAL = 整图一个常数、21 维全局统计、零训练（0.634）· B 臂 = 不加参数，从 R̂/T̂ 反推（0.605–0.680）· 理论：α→s·α 与 T→T/s 观测不变 → 单帧只能定乘积，本征分解的经典歧义 · 文献一致：Retinex / 去雾 / 阴影同为"先验 + 锚点"消歧 · 方向：与其继续调 α 头，不如引入结构锚
- **Fact IDs**: F05-alpha-plateau, F05-estimator-spread
- **Visualization**: 四个估计器在无膜区的原生读数块（名称 + 数值两列，非数据对象，不设 key）
- **Images**: `alpha_sat_curves.png`（左上）、`alpha_sat_maps.png`（左下）

#### Slide 07 - 困难②：数据卫生事故（T 池泄漏）

- **Audience move**: 以为训练池是干净的 → 接受"域匹配池 516 个文件只有 12 张内容且全是测试集 T"，相关结论一度全部作废
- **Relationships**: 事故（泄漏）→ 影响（一批结论作废）→ 处置（内容级核查固化 + 池重建），构成因果链
- **Composition**: 左栏事故叙述与处置清单，右栏方图展示孪生簇核查结果，左右等高
- **Title**: 域匹配训练池 516 个文件，只有 12 张内容，而且全是测试集
- **Core message**: 训练池一度被测试集 T 污染，相关结果全部作废；处置是把内容级（md5/感知）泄漏核查固化进流程，并按谱匹配 + 对比度 + JPEG 配方重建训练池
- **Content**: 发现：augmented 池 516 文件仅 12 张内容且全 = 测试集 T · 影响：域匹配训练集全部泄漏，B→B 一类结果作废 · 处置：内容级泄漏核查固化为流程护栏；训练池按谱匹配 + 对比度 + JPEG 重编配方重建 · 教训：泄漏核查必须按内容，不能按文件名或统计口径
- **Fact IDs**: F06-leak-count
- **Images**: `cluster_leak_check_0915.png`

#### Slide 08 - 困难③：三个统计陷阱

- **Audience move**: 以为指标一读就懂 → 意识到低档 PSNR、单测试集结论与 seed 噪声都足以翻转判断
- **Relationships**: 三个陷阱彼此并列，各配一条本项目的实例；三者的共同教训是"口径必须先于结论"
- **Composition**: 三栏等宽卡片式区块，每栏一个陷阱 + 一句实例 + 一句对策；顶带一条规则线统一
- **Title**: 三个足以翻转判断的统计陷阱
- **Core message**: 低对比 GT 下"涂一片灰"就有 17.68 dB，低档 PSNR 必须并读常数基线；同一组件在 v2.1 为负、在扩增集为正，单测试集结论不可外推；seed 极差 0.4–0.6 dB 与判据门槛同量级，一律配对待检验 + 聚类 CI
- **Content**: 陷阱一 常数基线——低对比 GT 下平坦图 17.68 dB，低档 PSNR 必须并读基线 · 陷阱二 结论跟测试集走——w 组件 v2.1 负、扩增集正；R̂ 偏差 v2.1 偏亮、v23 偏暗 · 陷阱三 口径依赖——seed 极差 0.4–0.6 dB 与门槛同量级，一律配对 + 聚类 CI
- **Visualization**: 三栏并列的"陷阱 → 实例 → 对策"自定义版式，非数据对象，不设 key

### Part 3: 进展

#### Slide 09 - 进展①：分离器域适配成功（主结果）

- **Audience move**: 只记得零样本塌了 → 亲眼看到六列对照里内容被找回来，接受 +6.4 dB 的主结果
- **Relationships**: 四列方法（输入 / GT / 零样本 / pilot / 域适配）沿同一场景横向对照，逐列改善是唯一的阅读路径；三个 α 档的结论并列
- **Composition**: 上部一行判断句 + 三档数字带，下部满宽六列对照带；这是全片的证据高峰，密度可以最大
- **Title**: 域适配把 ᾱ=0.3 档从 12.05 拉到 18.5 dB
- **Core message**: 域适配把 ᾱ=0.3 档从 12.05 提到 18.5 dB（+6.4，plain 臂 2 seed 均值，四臂 18.3–18.7），结构相关 0.26 → 0.63，α=0.6/1.0 档 19.7/19.7 不低于官方权重
- **Content**: 方法 = 冻 FocalNet 主干 + 30k iter 域适配微调 + 干净数据管线 · 三档结果：ᾱ=1.0 19.7 / 0.6 19.7 / 0.3 18.5（raw PSNR，2 seed 均值） · 结构相关 0.26 → 0.63 · 归因：失败原本是内容级损伤，域适配修的是内容不是亮度
- **Fact IDs**: F08-main-result, F08-structure-corr
- **Images**: `rdt_s1_v21_t_alpha03.png`

#### Slide 10 - 进展②：低 α 专支与它真正的代价

- **Audience move**: 以为专支可以替代主臂 → 看清增益与代价分别落在哪个工作点区间，接受"专支"定位
- **Relationships**: 两套测试集的 Δ-α 曲线符号方向相反，构成直接对照；数字表与曲线是同一组事实的两种读法（并列）
- **Composition**: 左栏曲线主图，右栏数字表；表与图等高，图给形状、表给口径与 CI
- **Title**: 低 α 专支：增益与代价分别落在哪
- **Core message**: 扩增集上 freeze−plain 为 ᾱ=0.1 +1.75、0.2 +1.09、0.3 +0.52（聚类 CI 下界均 >0，0.3 档 +0.06 贴线）；ᾱ=1.0 反而 −0.36；v2.1 上 ᾱ=0.6/1.0 分别 −1.75/−0.94。同一组件的符号随测试集与工作点翻转——它不是一个独立增益，而是把工作点往低 α 搬的旋钮
- **Content**: 扩增集（48 场景，leak-free 重评）Δ = +1.31/+1.75/+1.09/+0.52/−0.36（ᾱ=0.05/0.1/0.2/0.3/1.0） · v2.1（12 场景）Δ = +0.07/−1.75/−0.94（ᾱ=0.3/0.6/1.0） · 0.3 档 CI 下界 +0.06，边际价值有限，价值主要在 ≤0.2 档 · 定位：低 α 专支，与主臂 plain 分工，不作替代
- **Visualization**: 逐 α 档增益的数字带（增益 + 聚类 CI 以文字呈现；曲线图已带 CI 误差棒，不另设表）
- **Fact IDs**: F09-specialist-gain, F09-v21-cost
- **Images**: `fig_delta_alpha.png`

#### Slide 11 - 进展③：把边界量化出来

- **Audience move**: 只关心"我们现在多少分" → 关心"这题最多能到哪、我们卡在哪一级"
- **Relationships**: 四条水平参考（常数基线 / 当前成绩 / R̂ 天花板 / 真值天花板）在同一刻度上形成层级对照，前两者近、后两者远
- **Composition**: 左栏分档能力图，右栏四条参考的含义与读法；结论句压在顶部
- **Title**: 可辨识性地图：天花板在哪，我们卡在哪级
- **Core message**: 平坦图常数基线 17.68 dB；α 可辨识性平台约 0.7；用我们训出的 R̂ 做除法只剩 17.2–22.3 dB，而真 R 能到 30.7–34.9——差距几乎全在 R̂ 与 α 上，不在出口
- **Content**: 平坦图常数基线 17.68 dB（v2.1 三档同值） · α 平台 ≈ 0.7（四独立估计器） · R̂ 除法 17.2 / 20.3 / 22.3 vs 真值天花板 30.7 / 34.1 / 34.9（ᾱ=0.3/0.6/1.0） · 学术意义：别人问"这题能到哪"，我们有分档答案
- **Fact IDs**: F10-baseline, F10-ceiling
- **Images**: `fig_boundary_map.png`

### Part 4: 负结果

#### Slide 12 - 负结果①：物理除法出口（本轮最关键的科学发现）

- **Audience move**: 期待"物理除法"能兑现上限 → 接受"E 与 G 完全等价、出口净效应只是工作点搬移"，并看到瓶颈的定位
- **Relationships**: E 臂与 G 臂几乎重合（等价）；反事实（同 β、换真值 α̃）跳升到天花板附近——三组对照锁定归因：机制没问题，α 精度是绑定约束
- **Composition**: 左栏给判决与归因结论，右栏放反事实对比图；顶部一行判决句横贯
- **Title**: 出口机制没问题，瓶颈是 α 的跨域校准
- **Core message**: 冻结主干 + 22k 参数出口头，E（α̃ 除法）与 G（自由增益）差异 ≤0.02 dB 完全等价，ᾱ=1.0 还掉 2.02 dB；但保留训练好的 β、只把 α̃ 换成真值，ᾱ=1.0 立刻 +4.26 dB（17.89 → 22.15），与训练前算的天花板 22.25 重合
- **Content**: 设计 = T̂=(1−β)·T̂_head + β·(I−R̂)/α̃，冻结主干 + 22k 参数出口头，E/G 各 2 seed · 判决：E≡G（|E−G| ≤ 0.02 dB），ᾱ=1.0 相对 plain −2.02 dB（CI [−2.29,−1.74]） · 反事实：同 β 换真值 α̃ → 22.15 / 21.18 / 18.85（ᾱ=1.0/0.6/0.3），ᾱ=1.0 单档 +4.26，与天花板 22.25 几乎重合 · 结论：出口线条件性冻结，重启条件 = α̃ 跨域误差够小（当前测试端场景均值 MAE 0.26/0.17/0.07，远高于 0.1 门）
- **Mathematical content**: \hat{T} = (1-\beta)\cdot\hat{T}_{head} + \beta\cdot(I-\hat{R})/\tilde{\alpha}
- **Fact IDs**: F11-EG-equiv, F11-counterfactual
- **Images**: `fig_exit_counterfactual.png`

#### Slide 13 - 负结果②：R̂ 是全局瓶颈

- **Audience move**: 以为提升点还在出口或损失函数 → 转向"R̂ 质量与 α 跨域校准"这两个真正的约束
- **Relationships**: 误差预算的层级落差（真 R → R̂）是主导项；R̂ 偏差跨域翻符号是同一瓶颈的第二重证据；两者共同指向 R 池规模
- **Composition**: 左栏机制说明与偏差现象，右栏误差预算图；底部一行 R 池规模的引出句
- **Title**: 瓶颈是 R̂：误差被 1/α 放大
- **Core message**: 真 R 时物理除法能到 30.7–34.9 dB，换成我们 23.6 dB 的 R̂ 只剩 17.2–22.3（落差 12.6–13.7 dB，低 α 被 1/α≈2.44 放大）；R̂ 偏差在 v2.1 偏亮、在扩增集偏暗——跨域翻符号，根因假设是 R 训练池只有 131 张
- **Content**: 误差预算：真 R 30.7/34.1/34.9 → 换 R̂ 17.2/20.3/22.3，落差 13.5/13.7/12.6 dB · 低 α 档 R̂ 误差被 1/α 放大（ᾱ=0.3 档 ᾱ_eff=0.410 → 2.44×） · R̂ 偏差跨域翻符号：v2.1 偏亮（+4~+15 灰阶，92% 同号）、v23 偏暗 · 根因假设：R 训练池只有 131 张
- **Fact IDs**: F12-budget-gap, F12-rhat-bias
- **Images**: `fig_error_budget.png`

### Part 5: 新方向与计划

#### Slide 14 - 方向 A：Mona PEFT 适配器（进行中）

- **Audience move**: 以为"全量微调不行就是没救" → 看到"全量掉分、适配器涨分"的先例，接受这是一次判别性实验
- **Relationships**: 先例（GFRRN：全量微调 26.70→25.35，Mona 适配器 27.33）与本项目现象（长训 R̂ 反降）同构，构成类比关系；两条可能结局（有效→特征侧 / 无效→数据侧）并列且都被预先承认
- **Composition**: 左栏写依据与同构性，右栏放 FocalNet Block 插入示意；底部一行预期区间与判别条件
- **Title**: Mona 适配器：桥接预训练语义与去反射语义
- **Core message**: 依据是 CVPR'26 GFRRN 的先例——全量微调掉分、Mona 适配器涨分，与本项目"长训 R̂ 反降"同构；已确认插入点（FocalNet 块内两个残差之后），实现与冒烟进行中。诚实预期 +0.2~+0.8 dB（点估 +0.5），无论结果都有信息量
- **Content**: 依据 = GFRRN 全量微调 26.70→25.35、Mona 适配器 27.33 · 同构性 = 本项目 pilot 长训 R̂ 反降 · 已落地 = 官方实现调研 + 插入点确认（FocalNet 块内两个残差后）+ 实现与冒烟中 · 判别性实验：有效 → 特征侧是瓶颈；无效 → 数据侧是瓶颈 · 预期诚实标注 +0.2~+0.8 dB（点估 +0.5）
- **Visualization**: FocalNet Block 结构 + 两处 Mona 插入点的自定义示意（非数据对象，不设 key）

#### Slide 15 - 方向 B：RRW 扩池（进行中）

- **Audience move**: 以为本地数据还有大片没开采 → 接受"本地已挖干净、余量在未下载场景"，认可扩池是性价比最高的杠杆
- **Relationships**: 601 → 131 的剔除构成解释了"池子为什么这么小"；本地已开采完毕与尚有未下载场景并列，共同支持扩池计划
- **Composition**: 左栏池构成图，右栏扩池计划数字与步骤，下部一条满宽现有 R 池预览带
- **Title**: R 池只有 131 张：本地已挖干净
- **Core message**: 601 个候选中只留下 131 张（剔除 226 冗余 + 169 孪生 + 75 与测试 R 同簇）；本地 RRW 的 10,508 帧 / 42 场景已全部开采，真正的余量是 107 个尚未下载的场景。目标把干净池从 131 扩到 287，同配方提取后过孪生簇安全网
- **Content**: 池构成：601 = 131 保留 + 226 冗余副本 + 169 孪生 + 75 同簇剔除 · 保留 131 的来源：rrw 场景层 17 + wsoft 114 · 本地 RRW：10,508 帧 / 42 场景 / 未用场景 = 0 · 扩池计划：新提取 160 窗 → 138 入池，目标 131 → 287（不含 alt），新池 ∩ 测试源 = 0 · 余量：HZ2–HZ6 + hf1/hf2 共 107 个未下载场景
- **Fact IDs**: F14-pool-composition, F14-expand-target
- **Images**: `fig_r_pool.png`（左栏）、`r_sources_preview_0915.png`（下部横带）

#### Slide 16 - 方向 C：结构锚 + 全膜协议（储备）

- **Audience move**: 以为 α 不可辨识就只能绕开 → 看到"用结构锚把 α 钉住"和"候选族 + 一次标定"两条可行协议
- **Relationships**: 边界台阶锚与配对不变量是两个独立的零训练抓手，并列；全膜（无参照）场景下"单帧不可辨"与"候选族 + 一次标定"构成问题与对策
- **Composition**: 上栏两个锚点各自的数据与用法，下栏全膜协议的两栏对照（协议 + 代价）；留白页密度
- **Title**: 用结构钉住 α，无参照场景交给一次标定
- **Core message**: 边界台阶锚 α̂ MAE 0.080（聚类 95% CI [0.059, 0.101]），多档配对不变量误差在无掩膜口径下仅 0.065%——两者都是零训练可得，下一步进训练做一致性损失；全膜（无参照）场景单帧不可辨，改用"候选族 + 一次标定"协议：oracle@k 扩档 k=7 为 20.90 dB，标定后选择悔值 +3.20 dB、top2 命中 86%，按采纳门仍需改进
- **Content**: 边界台阶锚：α̂ MAE 0.080（聚类 CI [0.059, 0.101]；R 真值下界 0.023），零训练 · 多档配对不变量：GT 掩膜口径 0.31%/0.25%，无掩膜口径 0.065% · 全膜协议：单帧不可辨 → 候选族 + 一次标定；oracle@k 20.90（k=7）、悔值 +3.20、top2 86%，采纳门 FAIL，需继续改进 · 新颖性：文献调研确认"单帧 + 硬边界 + 离散档 + 深衰减"无人占位（PSTM 为偏振路线且无开源）
- **Fact IDs**: F15-anchor-mae, F15-invariant
- **Visualization**: 候选族示意（同一图 5 档亮度的自定义版式，非数据对象，不设 key）

#### Slide 17 - 我的判断与计划

- **Audience move**: 听完一堆结果 → 带走三条明确判断与本周/下周/数据到位后的推进节奏，且知道风险在哪
- **Relationships**: 三条判断（主线 / 止损 / 新方向）彼此并列且互不冲突；时间线三段顺序推进；风险条目独立于判断之外
- **Composition**: 上栏三条判断并排，下栏时间线三段横排，页脚一行风险如实说；留白页密度
- **Title**: 三条判断与推进节奏
- **Core message**: 主线押"域适配 + 可辨识性地图"（12→18.5 的主结果 + 把边界量化成图），方法增量作为上行空间而非前提；物理除法出口在低 α 档判死并设明确重启条件；特征侧（Mona PEFT）与数据侧（RRW 扩池 131→287）两条腿并行验证，本周内出判决
- **Content**: 判断一 主线：论文押域适配 + 可辨识性地图，方法增量作上行空间 · 判断二 止损：出口在低 α 档判死，重启条件 = α̃ 跨域 MAE 足够小 · 判断三 新方向：特征侧与数据侧两条腿并行，本周内出判决 · 时间线：PEFT/扩池（本周）→ 结构锚进训练（下周）→ 实拍 sanity（数据到位即排） · 风险如实说：主结论均为合成域结论，实拍定量待补
- **Fact IDs**: F16-decisions
- **Visualization**: 三段甘特/时间线（自定义绘制，非数据对象，不设 key）

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under `notes/`
- **Content**: 以页面的最终 SVG 与 §IX 事实为准逐页撰写，不引入页面上没有的数字；每页给出：这一页要让它落地的判断、要念的数字及其口径、以及从上一页过渡过来的一句话。负结果页需明确说出"这条线被冻结了、重启条件是什么"，避免听众误以为项目停滞
- **Total duration**: 15–20 分钟（16 页）
- **Notes style**: conversational（组会讲解口吻：先给判断，再给数字与口径，口语化过渡）
- **Presentation purpose**: 按时间线汇报 9 月进展：先给基准与起点，再如实讲清三个困难，然后给主结果与两个负结果及其归因，最后落到下一阶段两条并行方向的判断与节奏。口径是"讲已做的决策与依据"，不向听众征求技术方向
