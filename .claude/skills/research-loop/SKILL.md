---
name: research-loop
description: >-
  本项目的研究循环工作流（P0 破题 → P1 方案 → P2 Opus 红队审核 → P3 执行 → P4 分析 →
  P5 写 md 汇报 → P6 确认后 git 提交）。当用户说"开新一轮 / 走工作流 / 研究循环 /
  按流程来"时调用，按其中阶段与三个必停门执行。
---

# 研究循环工作流（TintedGlass）

一轮 = 一个可独立回答的科学问题。全程中文交流；产物落到项目目录；GPU 遵守共享机守则（`nvidia-smi` 查空闲、`CUDA_VISIBLE_DEVICES` 钉卡）。

## 阶段与门

**P0 破题（人机研讨）**
- 复述用户的需求/发现的问题，澄清到可执行的一句话目标
- 与用户**预登记判决门**：什么数字算 PASS / FAIL（先写下再跑，事后不改口）
- 估成本（CPU/GPU 卡号与时长），写进本轮定义

**P1 方案（人机研讨）**
- 给出初步方案：步骤、用哪些脚本、数据从哪来、预期结果、主要风险
- ⛔ **GATE-1（必停）**：等用户确认方案定稿，才进入执行

**P2 Opus 红队审核（关键轮才开）**
- 触发条件：架构改动 / >1h GPU 预算 / 要进论文的结论 / 用户点名
- 启 Agent 子代理（model: opus），给自包含简报：目标、方案、判决门、数据路径
- 审核清单：内容级泄漏（`check_t_src_leak.py`）、对照缺失（常数/平凡基线、双测试集）、
  判决门可达性、成本估算、口径自洽
- 吸收意见后若有实质变更 → 回到 GATE-1 重新确认

**P3 执行（自动）**
- 用 `Makefile` 跑（`make help`）；建集后必跑 `make check-pool`（泄漏 exit 1 即停）
- 长任务 nohup 后台 + 守候任务通知；中间 ckpt 用 `make clean-ckpts` 清
- ⛔ **GATE-2（必停）**：启动 >30min GPU 任务前确认

**P4 分析（自动）**
- `make report` 出汇总表；区分"器质性结论 vs 噪声"；引用数字必须能溯源到文件
- 关键轮：Opus 审结果解读（数字引用、因果归因、口径），与 P2 同法

**P5 汇报（自动）**
- 写 `docs/YYYY.MM.DD_主题.md`（目标 / 判决门 / 命令 / 结果表 / 结论 / 下一步）
- 对话内 ≤10 行摘要汇报；图进 `fig/`

**P6 归档（必停）**
- ⛔ **GATE-3（必停）**：用户确认后 git commit
- commit 格式：`M.D 主题：要点` + bullet 正文（改了什么、为什么）+ `Co-Authored-By: Claude ...`
- 更新 memory 条目（结论、口径、待办）；数据/权重不入库（.gitignore 已覆盖），
  文档/CSV/图入库
- → 开始下一轮

## 项目标准护栏（每轮自动带上）
- 新数据源/新池 → 先 `scripts/check_t_src_leak.py`（**内容级**，按文件名排除无效）
- 评测面板必带**常数/平凡基线**；域相关结论必跑 **v2.1 + indomain 双测试集**
- 引用历史数字前查 `docs/` 修正框与 memory（有"泄漏作废"标记的不得引用）
- 命名：`data/train_sets/train_triplets_<轮>`、`runs/ckpt/fac_<配置>`、`eval/eval_iar_<配置>[_<tag>]`

## 工具速查
- 执行链：`Makefile`（help / check-pool / triplets / train / eval-all / report / clean-ckpts）
- 泄漏核查：`scripts/check_t_src_leak.py`；域统计：`scripts/domain_stats.py`
- 评测：`scripts/eval_iar.py`（输出 CSV + summary JSON）；汇总：`scripts/summarize_evals.py`
