# OPENSOURCE_TODO —— 开源前必须做完的事

> 2026-10-27 状态：代码已收敛到本目录（可离线、路径可覆盖、语法全通过），
> 但**还不能直接对外发布**。下面按"阻塞 / 需决策 / 建议"三档列清，每条给出验收标准。

---

## 0. 已完成（本次）

| 项 | 证据 |
|---|---|
| 代码单点收敛：`omega_repo` 铺平 + RDNet fork + scripts 子集 | 120 个代码文件，`MANIFEST.md` §1 |
| Ω 权威实现独立成模块（`omega/canonical.py`） | 与 `real_pairs_1026/omega` 掩码 IoU = 1.0000（服务器实测） |
| 判废清单落纸并归档（DINO 两套、19 个 `hole_*`、Otsu 掩码） | `DEPRECATED.md` |
| 路径参数化：37 处 / 25 个文件，默认值 = 服务器路径 | `TINTED_ROOT / TINTED_THIRD_PARTY / TINTED_PYTHON` |
| 语法复检：发布树 66 个 py 全部 `py_compile` 通过 | 2026-10-27 |
| **铺平后能起指挥**：`train/finetune_rdnet_s2.py --help`、`data_build/build_train_real_pairs.py --help`、`infer/eval_arm_1019.py --help`、`eval/table_final_1019.py --help` 全通过（在服务器 `/tmp/code_verify` 实测；补了 2 处跨目录 `sys.path`） | 2026-10-27 |
| **B1 的"值相等"前提已实测**：`canonical.omega_of` 对 `real_pairs_1026` 全部 **82 对** IoU = **1.0000**（min/median/mean/max 全 1.0，闸门 0 次触发）⇒ 接线不会改变掩码 | 2026-10-27 |
| 内容级去重：删 15 个逐字节相同副本，重复组 = 0 | `MANIFEST.md` §2 |

## 1. 阻塞项（不做完不能公开）

| # | 事项 | 为什么阻塞 | 验收标准 |
|---|---|---|---|
| **B1** | **Ω 接线**：`build_train_real_pairs.py` / `build_pairs_1019split.py` / `build_train_omega_0920.py` 仍 `from eval_g2_baseline import omega_of`（删减版），`omega/canonical.py` 无人 import | 对外说"Ω 只有一份实现"但代码里是两份——这正是审计要消灭的问题。**风险已实测很低：canonical 与训练池掩码 82/82 对 IoU=1.0** | 三处 import 改指 `omega.canonical`；重跑一次建池，掩码与旧版 IoU=1.0；或明确声明"训练用删减版、评测用 Otsu"并写进论文 |
| **B2** | **许可**：`RDNet/` 是 Zhao et al. CVPR'25 的 fork，**上游 LICENSE 未随仓库带过来** | 无许可证的代码不能发布 | 回上游仓库取 LICENSE（及其对本仓的约束）；`README.md` §8 补署名 |
| **B3** | **数据发布方案**：631 对真实配对 + `real_pair_1019` 15 场景 + `raw_1019` | 没有数据，"复现"只剩空壳 | 二选一：① 发布数据（含 `splits/split_v1.json`、md5 清单、Ω 掩码）；② 只发布**采集协议**（`../2026.9.22_自拍配对数据_拍摄规格.md`）+ 评测脚本，声明数据可申请 |
| **B4** | **隐私**：所有数据由拍摄者手持手机隔车窗拍摄，**画面里有拍摄者本人、行人和可能车牌** | 公开发布前必须处理 | 人脸/车牌模糊化，或在数据协议中写明"仅研究用途、需签署使用协议"；`datasheet` 写清采集环境与同意方式 |
| **B5** | **权重**：`rdnet.pth`（官方）、`cls_model.pth`、`focal.pth`、我们训练的 `ckpt_final.pth` | 复现评测必须能拿到基线 | 官方权重回链上游 release；本项目 ckpt 发布到 release/HF 并在 README 给出校验和 |
| **B6** | **端到端 dry-run**：在一份干净 checkout 上，只用 `TINTED_ROOT` + 公开数据 + 官方权重，跑通 §6 的四步 | 目前所有命令都在服务器路径下跑的 | 记录一次完整日志（含耗时、显存峰值、PSNR 表格与论文数字一致） |

## 2. 需决策（影响论文与代码口径）

| # | 决策 | 选项 |
|---|---|---|
| **D2** | 基准表的 Δ(洞−非洞) 是否换成 `omega_of` | 换 → 7 个公开方法的 Δgap 全部重算；不换 → 论文与 README 都必须注明"两套掩码" |
| **D3** | 训练脚本"复制"还是"移动"进新仓 | 已按**复制**处理（服务器原 `scripts/` 未动，四臂可能仍在跑） |
| **D4** | `eval/` 是否保留 legacy Otsu 定义 | 建议：改名 `legacy_hole_ratio_otsu`，输出里强制打印警告 |

## 3. 建议（不阻塞，但会让复现体验好很多）

| # | 事项 | 收益 |
|---|---|---|
| S1 | `common/` 真正落地：`luma`（8 版）、`psnr/ssim`、`affine`、`grad_mag` 统一 | 现在这些仍散在各脚本；审计 §3 的 9 组同名不同义只解决了 Ω 那一组 |
| S2 | 冒烟测试脚本 `smoke_test.sh`：`import` 全模块 + `canonical` 对 3 张样例的掩码哈希 + 官方权重跑 1 张图 | 把"能不能跑"从口头变成 CI |
| S3 | `Dockerfile` / `environment.yml`（本目录只有实测版本号） | 10 GB 卡 + CUDA 12.6 的环境较难复刻 |
| S4 | `datasheet`（采集日、场景数、行人出现频率、已知缺陷：每场景单张 GT 且与输入中位相隔 374 帧） | 审稿人和复现者都会问 |
| S5 | 清理 3 处 docstring 里的 `/DataDisk` 残留；`infer/eval_1019_multi.py` 的 usage 示例与实际路径对齐 | 小事，但显得干净 |

## 4. 发布前最后一遍清单

```text
[ ] B1 Ω 接线           [ ] B4 隐私处理
[ ] B2 上游 LICENSE     [ ] B5 权重与校验和
[ ] B3 数据方案         [ ] B6 端到端 dry-run
[ ] D2 掩码口径决策     [ ] README §6 命令逐条实测
[ ] 删掉 local_scripts/ local_tools/ archive/server_scripts_2026_10/（非发布内容）
[ ] 仓库根补 LICENSE、CITATION、.gitattributes
```
