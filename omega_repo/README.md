# omega_repo — 贴膜玻璃 SIRR 流程的唯一代码源

> 建立于 2026-10-27。起因：原仓库存在 **3 套 Ω 定义、22 个 hole_*.py 探索版本、
> 9 组同名不同义的函数**，论文里"洞区"在不同章节指的是不同像素集合。

## 三条铁律

1. **一个名字，一种语义。** 语义不同必须改名，不允许同名两份。
2. **Ω 只有一份实现**：`omega/canonical.py::omega_of`。其它定义一律进 `legacy/`，
   调用方必须在输出里标注用了哪一个。
3. **`archive/` 不参与任何 import。** 仅作追溯，禁止被 `train/ infer/ eval/ data_build/` 引用。

## 目录

| 目录 | 内容 | 可否被引用 |
|---|---|---|
| `omega/` | **唯一权威 Ω 定义** | 是 |
| `train/` | 微调入口 | 是 |
| `infer/` | 本方法推理 + 四个公开基线封装 | 是 |
| `eval/` | 基准出表 | 是 |
| `data_build/` | 建测试集 / 建训练池 / 拆域补 C_I | 是 |
| `legacy/` | 旧 Ω 变体（改名 + 警告） | 仅复现旧数字 |
| `archive/hole_exploration/` | 22 个 `hole_*.py` | **禁止** |
| `archive/rejected_dino/` | DINOv2 掩码（判废） | **禁止** |

## Ω 的语义（读之前必须知道）

`omega_of` 检出的是**遮挡物 footprint**（反射缺失 + 照度下降），
**不是**"纯反射缺失" —— 单帧统计上二者不可分。
权威物理表述见 `docs/2026.10.26_物理模型_最终版_遮挡阴影.md`。

## 已知限制（不要偷偷"修好"）

- **L1** 只有全局增益 `s = q05(I/T)`，无法表示空间增益变化（实测洞内外差 26%），
  因此在亮暗交界处会点火（5 月数据有 48% 的 Ω 落在画面边框内）。
- **L2** 没有 `gT` 判据，无法区分"阴影"与"反射缺失"。

## 冒烟

```bash
cd omega_repo
python -c "from omega.canonical import omega_of; print(omega_of.__doc__[:80])"
```

验收基线：`canonical` 与 `data/train_sets/real_pairs_1026/omega/` 内掩码的 IoU 必须为 **1.0000**。
