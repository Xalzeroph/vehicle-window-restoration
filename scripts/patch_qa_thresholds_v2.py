#!/usr/bin/env python3
"""协议 v2 门限语义修正（交接页 §8 待办 3）

背景
----
`qa_capture.py` 的门限是为**旧数据（AE/AWB 未锁）**设的。旧数据的 `g` 量的是
"相机动了多少"（光度漂移），所以对它设 `|g−1| ≤ 0.02` 是对的。
但新拍数据（2026-10-11 起手动锁 ISO/快门/WB）里，**段内 `g` 已经量的是采集纪律，
跨段 `g` 才量反射强度**——后者是**信号，不是缺陷**。原脚本把跨段的
`cross_g_mean` 当缺陷看，会误伤好数据。

改动（只改判决口径，不动任何测量）
--------------------------------
1. 新增严格门：**门1b 反射段逐通道 \|g−1\|**（`drift_refl_ch`，阈值 0.05）
   —— 单通道偏色即使三通道均值正常也应报警。
2. 门3 由"跨段一致性(cross_cv ≤0.15)"改为**只记录**：
   跨段偏移 `cross_g_mean` / `cross_g_chan` 逐通道记录进报告，
   **判决行改为"记录（反射是信号，不设上限）"**，永远计为 ✅ 但不计入通过数，
   避免"好数据因为反射强而挂门"。
3. 报告数值明细区把 `cross_g_mean`/`cross_g_chan` 标注为**信号**。

用法（在服务器上执行）：
    python3 scripts/patch_qa_thresholds_v2.py scripts/qa_capture.py
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


OLD_GATES_ANCHOR = "    gate('门3 跨段偏移逐帧变异系数', res['cross_cv'], hi=0.15,\n" \
                   "         note='≤0.15=场景级恒定偏移，可整体吸收')\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('target', help='qa_capture.py 路径')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-backup', action='store_true')
    args = ap.parse_args()

    p = Path(args.target)
    src = p.read_text(encoding='utf-8')
    orig = src

    # ---- 改动 1：段内漂移再加一条逐通道严格门 ----
    anchor1 = ("    gate('门1 反射段内漂移 |g−1| 中位', res['drift_refl'], hi=0.02,\n"
               "         note='AE/AWB 是否锁死（旧数据 0.1749）')\n")
    if anchor1 not in src:
        print('[错] 找不到门1锚点，目标文件可能已被改过', file=sys.stderr)
        return 1
    new1 = anchor1 + (
        "    # v2 新增：单通道偏色。三通道均值正常但某一通道偏，同样说明 AWB 没锁死。\n"
        "    gate('门1b 反射段逐通道 |g−1|', res.get('drift_refl_ch', float('nan')), hi=0.05,\n"
        "         note='v2 新增：抓单通道白平衡漂移（旧数据 0.1749）')\n"
    )
    src = src.replace(anchor1, new1, 1)

    # ---- 改动 2：门3 从"判决"降为"记录" ----
    if OLD_GATES_ANCHOR not in src:
        print('[错] 找不到门3锚点，目标文件可能已被改过', file=sys.stderr)
        return 1
    new3 = ("    # v2 改：跨段 g 现在是**反射强度信号**，不是采集缺陷，因此只记录不判决。\n"
            "    # （旧数据里 g 量的是相机光度漂移，那时设上限才对。）\n"
            "    gate('门3 跨段偏移 cross_g_mean（信号，仅记录）', res['cross_g_mean'],\n"
            "         ok=True, note='v2：反射强度是信号，不设上限；逐通道值见明细')\n")
    src = src.replace(OLD_GATES_ANCHOR, new3, 1)

    # ---- 改动 3：补上逐通道段内漂移的测量 ----
    anchor_meas = ("    d_refl, c_refl = in_seg_drift(refl)\n"
                   "    d_cloth, c_cloth = in_seg_drift(cloth)\n"
                   "    res['drift_refl'] = d_refl; res['drift_cloth'] = d_cloth\n"
                   "    res['chan_refl'] = c_refl; res['chan_cloth'] = c_cloth\n")
    if anchor_meas not in src:
        print('[错] 找不到段内漂移测量锚点', file=sys.stderr)
        return 1
    new_meas = anchor_meas + (
        "\n"
        "    # v2 新增：段内**逐通道** |g−1| 最大值（抓单通道白平衡漂移）\n"
        "    def in_seg_drift_ch(seg, max_pts=20):\n"
        "        if len(seg) < 3:\n"
        "            return float('nan')\n"
        "        pick = seg[::max(1, len(seg) // max_pts)]\n"
        "        cs = load_crops(pick)\n"
        "        if len(cs) < 3:\n"
        "            return float('nan')\n"
        "        ref = cs[0]\n"
        "        gs = np.array([robust_gain(x, ref) for x in cs[1:]])\n"
        "        return float(np.median(np.abs(gs - 1).max(axis=1)))\n"
        "\n"
        "    res['drift_refl_ch'] = in_seg_drift_ch(refl)\n"
        "    res['drift_cloth_ch'] = in_seg_drift_ch(cloth)\n"
    )
    src = src.replace(anchor_meas, new_meas, 1)

    # ---- 改动 4：明细区把跨段量标成信号 ----
    src = src.replace(
        "    for k in ('drift_refl', 'drift_cloth', 'chan_refl', 'chan_cloth',\n"
        "              'cross_g_mean', 'cross_g_chan', 'cross_cv', 'r_obs',",
        "    for k in ('drift_refl', 'drift_refl_ch', 'drift_cloth', 'drift_cloth_ch',\n"
        "              'chan_refl', 'chan_cloth',\n"
        "              'cross_g_mean', 'cross_g_chan', 'cross_cv', 'r_obs',", 1)

    # 明细区标注
    src = src.replace(
        "    lines.append('## 数值明细\\n')",
        "    lines.append('## 数值明细\\n')\n"
        "    lines.append('> v2 口径：`drift_*` = 采集纪律（越小越好，有门限）；'\n"
        "                 '`cross_g_*` / `r_obs` = **反射信号**（越大越有东西可学，不设上限）。\\n')",
        1)

    # ---- 改动 5：文档字符串里的门限说明同步 ----
    src = src.replace(
        "    5) 门 3  跨段一致性    —— 偏移是\"场景级恒定\"还是\"逐帧乱跳\"",
        "    5) 门 3  跨段偏移      —— **仅记录**（v2：反射强度是信号，不判决）", 1)
    src = src.replace(
        "    3) 门 1  段内漂移      —— AE/AWB 是否真的锁死",
        "    3) 门 1  段内漂移      —— AE/AWB 是否真的锁死（v2 增设门 1b：逐通道）", 1)

    if src == orig:
        print('[警告] 未发生任何改动', file=sys.stderr)
        return 1

    if args.dry_run:
        print(f'[dry-run] 将修改 {p}（{len(orig)} -> {len(src)} 字节）')
        return 0

    if not args.no_backup:
        bak = p.with_suffix(p.suffix + '.bak_v2_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        shutil.copy2(p, bak)
        print(f'[备份] {bak}')
    p.write_text(src, encoding='utf-8')
    print(f'[写入] {p}（{len(orig)} -> {len(src)} 字节）')

    # 语法自检
    import py_compile
    try:
        py_compile.compile(str(p), doraise=True)
        print('[语法] OK')
    except py_compile.PyCompileError as e:
        print(f'[语法] 失败：{e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
