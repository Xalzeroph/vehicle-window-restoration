#!/usr/bin/env python3
"""协议 v2 修正补丁 2：把"仅记录"的门与"通过"的门在计数上分开（2026-10-12）

问题
----
补丁 1 把门3 改成 `gate(..., ok=True, note='仅记录')`。
`gate()` 内部 `G.append((name, val, '✅' if v else '❌'))`，
而 `npass = sum(1 for g in G if g[2] == '✅')` ——
**"仅记录"被算成了"通过"，虚高了分子**。分母也跟着变大，判决失真。

本补丁
----
给 `gate()` 增加 `record_only=False` 参数：
  - record_only=True：结果列显示 `📝`，**不计入 npass，也不计入分母**；
  - 判决行改为 `判决: npass/ntotal 通过（另有 nrecord 项仅记录）`。
调用处门3 由 `ok=True` 改为 `record_only=True`。

用法：
    python3 scripts/patch_qa_thresholds_v2b.py scripts/qa_capture.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

OLD_GATE = (
    "    def gate(name, val, lo=None, hi=None, ok=None, fmt='{:.4f}', note=''):\n"
    "        v = ok if ok is not None else ((lo is None or val >= lo) and (hi is None or val <= hi))\n"
    "        G.append((name, fmt.format(val) if val == val else 'nan', '✅' if v else '❌', note))\n"
    "        return v\n"
)
NEW_GATE = (
    "    def gate(name, val, lo=None, hi=None, ok=None, fmt='{:.4f}', note='',\n"
    "             record_only=False):\n"
    "        v = ok if ok is not None else ((lo is None or val >= lo) and (hi is None or val <= hi))\n"
    "        mark = '📝' if record_only else ('✅' if v else '❌')\n"
    "        G.append((name, fmt.format(val) if val == val else 'nan', mark, note))\n"
    "        return v\n"
)

OLD_NPASS = "    npass = sum(1 for g in G if g[2] == '✅')\n"
NEW_NPASS = (
    "    npass = sum(1 for g in G if g[2] == '✅')\n"
    "    nrecord = sum(1 for g in G if g[2] == '📝')\n"
    "    ntotal = len(G) - nrecord   # 分母只算真正判决的门\n"
)

OLD_CALL = (
    "    gate('门3 跨段偏移 cross_g_mean（信号，仅记录）', res['cross_g_mean'],\n"
    "         ok=True, note='v2：反射强度是信号，不设上限；逐通道值见明细')\n"
)
NEW_CALL = (
    "    gate('门3 跨段偏移 cross_g_mean（信号，仅记录）', res['cross_g_mean'],\n"
    "         record_only=True, note='v2：反射强度是信号，不设上限；逐通道值见明细')\n"
)

OLD_HDR1 = "    lines.append(f'## 判决: {npass}/{len(G)} 通过\\n')\n"
NEW_HDR1 = ("    lines.append(f'## 判决: {npass}/{ntotal} 通过'\n"
            "                 + (f'（另有 {nrecord} 项仅记录）' if nrecord else '') + '\\n')\n")

OLD_HDR2 = "    print(f'  判决: {npass}/{len(G)} 通过')\n"
NEW_HDR2 = ("    print(f'  判决: {npass}/{ntotal} 通过'\n"
            "          + (f'（另有 {nrecord} 项仅记录）' if nrecord else ''))\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('target')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    p = Path(args.target)
    src = p.read_text(encoding='utf-8')
    orig = src

    for old, new, tag in ((OLD_GATE, NEW_GATE, 'gate()'),
                          (OLD_NPASS, NEW_NPASS, 'npass'),
                          (OLD_CALL, NEW_CALL, '门3 调用'),
                          (OLD_HDR1, NEW_HDR1, '报告判决行'),
                          (OLD_HDR2, NEW_HDR2, '终端判决行')):
        if old not in src:
            print(f'[错] 找不到锚点：{tag}', file=sys.stderr)
            return 1
        src = src.replace(old, new, 1)

    if src == orig:
        print('[警告] 无改动', file=sys.stderr)
        return 1
    if args.dry_run:
        print(f'[dry-run] {p}（{len(orig)} -> {len(src)} 字节）')
        return 0

    bak = p.with_suffix(p.suffix + '.bak_v2b_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    shutil.copy2(p, bak)
    print(f'[备份] {bak}')
    p.write_text(src, encoding='utf-8')
    print(f'[写入] {p}（{len(orig)} -> {len(src)} 字节）')

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
