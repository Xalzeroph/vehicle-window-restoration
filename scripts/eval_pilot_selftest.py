#!/usr/bin/env python3
"""人评材料自测对账（2026-09-19，用户 n=1 非盲自测——非证据，仅作材料/判据可用性检查）。

把用户口述答案对照 key.json 翻译为 左/右/无偏好，算机器-人一致率与 trap 表现。
用法：python scripts/eval_pilot_selftest.py
"""
import csv, json

# 用户原话（2026-09-19 自测，V1）
ANSWERS = {
    'pair_01': ('左', '茶叶盒上仍有部分三脚架影子没去掉'),
    'pair_02': ('右', ''),
    'pair_03': ('无偏好', '没有看到明显区别'),
    'pair_04': ('左', '左边自然；两边都没去掉影子；右边试图去人影但没去掉，所以糊了一些'),
    'pair_05': ('无偏好', '两边都没去掉；右边试图去了但没有效果'),
    'pair_06': ('左', '都没去掉；左边去了一部分吧'),
    'pair_07': ('无偏好', '都没去掉'),
    'pair_08': ('无偏好', '都没去掉；左边人影去了一些，但把图案也去掉了一些'),
    'pair_09': ('无偏好', '都没有去掉，都很一般'),
    'pair_10': ('左', '左边去了很多，但糊掉了一些'),
    'pair_11': ('无偏好', '都很一般，没去掉'),
    'pair_12': ('无偏好', '能看出右边暗一点'),
    'pair_13': ('左', '左边去了很多，但人的膀臂没去掉'),
    'pair_14': ('无偏好', '都很一般没去掉'),
    'pair_15': ('右', '都很一般没去掉；右边在努力了，有一点点效果'),
    'pair_16': ('无偏好', '都一般没去掉'),
}

key = json.load(open('data/human_pilot_0919/key.json'))
rows, main, traps = [], [], []
for pid in sorted(key):
    v = key[pid]
    ans, note = ANSWERS[pid]
    human_side = {'左': 'left', '右': 'right', '无偏好': None}[ans]
    machine_side = None
    if v['machine'] == 'igm':
        machine_side = 'left' if v['V1_left'] == 'identity_gm' else 'right'
    elif v['machine'] == 'rdnet':
        machine_side = 'left' if v['V1_left'] == 'rdnet_gm' else 'right'
    rec = dict(pair=pid, kind=v['kind'], case=v['case'], med_d=v['med_d'],
               V1_left=v['V1_left'], machine=v['machine'], machine_side=machine_side,
               human=ans, human_side=human_side, note=note,
               match=('' if (machine_side is None or human_side is None)
                      else int(machine_side == human_side)))
    rows.append(rec)
    (main if v['kind'] == 'main' else traps).append(rec)

decisive = [r for r in main if r['human_side'] is not None]
match_n = sum(r['match'] for r in decisive)
model_side_n = sum(1 for r in decisive if
                   (r['human_side'] == 'left') == (r['V1_left'] == 'rdnet_gm'))
print('=== 主对（12）===')
for r in main:
    tag = {'1': '一致', '0': '不一致', '': '未表态'}[str(r['match'])]
    side = {'left': '左', 'right': '右', None: '无偏好'}[r['human_side']]
    mside = {'left': '左', 'right': '右', None: 'tie'}[r['machine_side']]
    print(f"{r['pair']} medD={r['med_d']:>5} 机器→{mside:<3} 你选 {side:<3} [{tag}]  {r['note']}")
print(f"\n有表态 {len(decisive)}/12，一致 {match_n}/{len(decisive)} = {match_n/max(len(decisive),1)*100:.0f}%")
print(f"全体主对（无偏好计不一致）: {match_n}/12 = {match_n/12*100:.0f}%")
print(f"表态中你选到模型侧的: {model_side_n}/{len(decisive)}")

print('\n=== trap（4）===')
for r in traps:
    side = {'left': '左', 'right': '右', None: '无偏好'}[r['human_side']]
    ok = '通过' if r['human_side'] is None else '假偏好!'
    print(f"{r['pair']} {r['kind']:<10} 你选 {side:<3} [{ok}]  {r['note']}")
print(f"trap 无偏好率: {sum(1 for r in traps if r['human_side'] is None)}/4")

with open('eval/eval_pilot_selftest.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('\n写出 eval/eval_pilot_selftest.csv（标注：n=1 非盲自测，非证据）')
