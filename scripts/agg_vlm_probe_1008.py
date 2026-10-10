#!/usr/bin/env python3
"""汇总 VLM 探针 T1-T4 并按预登记门判决语义线生死（2026-10-08 交接第 4 节）。

预登记门（见 2026.10.08_今日归档与会话交接.md 第 39 行）：
  T1 命中率 >= 0.5 且 IoU >= 0.3
  T2 准确率 >= 0.75（n=30, p<0.01）
  T3 成立 -> 走"VLM 描述作弱监督"
  全部不成立 -> 语义线正式关闭
  T4 必须显著优于 DINOv2 的 0.33-0.61（协议可比性存疑，单独标注）
用法：python3 scripts/agg_vlm_probe_1008.py eval/vlm_probe_qwen.csv eval/vlm_probe_smolvlm.csv
"""
import csv, json, math, os, sys

# DINOv2 自监督基线（docs/2026.10.08_反射洞重定义与自监督察觉探针.md 第 3 节）
# 列为：种子相似度最佳 IoU（乐观上界，弱监督）/ 无监督聚类最佳 IoU
DINO = {
    'apr 宽松 x>=.25,y>=.75': (0.611, 0.347),
    'apr 紧  x>=.50,y>=.75': (0.502, 0.246),
    'may 宽松':              (0.388, 0.171),
    'may 紧':                (0.333, 0.194),
}
DINO_SEED_RANGE = (0.333, 0.611)


def binom_tail(k, n, p):
    """P(X >= k), X~Bin(n,p) —— 单侧精确检验。"""
    return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z*z/n
    c = (ph + z*z/(2*n)) / d
    h = z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))


def nums(rows, key):
    out = []
    for r in rows:
        v = r.get(key, '')
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def mean(v):
    return sum(v)/len(v) if v else float('nan')


def summarize(path):
    with open(path, encoding='utf-8-sig') as fh:
        rows = [r for r in csv.DictReader(fh) if r.get('key')]
    if not rows:
        return None
    model = rows[0].get('model', '?')
    n = len(rows)
    s = {'model': model, 'n': n, 'path': path}

    # ---- T1 定位：命中率(recall) + IoU ----
    iou_v, rec_v, prec_v = nums(rows, 'T1_iou'), nums(rows, 'T1_rec'), nums(rows, 'T1_prec')
    s['T1_n'] = len(iou_v)
    s['T1_iou'] = mean(iou_v)
    s['T1_hit'] = mean(rec_v)          # 命中率 = 用户 ROI 中被预测覆盖的比例
    s['T1_prec'] = mean(prec_v)
    s['T1_iou_median'] = sorted(iou_v)[len(iou_v)//2] if iou_v else float('nan')
    s['T1_iou_ge_03'] = sum(1 for v in iou_v if v >= 0.3)
    s['T1_empty'] = sum(1 for r in rows if not r.get('T1_pred', '').strip())  # 答"看不出反射"

    # ---- T2 判别 ----
    cor = nums(rows, 'T2_correct')
    s['T2_n'] = len(cor)
    s['T2_acc'] = mean(cor)
    s['T2_k'] = int(sum(cor))
    s['T2_ci'] = wilson(s['T2_k'], len(cor)) if cor else (0, 0)
    s['T2_p'] = binom_tail(s['T2_k'], len(cor), 0.5) if cor else float('nan')

    # ---- T3 语义一致性 ----
    hit = nums(rows, 'T3_hit')
    s['T3_n'] = len(hit)
    s['T3_acc'] = mean(hit)
    s['T3_k'] = int(sum(hit))
    s['T3_missing'] = sum(1 for r in rows if 'T3_hit' not in r)  # T3_txt 为 ERR 的行

    # ---- T4 对照 DINOv2 ----
    s['T4_dino_seed_range'] = DINO_SEED_RANGE
    s['T4_beats_dino_min'] = bool(iou_v) and s['T1_iou'] > DINO_SEED_RANGE[0]
    s['T4_beats_dino_max'] = bool(iou_v) and s['T1_iou'] > DINO_SEED_RANGE[1]
    # 分域统计（apr/may），用于与 DINO 分域对照
    for dom in ('apr', 'may'):
        sub = [float(r['T1_iou']) for r in rows
               if r.get('key', '').startswith(dom) and r.get('T1_iou') not in (None, '')]
        s[f'T1_iou_{dom}'] = mean(sub) if sub else float('nan')
        s[f'T1_n_{dom}'] = len(sub)

    # ---- 门判决 ----
    g1 = (s['T1_hit'] >= 0.5) and (s['T1_iou'] >= 0.3)
    g2 = (s['T2_acc'] >= 0.75) and (s['T2_p'] < 0.01)
    g3 = s['T3_acc'] >= 0.5
    s['gate_T1'] = g1
    s['gate_T2'] = g2
    s['gate_T3'] = g3
    if g1 or g2:
        s['verdict'] = '存活（T1/T2 至少一门成立 → 语义线保留）'
    elif g3:
        s['verdict'] = '仅 T3 成立 → 降级为"VLM 描述作弱监督"'
    else:
        s['verdict'] = '全部不成立 → 语义线正式关闭'
    return s


def show(s):
    print(f"\n{'='*66}")
    print(f"  {s['model']}  (n={s['n']}, {s['path']})")
    print('='*66)
    print(f"  T1 定位   n={s['T1_n']}  IoU均值 {s['T1_iou']:.3f} (中位 {s['T1_iou_median']:.3f})"
          f"  命中率 {s['T1_hit']:.3f}  精确率 {s['T1_prec']:.3f}")
    print(f"            IoU>=0.3 的帧 {s['T1_iou_ge_03']}/{s['T1_n']} | 答'看不出反射'的帧 {s['T1_empty']}")
    print(f"            分域: apr {s['T1_iou_apr']:.3f} (n={s['T1_n_apr']})  may {s['T1_iou_may']:.3f} (n={s['T1_n_may']})")
    print(f"  T2 判别   {s['T2_k']}/{s['T2_n']} = {s['T2_acc']:.3f}"
          f"  95%CI [{s['T2_ci'][0]:.3f},{s['T2_ci'][1]:.3f}]  单侧 p={s['T2_p']:.2e}")
    print(f"  T3 语义   {s['T3_k']}/{s['T3_n']} = {s['T3_acc']:.3f}"
          + (f"  （{s['T3_missing']} 帧无 T3 结果）" if s['T3_missing'] else ''))
    print(f"  T4 对照   T1_IoU {s['T1_iou']:.3f} vs DINOv2 种子相似度 {DINO_SEED_RANGE[0]:.3f}-{DINO_SEED_RANGE[1]:.3f}"
          f"  → {'超过下界' if s['T4_beats_dino_min'] else '未达下界'}"
          f" / {'超过上界' if s['T4_beats_dino_max'] else '未达上界'}")
    print(f"  门判决    T1(>=0.5命中 & >=0.3IoU) {'PASS' if s['gate_T1'] else 'FAIL'}"
          f" | T2(>=0.75 & p<0.01) {'PASS' if s['gate_T2'] else 'FAIL'}"
          f" | T3 {'PASS' if s['gate_T3'] else 'FAIL'}")
    print(f"  ==> {s['verdict']}")


if __name__ == '__main__':
    paths = sys.argv[1:] or ['eval/vlm_probe_qwen.csv', 'eval/vlm_probe_smolvlm.csv']
    all_s = []
    for p in paths:
        if not os.path.exists(p):
            print(f'[skip] 不存在: {p}')
            continue
        s = summarize(p)
        if s:
            show(s)
            all_s.append(s)
    if len(all_s) > 1:
        print(f"\n{'='*66}\n  两模型合并口径\n{'='*66}")
        for k in ('T1_iou', 'T1_hit', 'T2_acc', 'T3_acc'):
            vals = [s[k] for s in all_s if s.get('n')]
            print(f"  {k:8s} 均值 {mean(vals):.3f}   范围 [{min(vals):.3f}, {max(vals):.3f}]")
        best_t1 = max(s['T1_iou'] for s in all_s)
        print(f"  最强 T1_IoU {best_t1:.3f} vs DINOv2 {DINO_SEED_RANGE[0]:.3f}-{DINO_SEED_RANGE[1]:.3f}"
              f" → {'仍超过上界' if best_t1 > DINO_SEED_RANGE[1] else ('仅落在区间内' if best_t1 > DINO_SEED_RANGE[0] else '全面低于下界')}")
    print('\n[提示] T4 协议可比性存疑：DINOv2 的 0.33-0.61 是"用洞内点作种子扫出的乐观上界(弱监督)"，'
          '而 T1 是"无任何种子的 5x5 格号预测"，两者不是同一协议，不能直接比大小。')
