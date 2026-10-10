#!/usr/bin/env python3
"""取帧距离诊断：GT 与 I 应该隔多少帧？（2026-10-08）

分解两个随"间隔"增长的误差：
  (a) AE 曝光漂移   —— 用 g(I_t, I_ref) 随 t 的偏离量刻画
  (b) 场景自身变化  —— 用"无反射段内部"帧间差异刻画（该段无反射，差异只能是场景/噪声）

做法：在每个场景的**反射段**内，以某帧为参考，量
        g 漂移 |g(t)-1|        （曝光）
        稳健残差 median|I_t - I_ref|（场景变化 + 曝光）
      再对**黑布段内部**做同样的事，作为"无反射时的变化底噪"。
      比较两者，判断"靠多近"才安全。

用法: python3 scripts/probe_frame_gap.py
"""
import os, re, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_protocol import robust_gain          # noqa: E402

SRC = os.path.join(PROJ, 'data/My_Reflection_Videos_cropped')


def imread_any(p):
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def parse_labels(p):
    txt = open(p, encoding='utf-8', errors='ignore').read()
    ent = [(m.group(1), m.group(2).strip())
           for m in re.finditer(r'^(\d{6})\s*(.*)$', txt, re.M)]
    refl = [e for e in ent if '反' in e[1] and '无' not in e[1] and '不' not in e[1]]
    norefl = [e for e in ent if '无' in e[1] and '反' in e[1] and '补' not in e[1] and '加' not in e[1]]
    return ent, (refl[0][0] if refl else None), (norefl[0][0] if norefl else None)


def main():
    skipped = []
    print(f"{'场景':<26}{'反射帧':>8}{'黑布帧':>8}{'间隔':>7}{'g(跨黑)':>10}"
          f"{'反射段内':>10}{'黑布段内':>10}   结论")
    for sc in sorted(d for d in os.listdir(SRC) if d.startswith('VID_')):
        d = os.path.join(SRC, sc)
        lp = os.path.join(d, 'label.txt')
        if not os.path.exists(lp):
            continue
        ent, r0, n0 = parse_labels(lp)
        if not (r0 and n0):
            # apr 批次 label.txt 只有一行裸帧号，无段落标注 → 无法做跨黑诊断
            skipped.append((sc, len(ent)))
            continue
        r0, n0 = int(r0), int(n0)
        nums = sorted(int(f[6:-4]) for f in os.listdir(d) if f.startswith('frame_'))
        if not nums:
            continue
        ref = imread_any(os.path.join(d, f'frame_{r0}.jpg'))
        gtb = imread_any(os.path.join(d, f'frame_{n0}.jpg'))
        if ref is None or gtb is None or ref.shape != gtb.shape:
            continue
        g_cross = float(np.atleast_1d(robust_gain(gtb, ref)).mean())

        # 反射段内漂移：取 r0 起、n0 之前的所有可用帧
        refl_nums = [n for n in nums if r0 <= n < n0]
        drifts = []
        for n in refl_nums:
            x = imread_any(os.path.join(d, f'frame_{n:06d}.jpg'))
            if x is None or x.shape != ref.shape:
                continue
            g = float(np.atleast_1d(robust_gain(x, ref)).mean())
            drifts.append(abs(g - 1))
        # 黑布段内漂移：n0 之后的帧
        post = [n for n in nums if n > n0]
        drifts2 = []
        for n in post:
            x = imread_any(os.path.join(d, f'frame_{n:06d}.jpg'))
            if x is None or x.shape != gtb.shape:
                continue
            g = float(np.atleast_1d(robust_gain(x, gtb)).mean())
            drifts2.append(abs(g - 1))

        m1 = np.median(drifts) if drifts else float('nan')
        m2 = np.median(drifts2) if drifts2 else float('nan')
        # 判断：跨段漂移是否远大于段内漂移
        if not np.isnan(m1) and not np.isnan(m2):
            seg = max(m1, m2)
            verdict = ('跨段漂移可忽略' if abs(g_cross - 1) <= max(0.03, 1.5 * seg)
                       else '跨段漂移显著 > 段内')
        else:
            verdict = '数据不足'
        print(f"{sc:<26}{r0:>8}{n0:>8}{n0-r0:>7}{g_cross:>10.3f}"
              f"{m1:>10.4f}{m2:>10.4f}   {verdict}")

    if skipped:
        print(f"\n[跳过 {len(skipped)} 个场景：label.txt 无段落标注（只有裸帧号）]")
        for sc, ne in skipped:
            print(f"    {sc}  标注条数={ne}")

    print()
    print('读法：')
    print('  g(I起点→黑布) 越接近 1 越好；它 = 跨黑的曝光跳变')
    print('  反射段内漂移 / 黑布段内漂移 = 该段自身的 AE 抖动（与"跨黑"无关的底噪）')
    print('  若跨段 >> 段内 → 说明"跨黑"本身带来曝光跳变，取帧再近也压不掉')


if __name__ == '__main__':
    main()
