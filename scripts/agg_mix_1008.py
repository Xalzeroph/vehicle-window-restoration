#!/usr/bin/env python3
"""汇总混训 6 臂的判决指标（2026-10-08）。

读 runs/mix_1008/eval/<arm>_on_<dom>.csv，逐臂抽其自身臂列，输出对比表。
判据（项目口径，eval_g2_baseline 内定义）:
  pass1    = rho >= 0.5                (Ω 内相对参考的校正率)
  pass4    = med_ratio <= 1.0          (Ω 内中位残差/输入残差)
  ref_gain = LPIPS 改善 >= 0.02        (Ω 外扩带)
  out_dPSNR= Ω 外 PSNR 变化 >= -0.2
  pass_all = 四者合取
用法: python3 scripts/agg_mix_1008.py
"""
import csv, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
EV = os.path.join(PROJ, 'runs/mix_1008/eval')

ARMS = ['b1_apr_may', 'b2_may_apr', 'm3_aprjun_may', 'm4_mayjun_apr',
        'm5_aprmay_jun', 'm6_aprjun_gm']
# 臂 -> (训练域, 测试域, 类型)
DESC = {
    'b1_apr_may':    ('apr',     'may', '基线 单域'),
    'b2_may_apr':    ('may',     'apr', '基线 单域'),
    'm3_aprjun_may': ('apr+jun', 'may', '★混训'),
    'm4_mayjun_apr': ('may+jun', 'apr', '★混训'),
    'm5_aprmay_jun': ('apr+may', 'jun', '★混训'),
    'm6_aprjun_gm':  ('apr+jun', 'may', '★混训(gm)'),
}


def load(f):
    with open(f, encoding='utf-8-sig') as fh:
        return [r for r in csv.DictReader(fh) if r.get('case')]


def num(rows, key):
    out = []
    for r in rows:
        v = r.get(key, '')
        if v not in ('', None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def main():
    files = sorted(glob.glob(os.path.join(EV, '*_on_*.csv')))
    if not files:
        print(f'[错] {EV} 下没有评测 CSV')
        return
    print(f'读入 {len(files)} 个评测文件\n')

    hdr = f"{'臂':<16}{'训练域':<9}{'测试域':<7}{'n':>4}{'ρ均值':>8}{'①通过':>8}{'④通过':>8}{'ref_gain':>10}{'③dPSNR':>9}{'全通过':>8}  类型"
    print(hdr)
    print('-' * len(hdr))

    summary = {}
    for arm in ARMS:
        f = os.path.join(EV, f'{arm}_on_{DESC[arm][1]}.csv')
        if not os.path.exists(f):
            print(f'{arm:<16} [缺文件] {os.path.basename(f)}')
            continue
        rows = load(f)
        # 只统计 den_ok 的帧（参考对比不足的帧判据无效）
        ok = [r for r in rows if r.get('den_ok') == '1']
        rho = num(ok, f'{arm}_rho')
        p1 = num(ok, f'{arm}_pass1')
        p4 = num(ok, f'{arm}_pass4')
        rg = num(ok, f'{arm}_ref_gain')
        od = num(ok, f'{arm}_out_dPSNR')
        pa = num(ok, f'{arm}_pass_all')
        tr, te, kind = DESC[arm]
        summary[arm] = dict(rho=float(np.mean(rho)) if rho else float('nan'),
                            p1=int(sum(p1)), p4=int(sum(p4)), n=len(ok),
                            rg=float(np.mean(rg)) if rg else float('nan'),
                            od=float(np.mean(od)) if od else float('nan'),
                            pa=int(sum(pa)), kind=kind)
        print(f"{arm:<16}{tr:<9}{te:<7}{len(ok):>4}"
              f"{summary[arm]['rho']:>8.3f}{summary[arm]['p1']:>5}/{len(ok):<2}"
              f"{summary[arm]['p4']:>5}/{len(ok):<2}"
              f"{summary[arm]['rg']:>10.4f}{summary[arm]['od']:>9.2f}"
              f"{summary[arm]['pa']:>5}/{len(ok):<2}  {kind}")

    # ---- 关键对比 ----
    print('\n' + '=' * 76)
    print('关键对比')
    print('=' * 76)

    def cmp_pair(a, b, label):
        if a not in summary or b not in summary:
            print(f'  {label}: 数据不全')
            return
        sa, sb = summary[a], summary[b]
        print(f'  {label}')
        print(f'    {a:<16} ρ={sa["rho"]:.3f} 全通过 {sa["pa"]}/{sa["n"]}  ref_gain={sa["rg"]:+.4f}')
        print(f'    {b:<16} ρ={sb["rho"]:.3f} 全通过 {sb["pa"]}/{sb["n"]}  ref_gain={sb["rg"]:+.4f}')
        d = sb['rho'] - sa['rho']
        verdict = '混训更好' if d > 0.02 else ('单域更好' if d < -0.02 else '无实质差异')
        print(f'    Δρ = {d:+.3f}  → {verdict}')

    cmp_pair('b1_apr_may', 'm3_aprjun_may', '【核心】may 上：单域(apr) vs 混训(apr+jun)')
    cmp_pair('b2_may_apr', 'm4_mayjun_apr', '【核心】apr 上：单域(may) vs 混训(may+jun)')
    cmp_pair('m3_aprjun_may', 'm6_aprjun_gm', '【对照】色调：原始 vs gm 对齐（均训 apr+jun，测 may）')
    print('  【jun 域】m5_aprmay_jun（训 apr+may → 测 jun，唯一"多域→新域"实证格）')
    if 'm5_aprmay_jun' in summary:
        s = summary['m5_aprmay_jun']
        print(f'    ρ={s["rho"]:.3f} 全通过 {s["pa"]}/{s["n"]} ref_gain={s["rg"]:+.4f} '
              f'out_dPSNR={s["od"]:+.2f}')
        print(f'    ⚠️ jun 的 T 帧与 vehicle_ssd/dataset/train 同源（内容级），此格测试集存疑，不作泛化证据')

    print('\n[说明] ρ=Ω内校正率(相对参考 C_I−C_T)；①②③④为项目判据，全通过=四者合取。')
    print('       n 仅为 den_ok 的帧数；den 不足的帧（参考对比不足）已剔除。')


if __name__ == '__main__':
    main()
