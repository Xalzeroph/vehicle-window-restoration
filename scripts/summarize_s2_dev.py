#!/usr/bin/env python3
"""汇总 S2 评测卡 CSV 的 ①–④ 通过数（对照预登记门）。用法:
  python3 scripts/summarize_s2_dev.py eval/eval_s2_card_s2it1000.csv [tag]
"""
import csv, sys


def main():
    path = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ''
    rows = list(csv.DictReader(open(path)))
    sub = [r for r in rows if r.get('den_ok') == '1']
    n = len(sub)
    arms = sorted({k[:-6] for r in rows for k in r if k.endswith('_pass1')})

    def gf(r, k):
        try:
            return float(r.get(k, ''))
        except (TypeError, ValueError):
            return None

    print(f'== {tag} | dev=may 有效 {n} 行（预登记门：①≥8 ②≥7 ③11/11 ④≥10 且④中位≤0.8）==')
    for an in arms:
        c1 = sum(1 for r in sub if r.get(f'{an}_pass1') == '1')
        c2 = sum(1 for r in sub if (gf(r, f'{an}_ref_gain') or -9) >= 0.02)
        c3v = [gf(r, f'{an}_out_dPSNR') for r in sub]
        c3 = sum(1 for v in c3v if v is not None and v >= -0.2)
        c4 = sum(1 for r in sub if r.get(f'{an}_pass4') == '1')
        mr = sorted(v for v in (gf(r, f'{an}_med_ratio') for r in sub) if v is not None)
        med = mr[len(mr) // 2] if mr else float('nan')
        rho = sorted(v for v in (gf(r, f'{an}_rho') for r in sub) if v is not None)
        p25 = rho[max(0, len(rho) // 4 - 1)] if rho else float('nan')
        print(f'{an:<10} ①{c1}/{n} ②{c2}/{n} ③{c3}/{n} ④{c4}/{n} (④中位 {med:.3f}, ρ_p25 {p25:.2f})')


if __name__ == '__main__':
    main()
