import csv
rows = list(csv.DictReader(open('eval/eval_s2_card_s101_it7000.csv')))
print(f'{"case":<42} {"ρ":>6} {"med_ratio":>9} {"②":>8} {"③":>7} {"Ω%":>5}')
for r in rows:
    if r.get('den_ok') != '1': continue
    print(f'{r["case"]:<42} {r["s2_rho"]:>6} {r["s2_med_ratio"]:>9} {r["s2_ref_gain"]:>8} {r["s2_out_dPSNR"]:>7} {r["omega_pct"]:>5}')
