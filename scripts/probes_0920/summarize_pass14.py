import csv
rows = list(csv.DictReader(open('eval/eval_g2_baseline.csv')))
print('== 判据集 ①–④ 汇总（den_ok 内）==')
for tag in ('jun','apr','may'):
    for an in ('identity','perfect','rdnet','remo'):
        sub = [r for r in rows if r['set']==tag and r['den_ok']=='1' and r.get(f'{an}_pass1') not in (None,'')]
        n = len(sub)
        if not n: continue
        c = lambda k: sum(1 for r in sub if r[f'{an}_{k}']=='1')
        print(f'{tag}/{an:<9} n={n:>2} | ①{c("pass1"):>2} ④{c("pass4"):>2} 全过{c("pass_all"):>2}')
print()
allw = [r for r in rows if r.get('rdnet_pass_all')=='1' or r.get('remo_pass_all')=='1']
print('真实臂全过 case:', [(r['set'], r['case']) for r in allw] or '无')
