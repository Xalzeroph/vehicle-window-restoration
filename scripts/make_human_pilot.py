#!/usr/bin/env python3
"""人评 pilot 材料制作（2026-09-19 轮 §5）。

规格（预登记）：
- 12 图分层：4 强反射 / 4 中 / 4 近零（按掩膜内 D 中位分位，每场景 ≤2）
- A/B = (RDNet_gm, identity_gm) 随机左右；插 2 对 trap=(identity, identity_gm)
- 双问：Q1 整体谁更好（A/B/无偏好）；Q2 有无不该出现的损伤/残留（有/无+备注）
- 5 人，两版本 V1/V2 左右镜像，分发 3/2
- 输出：data/human_pilot_0919/materials/{V1,V2}/*.png（1920×960 并排）
        data/human_pilot_0919/问卷_被试S*.csv、说明.txt、key.json（不随包发放）、发放包_{V1,V2}.zip

machine 判"更优者"规则（本脚本定义，供一致率计算，P4 复核）：
  d_perc>0 → igm；否则 d_struct<0 → rdnet；否则 tie。
"""
import csv, json, os, sys, zipfile
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_protocol import robust_gain, ls_scale, apply_gm  # noqa: E402

OUT = 'data/human_pilot_0919'
SETS = {
    'jun': dict(d='data/test_sets/real_pair_0917', gtn='T.png', layout='jun',
                pred=('runs/pred/real_pair_0917_rdnet', 'ytmt_ucs_sirs_l.png'), csv='eval/eval_proto_jun.csv'),
    'apr': dict(d='data/test_sets/real_pair_apr_0917', gtn='GT.png', layout='multi',
                pred=('runs/pred/apr_rdnet', 'ytmt_ucs_sirs_l.png'), csv='eval/eval_proto_apr.csv'),
    'may': dict(d='data/test_sets/real_pair_may_0917', gtn='GT.png', layout='multi',
                pred=('runs/pred/may_rdnet', 'ytmt_ucs_sirs_l.png'), csv='eval/eval_proto_may.csv'),
}


def load_cases():
    cases = []
    for tag, cfg in SETS.items():
        rows = {r['case']: r for r in csv.DictReader(open(cfg['csv']))}
        for sc in sorted(x for x in os.listdir(cfg['d']) if x.startswith('VID_')):
            sd = os.path.join(cfg['d'], sc)
            ins = [('I.png', sc)] if cfg['layout'] == 'jun' else \
                [(f, f'{sc}__{f[:-4]}') for f in sorted(os.listdir(sd)) if f.startswith('I_')]
            for fn, key in ins:
                if key not in rows:
                    continue
                T = cv2.imread(os.path.join(sd, cfg['gtn']))
                I = cv2.imread(os.path.join(sd, fn))
                pp = os.path.join(cfg['pred'][0], key, cfg['pred'][1])
                if T is None or I is None or not os.path.exists(pp):
                    continue
                P = cv2.imread(pp)
                if P.shape != T.shape:
                    P = cv2.resize(P, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
                g = robust_gain(I, T)
                D = np.abs(I.astype(np.float64) - T.astype(np.float64) * g.reshape(1, 1, 3)).max(axis=2)
                thr = float(rows[key]['thr'])
                M = D > thr
                med_d = float(np.median(D[M])) if M.sum() else 0.0
                r = rows[key]
                d_perc = r.get('rdnet__d_perc')
                d_perc = float(d_perc) if d_perc not in (None, '') else \
                    float(r['rdnet__lpips']) - float(r['identity_gm__lpips'])
                d_struct = float(r['rdnet__d_struct']) if r['rdnet__d_struct'] not in (None, '') else 0.0
                machine = 'igm' if d_perc > 0 else ('rdnet' if d_struct < 0 else 'tie')
                cases.append(dict(id=f'{tag}/{key}', set=tag, scene=sc, key=key, med_d=med_d,
                                  I=I, T=T, P=P, machine=machine,
                                  d_perc=d_perc, d_struct=d_struct))
    return cases


def pick_stratified(cases, per=4, cap=2):
    order = sorted(cases, key=lambda c: c['med_d'], reverse=True)
    picked, picked_ids, scene_cnt = [], set(), {}

    def take(pool):
        n = 0
        for c in pool:
            if c['id'] in picked_ids or scene_cnt.get(c['scene'], 0) >= cap:
                continue
            picked.append(c); picked_ids.add(c['id'])
            scene_cnt[c['scene']] = scene_cnt.get(c['scene'], 0) + 1
            n += 1
            if n >= per:
                break
        return n

    n_strong = take(order)                                # 强：从最高端
    n_near = take(list(reversed(order)))                  # 近零：从最低端
    mid = order[n_strong:len(order) - n_near]
    mid_med = np.median([c['med_d'] for c in mid]) if mid else 0.0
    take(sorted(mid, key=lambda c: abs(c['med_d'] - mid_med)))
    rest = [c for c in order if c['id'] not in picked_ids and scene_cnt.get(c['scene'], 0) < cap]
    traps = rest[len(rest) // 2:len(rest) // 2 + 4]
    if len(traps) < 4:
        traps = rest[:4]
    return picked, traps


def montage_pair(imgA, imgB, h=960, W=1920, gap=8):
    def fit(im):
        H, Wi = im.shape[:2]
        return cv2.resize(im, (max(1, int(Wi * h / H)), h), interpolation=cv2.INTER_AREA)
    half = (W - gap) // 2
    canvas = np.zeros((h, W, 3), np.uint8)
    for im, x0 in ((fit(imgA), 0), (fit(imgB), half + gap)):
        w_ = min(im.shape[1], half)
        x = x0 + (half - w_) // 2
        canvas[:, x:x + w_] = im[:, :w_]
    return canvas


def main():
    os.makedirs(f'{OUT}/materials/V1', exist_ok=True)
    os.makedirs(f'{OUT}/materials/V2', exist_ok=True)
    cases = load_cases()
    print(f'候选 {len(cases)} 例')
    picked, traps = pick_stratified(cases)
    print(f'选中 {len(picked)} + trap {len(traps)}')
    if len(picked) != 12 or len(traps) != 4:
        print('!! 分层不足，需人工检查'); return
    rng = np.random.default_rng(20260919)
    # 主对 + 2 对 identity-vs-igm trap + 2 对 A=A 自比（位置偏置对照）
    items = [dict(c=c, kind='main') for c in picked] \
        + [dict(c=c, kind='trap_igm') for c in traps[:2]] \
        + [dict(c=c, kind='trap_same') for c in traps[2:]]
    rng.shuffle(items)                                    # trap 混入并乱序
    key = {}
    for i, it in enumerate(items):
        c, kind = it['c'], it['kind']
        pid = f'pair_{i + 1:02d}'
        igm = apply_gm(c['I'], ls_scale(c['I'], c['T'])).astype(np.uint8)
        if kind == 'trap_igm':
            x, y = c['I'], igm
        elif kind == 'trap_same':
            x = y = igm
        else:
            x = apply_gm(c['P'], ls_scale(c['P'], c['T'])).astype(np.uint8)   # rdnet_gm
            y = igm
        if rng.random() < 0.5:                            # 逐对随机左右（V1）
            left, right = y, x
        else:
            left, right = x, y
        cv2.imwrite(f'{OUT}/materials/V1/{pid}.png', montage_pair(left, right))
        cv2.imwrite(f'{OUT}/materials/V2/{pid}.png', montage_pair(right, left))
        roles = {'main': ('rdnet_gm', 'identity_gm'), 'trap_igm': ('identity', 'identity_gm'),
                 'trap_same': ('identity_gm', 'identity_gm')}[kind]
        v1_left = roles[0] if left is x else roles[1]
        key[pid] = dict(set=c['set'], case=c['key'], scene=c['scene'], kind=kind,
                        trap=kind != 'main', med_d=round(c['med_d'], 2), machine=c['machine'],
                        d_perc=round(c['d_perc'], 4), d_struct=round(c['d_struct'], 4),
                        V1_left=v1_left)
    json.dump(key, open(f'{OUT}/key.json', 'w'), indent=1, ensure_ascii=False)

    qcols = ['pair', 'Q1_哪边更好(A/B/无偏好)', 'Q2_有无不该出现的损伤或残留(有/无)', 'Q2_备注']
    for s in range(1, 6):
        with open(f'{OUT}/问卷_被试S{s}.csv', 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh); w.writerow(qcols)
            for i in range(len(items)):
                w.writerow([f'pair_{i + 1:02d}', '', '', ''])
    open(f'{OUT}/说明.txt', 'w', encoding='utf-8').write(
        '材料说明（人评 pilot，2026-09-19）\n'
        '1) materials/ 下每个 pair_XX.png 为左右并排两张同场景图（左 A / 右 B）。\n'
        '2) 请只看"有没有反射/人影残留"和"画面是否自然（有无被抹糊、拖影、发灰等损伤）"。\n'
        '3) 每个 pair 回答两问：Q1 整体哪边更好（A/B/无偏好）；Q2 有无不该出现的损伤或残留（有/无，可备注）。\n'
        '4) 填 问卷_被试S*.csv，不写姓名，只填编号。\n')
    with zipfile.ZipFile(f'{OUT}/发放包_V1.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for i in range(len(items)):
            z.write(f'{OUT}/materials/V1/pair_{i + 1:02d}.png', f'materials/pair_{i + 1:02d}.png')
        z.write(f'{OUT}/说明.txt', '说明.txt')
        for s in (1, 3, 5):
            z.write(f'{OUT}/问卷_被试S{s}.csv', f'问卷_被试S{s}.csv')
    with zipfile.ZipFile(f'{OUT}/发放包_V2.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for i in range(len(items)):
            z.write(f'{OUT}/materials/V2/pair_{i + 1:02d}.png', f'materials/pair_{i + 1:02d}.png')
        z.write(f'{OUT}/说明.txt', '说明.txt')
        for s in (2, 4):
            z.write(f'{OUT}/问卷_被试S{s}.csv', f'问卷_被试S{s}.csv')
    print(f'写出 {OUT}/（materials V1/V2、问卷×5、说明、key.json、发放包 zip）')
    for pid, v in key.items():
        print(f"{pid}: {v['set']}/{v['case']} medD={v['med_d']} trap={v['trap']} machine={v['machine']}")


if __name__ == '__main__':
    main()
