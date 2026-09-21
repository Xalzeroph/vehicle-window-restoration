import sys, os, csv, cv2
import numpy as np
os.chdir('/DataDisk/zmy_code/TintedGlass_Enhance')

def load_stats(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[r['case']] = (r.get('s2_rho'), r.get('s2_med_ratio'))
    return out

APR = [('VID_20260426_172719', 'I_000000.png'), ('VID_20260426_172933', 'I_000250.png'),
       ('VID_20260426_173237', 'I_000000.png'), ('VID_20260426_173349', 'I_000000.png'),
       ('VID_20260426_173604', 'I_000007.png'), ('VID_20260426_173827', 'I_000000.png')]
MAY = [('VID_20260512_162203', 'I_000000.png'), ('VID_20260512_162623', 'I_000700.png'),
       ('VID_20260512_162901', 'I_000675.png'), ('VID_20260512_163859', 'I_000000.png'),
       ('VID_20260512_164653', 'I_000000.png')]

def build(tag, dta, gtn, cases, predroot, b1root, statf, outfile):
    st = load_stats(statf)
    H, WP = 300, 400
    def fit(im):
        if im is None: return np.zeros((H, WP, 3), np.uint8)
        x = cv2.resize(im, (max(1, int(im.shape[1]*H/im.shape[0])), H), interpolation=cv2.INTER_AREA)
        c = np.zeros((H, WP, 3), np.uint8)
        c[:, :min(x.shape[1], WP)] = x[:, :WP]
        return c
    rows = []
    for hdr in (['', 'I 输入', 'T 真值', 'S2 输出', 'B1 对照'],):
        lab = np.zeros((H-250, 270, 3), np.uint8); row = [lab]
        for t in hdr: row.append(np.full((H-250, WP, 3), 30, np.uint8))
        # 用文字行做表头
        cells = []
        for t in hdr:
            c = np.full((60, WP if t else 270, 3), 25, np.uint8)
            if t: cv2.putText(c, t, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cells.append(c)
        rows.append(np.hstack(cells))
    for sc, fn in cases:
        key = f'{sc}__{fn[:-4]}'
        I = cv2.imread(f'{dta}/{sc}/{fn}'); T = cv2.imread(f'{dta}/{sc}/{gtn}')
        P = cv2.imread(f'{predroot}/{key}/ytmt_ucs_sirs_l.png')
        B = cv2.imread(f'{b1root}/{key}/ytmt_ucs_sirs_l.png')
        rho, mr = st.get(key, ('?', '?'))
        lab = np.zeros((H, 270, 3), np.uint8)
        for i, t in enumerate([sc[-6:], f'ρ={rho}  ④={mr}']):
            cv2.putText(lab, t, (8, 40+i*32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        rows.append(np.hstack([lab, fit(I), fit(T), fit(P), fit(B)]))
    gap = np.full((8, rows[0].shape[1], 3), 235, np.uint8)
    out = rows[0]
    for r in rows[1:]: out = np.vstack([out, gap, r])
    cv2.imwrite(outfile, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(outfile, out.shape)

build('apr', 'data/test_sets/real_pair_apr_0917', 'GT.png', APR,
      'runs/pred/apr_s101_apr10k', 'runs/pred/apr_b1_apr',
      'eval/eval_s2_apr10k_s101.csv', 'fig/s2_apr_cases_0921.jpg')
build('may', 'data/test_sets/real_pair_may_0917', 'GT.png', MAY,
      'runs/pred/may_rdt_s2_main_s101_10000', 'runs/pred/may_b1_final',
      'eval/eval_s2_card_rdt_s2_main_s101_10000.csv', 'fig/s2_may_cases_0921.jpg')
