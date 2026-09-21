#!/usr/bin/env python3
"""官方原生 vs S2 微调 推理对比图（I | 官方 RDNet（原生） | S2 微调 | 真值 T）。

--set may：全图 5 场景 + 茶叶罐区 1:1 放大（绿色 HSV 定位）；
--set apr：全图 6 场景 + 人影洞（Ω）区 1:1 放大（Ω 最大连通域质心，取 Ω 面积最大 3 帧）。
左列标签含逐帧"可见区结构保留"（sP/sI，来自 eval/probe_s2_structure_visibility_*_0921.csv）。
用法: python3 scripts/probes_0921/fig_s2_official_vs_ft.py [--set may|apr]
"""
import argparse, csv, importlib.util, os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir('/DataDisk/zmy_code/TintedGlass_Enhance')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
LAB_W, CELL, H_FULL, CROP = 270, 560, 300, 560
MAY_ROWS = [('VID_20260512_162203', 'I_000000.png'), ('VID_20260512_162623', 'I_000700.png'),
            ('VID_20260512_162901', 'I_000675.png'), ('VID_20260512_163859', 'I_000000.png'),
            ('VID_20260512_164653', 'I_000000.png')]
MAY_ZOOM = [('VID_20260512_163859', 'I_000000.png', (697, 649)),
            ('VID_20260512_164653', 'I_000000.png', (898, 709)),
            ('VID_20260512_162203', 'I_000000.png', (693, 822))]
APR_ROWS = [('VID_20260426_172719', 'I_000000.png'), ('VID_20260426_172933', 'I_000250.png'),
            ('VID_20260426_173237', 'I_000000.png'), ('VID_20260426_173349', 'I_000000.png'),
            ('VID_20260426_173604', 'I_000007.png'), ('VID_20260426_173827', 'I_000000.png')]
SETS = {
    'may': dict(d='data/test_sets/real_pair_may_0917', off='runs/pred/may_rdnet_native',
                ft='runs/pred/may_rdt_s2_main_s101_10000',
                st_o='eval/probe_s2_structure_visibility_official_0921.csv',
                st_f='eval/probe_s2_structure_visibility_0921.csv',
                out='fig/s2_official_vs_ft_0921.jpg', rows=MAY_ROWS, zoom=MAY_ZOOM,
                zoom_title='以下：茶叶罐区域 1:1 原尺寸放大'),
    'apr': dict(d='data/test_sets/real_pair_apr_0917', off='runs/pred/apr_rdnet_native',
                ft='runs/pred/apr_s101_apr10k',
                st_o='eval/probe_s2_structure_visibility_apr_official_0921.csv',
                st_f='eval/probe_s2_structure_visibility_apr_0921.csv',
                out='fig/s2_official_vs_ft_apr_0921.jpg', rows=APR_ROWS, zoom=None,
                zoom_title='以下：人影洞（Ω）区域 1:1 原尺寸放大（取 Ω 面积最大 3 帧）'),
}


def stats(path):
    return {f"{r['scene']}__{r['frame']}": float(r['sP_sI_vis']) for r in csv.DictReader(open(path))}


def text_img(lines, w, h, fs=19, bg=(16, 16, 16)):
    im = Image.new('RGB', (w, h), bg)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, fs)
    y = 12
    for t, col in lines:
        d.text((10, y), t, font=f, fill=col)
        y += fs + 8
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def fit(im, h, w):
    x = cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_AREA)
    c = np.zeros((h, w, 3), np.uint8)
    c[:, :min(x.shape[1], w)] = x[:, :w]
    return c


def crop1to1(im, cx, cy, s=CROP):
    h, w = im.shape[:2]
    x0 = min(max(cx - s // 2, 0), w - s)
    y0 = min(max(cy - s // 2, 0), h - s)
    return im[y0:y0 + s, x0:x0 + s]


def auto_zoom(cfg):
    """Ω（人影洞）最大连通域质心，取 Ω 面积最大的 3 帧。"""
    spec = importlib.util.spec_from_file_location('eg2', 'scripts/eval_g2_baseline.py')
    eg2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eg2)
    cand = []
    for sc, fn in cfg['rows']:
        I = cv2.imread(f"{cfg['d']}/{sc}/{fn}")
        T = cv2.imread(f"{cfg['d']}/{sc}/GT.png")
        o = eg2.omega_of(I, T)
        if not o:
            continue
        n, _, st, cen = cv2.connectedComponentsWithStats(o['Om'].astype(np.uint8), 8)
        if n < 2:
            continue
        i = 1 + int(np.argmax(st[1:, 4]))
        cand.append((int(st[i, 4]), (sc, fn, (int(cen[i][0]), int(cen[i][1])))))
    cand.sort(key=lambda x: -x[0])
    out = [c[1] for c in cand[:3]]
    print('auto zoom:', [(c[0][-6:], c[1][:-4], c[2]) for c in out])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['may', 'apr'], default='may')
    args = ap.parse_args()
    cfg = SETS[args.set]
    so, sf = stats(cfg['st_o']), stats(cfg['st_f'])
    zoom = cfg['zoom'] if cfg['zoom'] is not None else auto_zoom(cfg)
    W = LAB_W + CELL * 4
    gap = np.full((8, W, 3), 235, np.uint8)

    def panel(sc, fn, cx=None):
        key = f'{sc}__{fn[:-4]}'
        ims = [cv2.imread(f"{cfg['d']}/{sc}/{fn}"), cv2.imread(f"{cfg['off']}/{key}/ytmt_ucs_sirs_l.png"),
               cv2.imread(f"{cfg['ft']}/{key}/ytmt_ucs_sirs_l.png"), cv2.imread(f"{cfg['d']}/{sc}/GT.png")]
        if cx is None:
            lab = text_img([(sc[-6:], (255, 255, 255)), (fn[:-4], (150, 150, 150)),
                            (f'可见区保留 官{so[key]:.2f} / S2{sf[key]:.2f}', (255, 220, 120))],
                           LAB_W, H_FULL)
            return np.hstack([lab] + [fit(x, H_FULL, CELL) for x in ims])
        lab = text_img([(sc[-6:], (255, 255, 255)), (fn[:-4], (150, 150, 150)),
                        (f'可见区保留 官{so[key]:.2f} / S2{sf[key]:.2f}', (255, 220, 120)),
                        ('1:1 放大', (150, 150, 150))], LAB_W, CROP)
        return np.hstack([lab] + [crop1to1(x, *cx) for x in ims])

    rows = [np.hstack([text_img([], LAB_W, 64)] +
                      [text_img([(t, (255, 255, 255))], CELL, 64, 26) for t in
                       ['输入 I', '官方 RDNet（原生）', 'S2 微调（s101@10k）', '真值 T（GT）']])]
    for sc, fn in cfg['rows']:
        rows.append(panel(sc, fn))
    rows.append(np.hstack([text_img([(cfg['zoom_title'], (200, 200, 200))], W, 40, 22)]))
    for sc, fn, cx in zoom:
        rows.append(panel(sc, fn, cx))
    mo = float(np.mean(list(so.values())))
    mf = float(np.mean(list(sf.values())))
    rows.append(text_img(
        [(f'可见区结构保留 = sP/sI（同帧同口径，1.0 = 输入里看得见的结构完全保住）；'
          f'{len(sf)} 帧均值：官方 {mo:.2f} vs S2 微调 {mf:.2f}；'
          f'数据 eval/probe_s2_structure_visibility_*.csv', (200, 200, 200))], W, 40, 22))

    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, gap, r])
    cv2.imwrite(cfg['out'], out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(cfg['out'], out.shape, f'{os.path.getsize(cfg["out"])/1e6:.1f}MB')


if __name__ == '__main__':
    main()
