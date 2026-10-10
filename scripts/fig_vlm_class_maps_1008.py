#!/usr/bin/env python3
"""SmolVLM / Qwen 对各类反射物的空间判断图（5x5 网格，30 帧口径）。

注意：这不是"置信度"。生成式 LMM 不输出概率图；此处的细胞数值 =
该类别被模型指派到该格号的帧数占比。它反映的是"模型每次说了什么"的空间分布。
用法：python3 scripts/fig_vlm_class_maps.py
"""
import csv, json, os, re, textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 5
CELL = re.compile(r'"cells"\s*:\s*\[([^\]]*)\]')
OBJ = re.compile(r'"obj"\s*:\s*"([^"]*)"')

# 用户关心的类别 <- SmolVLM 实际用词
CLASS_MAP = {
    '人影': ['人的身体部位', '人', '手', '人影'],
    '手机': ['手机'],
    '树影': ['树叶枝条', '树叶', '树枝'],
    '三脚架': ['三脚架'],
    '灯影': ['灯影'],
}
FONT_CAND = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]


def font(sz):
    for p in FONT_CAND:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def cellmask(cells, n=N):
    m = np.zeros((n, n), bool)
    for c in cells:
        col, row = ord(c[0]) - 65, int(c[1]) - 1
        if 0 <= col < n and 0 <= row < n:
            m[row, col] = True
    return m


def parse(txt):
    """返回 [(obj, [cells])]，用正则配对以免 T1_txt 尾部多余内容破坏 json 解析。"""
    out = []
    for om, cm in zip(OBJ.finditer(txt or ''), CELL.finditer(txt or '')):
        cells = [c.strip().strip('"').upper() for c in cm.group(1).split(',')]
        cells = [c for c in cells if re.fullmatch(r'[A-E][1-5]', c)]
        if cells:
            out.append((om.group(1), cells))
    return out


def load(tag):
    p = f'{PROJ}/eval/vlm_probe_{tag}.csv'
    return [r for r in csv.DictReader(open(p, encoding='utf-8-sig')) if r.get('key')]


def draw_map(dr, x0, y0, M, cell, title, sub, n=N, vmax=None, gt=False):
    """画一个 5x5 图。M 为 n x n 的数值（帧数占比 0-1）。"""
    f_t, f_c, f_s = font(17), font(13), font(11)
    dr.text((x0, y0), title, font=f_t, fill=(0, 0, 0))
    dr.text((x0, y0 + 21), sub, font=f_s, fill=(90, 90, 90))
    gx, gy = x0, y0 + 56          # 网格下移，给副标题留出空间
    # 列标 A-E / 行标 1-5
    for c in range(n):
        dr.text((gx + c*cell + cell//2 - 5, gy - 17), chr(65+c), font=f_c, fill=(60, 60, 60))
    for r in range(n):
        dr.text((gx - 15, gy + r*cell + cell//2 - 7), str(r+1), font=f_c, fill=(60, 60, 60))
    vmax = vmax or max(M.max(), 1e-9)
    for r in range(n):
        for c in range(n):
            v = M[r, c]
            t = v / vmax if vmax > 0 else 0
            if v == 0:
                fill = (248, 248, 248)
            else:
                # 白 -> 黄 -> 红
                t = min(1.0, t)
                if t < 0.5:
                    u = t/0.5; fill = (255, int(255-40*u), int(230-230*u))
                else:
                    u = (t-0.5)/0.5; fill = (255, int(215-215*u), 0)
            X, Y = gx + c*cell, gy + r*cell
            dr.rectangle([X, Y, X+cell-1, Y+cell-1], fill=fill,
                         outline=(170, 60, 60) if gt else (200, 200, 200), width=2 if gt else 1)
            if v > 0:
                txt = f'{v*100:.0f}' if v < 1 else '100'
                dr.text((X + cell//2 - (7 if len(txt) == 1 else 13), Y + cell//2 - 8),
                        txt, font=f_c, fill=(40, 40, 40))
    return gy + n*cell


def main():
    rows = load('smolvlm')
    J = json.load(open(f'{PROJ}/data/annot_hole_cells.json', encoding='utf-8'))
    keys = [r['key'] for r in rows]
    nframe = len(keys)

    # ---- 真值 ROI 频率 ----
    gt = np.zeros((N, N))
    for k in keys:
        gt += cellmask(J[k]['pos'])
    gt /= nframe

    # ---- 模型：泛称 vs 具体物 ----
    generic = np.zeros((N, N)); percls = {c: np.zeros((N, N)) for c in CLASS_MAP}
    wrong = np.zeros((N, N)); nall = np.zeros((N, N))
    WRONG = {'玻璃后面的真实场景'}
    ALL_ALIAS = {a for v in CLASS_MAP.values() for a in v}
    # 每类的"出现帧数"（用于副标题，避免与格指派次数混淆）
    cls_frames = {c: set() for c in CLASS_MAP}
    gen_frames, wrong_frames = set(), set()
    for r in rows:
        k = r['key']
        for obj, cells in parse(r.get('T1_txt', '')):
            if obj in WRONG:
                wrong_frames.add(k)
            # 泛称 = 既不是具体物、也不是"误标真实场景"的其它一切
            if obj not in ALL_ALIAS and obj not in WRONG:
                gen_frames.add(k)
            for c in cells:
                col, row = ord(c[0])-65, int(c[1])-1
                nall[row, col] += 1
                if obj in WRONG:
                    wrong[row, col] += 1
                elif obj not in ALL_ALIAS:
                    generic[row, col] += 1
                for cname, alias in CLASS_MAP.items():
                    if obj in alias:
                        percls[cname][row, col] += 1
                        cls_frames[cname].add(k)
    for M in (generic, wrong, nall):
        M /= nframe
    for c in percls:
        percls[c] /= nframe

    # ---- 画布 ----
    panels = [('真值：用户标注 ROI 出现频率', f'30 帧，格内数字=帧占比%（平均覆盖 {gt.mean()*100:.0f}% 画面）', gt, True),
              ('SmolVLM：泛称"玻璃反射/反射物体"', f'{len(gen_frames)} 帧出现 | {int(round(generic.sum()*nframe))} 次格指派', generic, False)]
    for cname in CLASS_MAP:
        cls = percls[cname]
        panels.append((f'SmolVLM：{cname}',
                       f'{len(cls_frames[cname])} 帧出现 | {int(round(cls.sum()*nframe))} 次格指派',
                       cls, False))

    cell, gap = 62, 26
    pw = N*cell + 60
    ph = N*cell + 108          # 网格下移后每格面板加高，避免下一行标题叠上来
    cols = 4
    rowsn = (len(panels) + cols - 1)//cols
    W = cols*pw + 40
    H = rowsn*ph + 190         # 底部留足结论区，防止被裁
    img = Image.new('RGB', (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)

    f_h, f_n = font(21), font(12)
    dr.text((24, 14), 'SmolVLM / Qwen 对各类反射物的空间判断（5×5 格，n=30 帧）', font=f_h, fill=(0, 0, 0))
    dr.text((24, 44), '数值 = 该类别被指派到该格的帧数占比（%）。这不是模型置信度，是"它每次说了什么"的空间分布。', font=f_n, fill=(80, 80, 80))

    vmax = max([p[2].max() for p in panels] + [1e-9])
    for i, (title, sub, M, is_gt) in enumerate(panels):
        cx = 24 + (i % cols)*pw
        cy = 78 + (i // cols)*ph
        draw_map(dr, cx + 20, cy, M, cell, title, sub, vmax=vmax, gt=is_gt)

    # Qwen 说明板
    qx = 24 + (len(panels) % cols)*pw + 20
    qy = 78 + (len(panels)//cols)*ph
    dr.rectangle([qx-10, qy+30, qx + N*cell + 40, qy + 40 + N*cell], outline=(200, 60, 60), width=3)
    dr.text((qx, qy), 'Qwen2.5-VL-3B：空', font=font(17), fill=(190, 30, 30))
    for j, line in enumerate(['30/30 帧输出 []（拒答）', '带网格叠图时完全不报反射物', '', '去掉网格后能答"有反射"，', '但方位答成左上+右上，', '真值在左下/中下 → 方位错',
                              '', '∴ 无类别可画']):
        dr.text((qx, qy + 50 + j*19), line, font=font(12), fill=(150, 40, 40))

    # 底部结论（自动折行，避免超出画布被裁）
    yb = 78 + rowsn*ph + 4
    concl = [
        '读图要点：',
        '1) 具体物体类（人影/手机/树影/三脚架/灯影）每类只有 2-3 帧出现、格指派全部挤在最左列 A/B —— 与真值 ROI（B4/C5 为中心）不重合。',
        '2) 类别名与提示词给出的例子列表（人的身体部位/手/手机/树叶枝条/三脚架/灯影）逐一对应，疑似照着 prompt 复述，而非看图归因。',
        '3) 大量输出是泛称"玻璃反射/反射物体"（把区域喊成反射，不说是谁），另有 4 帧共 20 格把反射误标成"玻璃后面的真实场景"。',
        '4) 结论：模型对"哪一类反射物、在哪里"没有可用判断；此前的高 IoU 帧靠的是把左下大片涂满（平均覆盖 80% 画面）。',
    ]
    f_cj = font(13 if True else 12)
    yy = yb
    for j, line in enumerate(concl):
        fnt = font(14 if j == 0 else 12)
        for seg in textwrap.wrap(line, width=88) or ['']:
            dr.text((24, yy), seg, font=fnt, fill=(0, 0, 0) if j == 0 else (60, 60, 60))
            yy += 19
        if j == 0:
            yy += 3

    os.makedirs(f'{PROJ}/fig', exist_ok=True)
    out = f'{PROJ}/fig/vlm_class_maps_1008.png'
    img.save(out)
    print('[saved]', out, img.size)
    print(f'[真值 ROI 平均覆盖 {gt.mean()*100:.0f}% 画面]')
    print('[各类别] ' + '  '.join(f'{k}: {len(cls_frames[k])}帧/{int(round(v.sum()*nframe))}格' for k, v in percls.items()))
    print(f'[泛称] {len(gen_frames)}帧/{int(round(generic.sum()*nframe))}格   [误标真实场景] {len(wrong_frames)}帧/{int(round(wrong.sum()*nframe))}格')
    print(f'[全部指派] {int(round(nall.sum()*nframe))} 格次，覆盖 {len([1 for r in rows if parse(r.get("T1_txt",""))])} 帧')


if __name__ == '__main__':
    main()
