#!/usr/bin/env python3
"""VOC2012 训练 T 源预处理（S2 数据管线 P3-1）。

按 RDNet 官方列表 `RDNet/data/VOC2012_224_train_png.txt`（15287 张）筛 VOC JPEGImages，
短边缩放到 320（供训练时随机裁 224–320）、JPEG q90 重编（匹配实拍压缩域），
存 `data/external/VOC2012/VOC2012_s320/`。

用法: python3 scripts/prep_voc_train_0920.py [--workers 16]
"""
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJ, 'data/external/VOC2012/VOCdevkit/VOC2012/JPEGImages')
LIST = os.path.join(PROJ, 'RDNet/data/VOC2012_224_train_png.txt')
OUT = os.path.join(PROJ, 'data/external/VOC2012/VOC2012_s320')
SHORT = 320


def one(name):
    stem = os.path.splitext(name)[0]
    src = os.path.join(SRC, stem + '.jpg')
    img = cv2.imread(src)
    if img is None:
        return (stem, 'missing')
    h, w = img.shape[:2]
    s = SHORT / min(h, w)
    if s != 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(os.path.join(OUT, stem + '.jpg'), img,
                     [cv2.IMWRITE_JPEG_QUALITY, 90])
    return (stem, 'ok' if ok else 'write_fail')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = [l.strip() for l in open(LIST) if l.strip()]
    print(f'列表 {len(names)} 张 -> {OUT}')
    stats = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (stem, st) in enumerate(ex.map(one, names, chunksize=32)):
            stats[st] = stats.get(st, 0) + 1
            if (i + 1) % 2000 == 0:
                print(f'  {i+1}/{len(names)} {stats}', flush=True)
    print(f'完成: {stats}')
    n_out = len([f for f in os.listdir(OUT) if f.endswith('.jpg')])
    print(f'输出文件数 {n_out}')


if __name__ == '__main__':
    main()
