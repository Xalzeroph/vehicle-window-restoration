#!/usr/bin/env python3
"""机制A探针：现成语义模型能否认出"反射洞"（遮挡者轮廓）？
真值侧洞定义（仅用于分析，用了 GT）：H = 大而连通的"输入比真值暗"区（反射被挡）
单帧侧探测（不许用 GT）：Faster R-CNN person 框 / SegFormer person 类
判据：探测器与 H 的 IoU ≥ 0.3 → 机制 A 可行（语义能认出洞）
"""
import glob, os, sys
import cv2, numpy as np, torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T2
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
sys.path.insert(0, 'scripts')
from eval_protocol import robust_gain
dev = torch.device('cuda:0')

det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).to(dev).eval()
seg = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512').to(dev).eval()
spr = AutoImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
PERSON = [i for i, l in seg.config.id2label.items() if 'person' in l.lower() or 'human' in l.lower()]
print('ADE 里 person 类 id:', PERSON, [seg.config.id2label[i] for i in PERSON], flush=True)


def true_hole(I, T, q=15, min_frac=0.02):
    """分析侧：输入显著比真值暗（反射被挡）的大连通区"""
    g = robust_gain(I, T)
    signed = I.astype(np.float64).mean(2) - T.astype(np.float64).mean(2)*g.mean()
    neg = ((signed < -np.percentile(np.abs(signed), 100-q))*255).astype(np.uint8)
    neg = cv2.morphologyEx(neg, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    neg = cv2.morphologyEx(neg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(neg)
    if n <= 1:
        return np.zeros_like(neg), None
    k = 1 + int(np.argmax(st[1:, 4]))
    if st[k, 4] < min_frac*neg.size:
        return np.zeros_like(neg), None
    H = (lab == k).astype(np.uint8)
    x, y, w, h, _ = st[k]
    return H, (x, y, w, h)


rows = []
for sname in ('apr', 'may'):
    for f in sorted(glob.glob(f'data/gate_v2/lovo_k2/{sname}/*.npz')):
        z = np.load(f); I, T = z['I'], z['T']
        H, box = true_hole(I, T)
        if box is None:
            continue
        # 单帧探测：person 检测
        xt = torch.from_numpy(I[:, :, ::-1].copy().transpose(2, 0, 1)).float().div(255).to(dev)
        with torch.no_grad():
            out = det([xt])[0]
        pbox = None; pscore = 0.0
        for b, s, l in zip(out['boxes'], out['scores'], out['labels']):
            if int(l) == 1 and float(s) > pscore:      # COCO 1 = person
                pbox = [int(v) for v in b.tolist()]; pscore = float(s)
        # 单帧探测：SegFormer person 掩膜
        with torch.no_grad():
            lg = seg(**spr(images=I, return_tensors='pt').to(dev)).logits
            pg = torch.nn.functional.interpolate(lg, size=I.shape[:2], mode='bilinear', align_corners=False)
            pr = pg.softmax(1)[0]
            pm = torch.zeros(I.shape[:2], device=dev)
            for i in PERSON:
                pm = torch.maximum(pm, pr[i])
        pmask = (pm > 0.5).cpu().numpy().astype(np.uint8)
        # 真值侧 H 可能是多个框（取最大），IoU 对 pmask 与 H
        inter = np.logical_and(pmask > 0, H > 0).sum(); union = np.logical_or(pmask > 0, H > 0).sum()
        iou = inter/max(union, 1)
        # 框对框 IoU
        biou = 0.0
        if pbox:
            x, y, w, h = box
            ix1, iy1 = max(x, pbox[0]), max(y, pbox[1]); ix2, iy2 = min(x+w, pbox[2]), min(y+h, pbox[3])
            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
            biou = (iw*ih)/max(w*h + (pbox[2]-pbox[0])*(pbox[3]-pbox[1]) - iw*ih, 1)
        rows.append((sname, os.path.basename(f)[:-4], box, 100*H.mean(), pscore, pbox, 100*pmask.mean(), iou, biou))
        print(f'  {sname}/{os.path.basename(f)[:-4]:<32} 真洞框{box} 占{100*H.mean():4.1f}% | '
              f'person检测 score={pscore:.2f} 框={pbox} | seg person 占{100*pmask.mean():4.1f}% | 掩膜IoU={iou:.3f} 框IoU={biou:.3f}', flush=True)

a = np.array([[r[7], r[8], r[4]] for r in rows], float)
print(f'\n== 汇总 n={len(rows)}：掩膜 IoU 均值 {a[:,0].mean():.3f}（中位 {np.median(a[:,0]):.3f}）| '
      f'框 IoU 均值 {a[:,1].mean():.3f} | person 检测得分均值 {a[:,2].mean():.2f}')
print('判据：掩膜或框 IoU ≥ 0.3 → 机制 A（语义认洞）可行')
