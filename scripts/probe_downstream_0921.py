#!/usr/bin/env python3
"""下游任务伤害探针 0921：人影区 Ω 会不会欺骗 COCO 检测器？

问题：实拍时拍摄者挡住反射光 → 画面出现「暗人形」区域 Ω（洞）。本探针查：
  (a) 检测器会不会在这个人影上「凭空检出」（ghost，I 有 GT 无）；
  (b) 会不会因人影遮挡而「漏检」本该有的目标（missed，GT 有 I 无）。

====================== 预登记判定门（跑之前写死，不调参）======================
  置信度阈值 CONF = 0.50；框匹配 IoU 阈值 IOU = 0.50；类别敏感匹配（同类才配）
  三分类：共同框(common) / ghost 框(只在 I) / 漏检框(missed, 只在 GT)
  「伤害存在」 := 方向跨场景一致（≥2/3 场景同方向：ghost 或 漏检为主）
                 且 该现象在 ≥1/3 的人影帧上出现
  达不到 → 如实报「无伤害」（这也是有效结果）
  弱结论门（附加，不改判决）：若 <1/3 的帧「两张图任一有检出」，
  说明检测器整体不灵敏 → 结论标注 WEAK，并给出低置信(0.3)诊断列供参考
============================================================================

数据（红线：jun 数据绝不使用）：
  - apr: data/test_sets/real_pair_apr_0917 （6 场景 / 16 帧 I）
  - may: data/test_sets/real_pair_may_0917 （5 场景 / 14 帧 I）
    GT.png = 同机位手工挑选的「无人影」干净帧（shift=0 已对齐核验），
    故 GT 与 I 背景同源，框级 IoU 匹配成立；I 与 GT 的「差」即人影 Ω。
  - 第三态（修复输出）：runs/pred/{apr_s101_apr10k, may_rdt_s2_main_s101_10000}/<key>/ytmt_ucs_sirs_l.png
  - MyVideo 基线：data/My_Reflection_Videos_cropped（24 场景 / 抽样，无 GT，只做
    「检出的人形框是否落在画面中下部 = 拍摄者自反射典型位置」的粗略基线）

Ω 定义：复用 scripts/eval_g2_baseline.py::omega_of(I, GT)（v2 定式，不重写）。

检测器：
  - Faster R-CNN ResNet50-FPN (COCO)   —— 主力，类别计数/三分类
  - Keypoint R-CNN ResNet50-FPN (COCO) —— 人形专项，「骨架落在人影上」的最强证据

产物：
  eval/eval_downstream_0921.csv          逐帧主表（pair：共同/ghost/漏检/落 Ω）
  eval/eval_downstream_0921_myvideo.csv  MyVideo 抽样逐帧表
  eval/eval_downstream_0921_verdict.json 汇总 + 门控判定
  fig/downstream_ghost_0921.jpg          3 行 × 4 列可视化拼图
"""
import argparse, csv, json, os, sys, time

# 共享机守则：钉卡在 import torch 之前生效（把 --gpu N 转成 CUDA_VISIBLE_DEVICES）
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
    elif _a.startswith('--gpu='):
        os.environ['CUDA_VISIBLE_DEVICES'] = _a.split('=', 1)[1]

import faulthandler  # noqa: E402

faulthandler.enable()   # 硬崩溃（段错误等）也留痕，便于定位

from collections import Counter  # noqa: E402

import cv2
import numpy as np
import torch
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
                                          keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_g2_baseline import omega_of  # noqa: E402  （Ω v2 定式，唯一权威实现）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = 0.50          # 预登记：置信度阈值（不调）
CONF_LO = 0.30       # 仅诊断列，不参与判决
IOU = 0.50           # 预登记：匹配 IoU（不调）
# 画面中下部带（「拍摄者自反射/人影」的典型位置）
BAND_CX = (0.20, 0.80)
BAND_CY = 0.45
BAND_CY_SENS = 0.50   # 敏感度口径
PAIRS = {'apr': ('data/test_sets/real_pair_apr_0917', 'runs/pred/apr_s101_apr10k'),
         'may': ('data/test_sets/real_pair_may_0917', 'runs/pred/may_rdt_s2_main_s101_10000')}
REPAIR_FILE = 'ytmt_ucs_sirs_l.png'
MYVIDEO = 'data/My_Reflection_Videos_cropped'
KP_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16), (11, 12)]
CLR = dict(ghost=(0, 0, 255), common=(0, 200, 0), missed=(255, 120, 0), omega=(255, 255, 0),
           kp=(0, 0, 255), band=(180, 180, 180))


# ---------------------------------------------------------------- 基础工具
def to_tensor(img, dev):
    return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(dev)


def _sub(d, thr):
    """按阈值切出子集（detect 一次跑低阈值，判决用 CONF，诊断用 CONF_LO，省 GPU 前向）"""
    k = d['scores'] >= thr
    return dict(boxes=d['boxes'][k], scores=d['scores'][k], labels=d['labels'][k],
                names=[n for n, kk in zip(d['names'], k) if kk],
                keypoints=(d['keypoints'][k] if 'keypoints' in d else None),
                kp_scores=(d['kp_scores'][k] if 'kp_scores' in d else None))


def frcnn_detect(model, cats, img, dev, conf=0.0):
    with torch.no_grad():
        o = model([to_tensor(img, dev)])[0]
    k = (o['scores'] >= conf).cpu().numpy()
    return dict(boxes=o['boxes'].cpu().numpy()[k], scores=o['scores'].cpu().numpy()[k],
                labels=o['labels'].cpu().numpy()[k],
                names=[cats[i] for i in o['labels'].cpu().numpy()[k]])


def kprcnn_detect(model, img, dev, conf=0.0):
    with torch.no_grad():
        o = model([to_tensor(img, dev)])[0]
    k = (o['scores'] >= conf).cpu().numpy()
    return dict(boxes=o['boxes'].cpu().numpy()[k], scores=o['scores'].cpu().numpy()[k],
                keypoints=o['keypoints'].cpu().numpy()[k], kp_scores=o['keypoints_scores'].cpu().numpy()[k])


def kp_person_scores(kp_scores):
    """(K,17) 逐关键点分 → 每人一个标量（均值），供绘制标注"""
    ks = np.atleast_2d(np.asarray(kp_scores))
    return [round(float(np.mean(v)), 2) for v in ks] if ks.size else []


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] \
        + (b[:, 2] - b[:, 0])[None, :] * (b[:, 3] - b[:, 1])[None, :] - inter
    return inter / np.maximum(ua, 1e-6)


def match(A, B, thr=IOU, class_aware=True):
    """A(预测, 分数降序贪心) vs B(参考) → (共同对, ghost 下标, 漏检下标)"""
    pairs, used_b = [], set()
    if not len(A['boxes']) or not len(B['boxes']):
        return pairs, list(range(len(A['boxes']))), list(range(len(B['boxes'])))
    M = iou_mat(A['boxes'], B['boxes'])
    for i in np.argsort(-A['scores']):
        best, bj = thr, -1
        for j in range(len(B['boxes'])):
            if j in used_b or M[i, j] < best:
                continue
            if class_aware and A['names'][i] != B['names'][j]:
                continue
            best, bj = M[i, j], j
        if bj >= 0:
            used_b.add(bj); pairs.append((int(i), int(bj), float(M[i, bj])))
    ghost = [int(i) for i in range(len(A['boxes'])) if i not in {p[0] for p in pairs}]
    missed = [j for j in range(len(B['boxes'])) if j not in used_b]
    return pairs, ghost, missed


def ctr(bb):
    return (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0


def box_in_mask(bb, mask):
    """框中心是否落在掩膜内 + 框面积与掩膜的重叠比"""
    cx, cy = int(round((bb[0] + bb[2]) / 2)), int(round((bb[1] + bb[3]) / 2))
    h, w = mask.shape
    inside = bool(mask[min(max(cy, 0), h - 1), min(max(cx, 0), w - 1)])
    x1, y1 = int(max(bb[0], 0)), int(max(bb[1], 0))
    x2, y2 = int(min(bb[2], w)), int(min(bb[3], h))
    sub = mask[y1:y2, x1:x2]
    return inside, (float(sub.mean()) if sub.size else 0.0)


def luma(x):
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def box_luma_ratio(img, bb):
    """框内平均亮度 / 全图亮度中位数（<0.85 视为「框压在一块暗斑上」= 人影诊断）"""
    l = luma(img)
    x1, y1 = int(max(bb[0], 0)), int(max(bb[1], 0))
    x2, y2 = int(min(bb[2], l.shape[1])), int(min(bb[3], l.shape[0]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return float(l[y1:y2, x1:x2].mean() / max(float(np.median(l)), 1e-6))


def rd(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


class _Tee:
    """stdout 同时写运行日志（规范产物 /tmp/downstream_probe.log）"""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, 'w')
        self.out = sys.__stdout__

    def write(self, s):
        self.out.write(s); self.fh.write(s); self.fh.flush()
        return len(s)

    def flush(self):
        self.out.flush(); self.fh.flush()


# ---------------------------------------------------------------- 绘图
def draw_boxes(img, boxes, names, colors, scores=None, thick=2):
    out = img.copy()
    for i, bb in enumerate(boxes):
        c = colors[i] if isinstance(colors, list) else colors
        p1, p2 = (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3]))
        cv2.rectangle(out, p1, p2, c, thick)
        tag = names[i] + (f' {scores[i]:.2f}' if scores is not None else '')
        cv2.putText(out, tag, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
    return out


def draw_kp(img, kps, kp_scores=None):
    out = img.copy()
    for i, kp in enumerate(kps):
        for a, b in KP_EDGES:
            if kp[a, 2] > 0.5 and kp[b, 2] > 0.5:
                cv2.line(out, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), CLR['kp'], 3, cv2.LINE_AA)
        for j in range(kp.shape[0]):
            if kp[j, 2] > 0.5:
                cv2.circle(out, (int(kp[j, 0]), int(kp[j, 1])), 4, (0, 255, 255), -1, cv2.LINE_AA)
        if kp_scores is not None:
            x1, y1 = int(kp[:, 0].min()), int(kp[:, 1].min())
            cv2.putText(out, f'kpscore={kp_scores[i]:.2f}', (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR['kp'], 2, cv2.LINE_AA)
    return out


def tile(img, boxes=None, names=None, colors=None, scores=None, title='', h=340,
         omega=None, kps=None, kp_scores=None, band=True):
    im = img.copy()
    if omega is not None and omega.any():
        cnts, _ = cv2.findContours(omega.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, CLR['omega'], 2)
    if boxes is not None and len(boxes):
        im = draw_boxes(im, boxes, names, colors, scores)
    if kps is not None and len(kps):
        im = draw_kp(im, kps, kp_scores)
    if band:
        cv2.line(im, (0, int(BAND_CY * im.shape[0])), (im.shape[1], int(BAND_CY * im.shape[0])), CLR['band'], 1)
    s = h / im.shape[0]
    im = cv2.resize(im, (max(60, int(im.shape[1] * s)), h))
    bar = np.full((26, im.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, title[:58], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


def hstack_pad(tiles, pad=6, bg=255):
    H = max(t.shape[0] for t in tiles)
    t2 = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(bg,) * 3) for t in tiles]
    out = t2[0]
    sep = np.full((H, pad, 3), bg, np.uint8)
    for t in t2[1:]:
        out = np.hstack([out, sep, t])
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='eval/eval_downstream_0921.csv')
    ap.add_argument('--out-myvideo', default='eval/eval_downstream_0921_myvideo.csv')
    ap.add_argument('--verdict', default='eval/eval_downstream_0921_verdict.json')
    ap.add_argument('--fig', default='fig/downstream_ghost_0921.jpg')
    ap.add_argument('--myvideo-per-scene', type=int, default=16)
    ap.add_argument('--keypoint-frames', type=int, default=24, help='MyVideo 上跑 Keypoint R-CNN 的最大帧数')
    ap.add_argument('--skip-keypoint', action='store_true')
    ap.add_argument('--gpu', default=None, help='物理卡号，钉 CUDA_VISIBLE_DEVICES（共享机守则）')
    ap.add_argument('--log', default='/tmp/downstream_probe.log')
    args = ap.parse_args()
    sys.stdout = _Tee(args.log)

    dev = torch.device('cuda:0')
    t0 = time.time()
    print(f'[dev] {dev} {torch.cuda.get_device_name(0)}')
    w = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    frcnn = fasterrcnn_resnet50_fpn(weights=w).eval().to(dev)
    cats = w.meta['categories']
    kprcnn = None
    if not args.skip_keypoint:
        kprcnn = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(dev)
    print(f'[model] ready ({time.time()-t0:.1f}s), CONF={CONF} IOU={IOU} 预登记门已固定')

    rows, state, ctrl_rows = [], [], []
    gt_cache, gtkp_cache = {}, {}

    # ============ Part 1: apr / may 配对三态 ============
    for tag, (d, predroot) in PAIRS.items():
        for sc in sorted(x for x in os.listdir(rd(d)) if x.startswith('VID_')):
            sd = rd(os.path.join(d, sc))
            GT = cv2.imread(os.path.join(sd, 'GT.png'))
            if GT is None:
                continue
            if sc not in gt_cache:
                B_all = frcnn_detect(frcnn, cats, GT, dev, CONF_LO)
                gt_cache[sc] = dict(B=_sub(B_all, CONF), allo=len(B_all['boxes']))
                if kprcnn is not None:
                    kk = kprcnn_detect(kprcnn, GT, dev, CONF)
                    gtkp_cache[sc] = kk
            B = gt_cache[sc]['B']
            Bkp = gtkp_cache.get(sc)
            scene_det = []      # [(fn, A)] 供 I-vs-I 闪烁对照
            for fn in sorted(f for f in os.listdir(sd) if f.startswith('I_')):
                I = cv2.imread(os.path.join(sd, fn))
                if I is None or I.shape != GT.shape:
                    print(f'[skip] {tag}/{sc}/{fn} 尺寸不一致')
                    continue
                key = f'{sc}__{fn[:-4]}'
                o = omega_of(I, GT)
                Om = o['Om'] if o is not None else None
                A_all = frcnn_detect(frcnn, cats, I, dev, CONF_LO)
                A = _sub(A_all, CONF)
                pairs, ghost, missed = match(A, B)
                _, ghost_any, missed_any = match(A, B, class_aware=False)
                n_gin = sum(box_in_mask(A['boxes'][i], Om)[0] for i in ghost) if Om is not None else None
                gfrac = [round(box_in_mask(A['boxes'][i], Om)[1], 3) for i in ghost] if Om is not None else []
                # 第三态：修复输出
                rp = rd(os.path.join(predroot, key, REPAIR_FILE))
                R = cv2.imread(rp) if os.path.exists(rp) else None
                if R is not None and R.shape != I.shape:
                    R = cv2.resize(R, (I.shape[1], I.shape[0]))
                CR = _sub(frcnn_detect(frcnn, cats, R, dev, CONF_LO), CONF) if R is not None else None
                ncomR = nghR = nmisR = nkillR = None
                if CR is not None:
                    pR, gR, mR = match(CR, B)
                    ncomR, nghR, nmisR = len(pR), len(gR), len(mR)
                    MIG = iou_mat(A['boxes'][ghost], CR['boxes'])
                    nkillR = int(sum(1 for i in range(len(ghost)) if MIG.shape[1] == 0 or MIG[i].max() < IOU))
                kpI = kprcnn_detect(kprcnn, I, dev, CONF) if kprcnn is not None else None
                rec = dict(part='pair', set=tag, scene=sc, frame=fn, key=key,
                           n_det_I=len(A['boxes']), n_det_GT=len(B['boxes']),
                           n_det_R=(len(CR['boxes']) if CR is not None else ''),
                           n_common=len(pairs), n_ghost=len(ghost), n_missed=len(missed),
                           common_classes='|'.join(A['names'][i] for i, _, _ in pairs),
                           ghost_classes='|'.join(A['names'][i] for i in ghost),
                           missed_classes='|'.join(B['names'][j] for j in missed),
                           ghost_max_score=round(max([float(A['scores'][i]) for i in ghost], default=0.0), 3),
                           missed_max_score=round(max([float(B['scores'][j]) for j in missed], default=0.0), 3),
                           omega_pct=(round(100 * float(Om.mean()), 2) if Om is not None else ''),
                           n_ghost_in_omega=(n_gin if n_gin is not None else ''),
                           ghost_omega_fracs='|'.join(str(x) for x in gfrac),
                           n_ghost_any_cls=len(ghost_any), n_missed_any_cls=len(missed_any),
                           n_common_R=ncomR, n_ghost_R=nghR, n_missed_R=nmisR, n_ghostI_killed_by_R=nkillR,
                           kp_I_person=(len(kpI['boxes']) if kpI is not None else ''),
                           kp_I_kpscore=(round(float(kpI['kp_scores'].max()), 2)
                                         if kpI is not None and len(kpI['boxes']) else
                                         (0.0 if kpI is not None else '')),
                           kp_GT_person=(len(Bkp['boxes']) if Bkp is not None else
                                         (0 if kprcnn is not None else '')),
                           n_det_I_conf03=len(A_all['boxes']), n_det_GT_conf03=gt_cache[sc]['allo'])
                rows.append(rec)
                scene_det.append((fn, A))
                state.append(dict(tag=tag, sc=sc, fn=fn, key=key, n_ghost=len(ghost), n_missed=len(missed),
                                  ghost_max=rec['ghost_max_score'], A=A, B=B, ghost=ghost, missed=missed,
                                  nkillR=nkillR, n_det_R=rec['n_det_R'], kp_n=(len(kpI['boxes']) if kpI is not None else '')))
                print(f"[pair] {tag}/{key}: I={rec['n_det_I']} GT={rec['n_det_GT']} R={rec['n_det_R']} "
                      f"common={rec['n_common']} ghost={rec['n_ghost']}({rec['ghost_classes']}) "
                      f"missed={rec['n_missed']}({rec['missed_classes']}) Ω={rec['omega_pct']}% "
                      f"ghost∈Ω={rec['n_ghost_in_omega']} kp={rec['kp_I_person']}")
            # 闪烁对照：同场景相邻 I 帧互配（不涉及 GT）——判「I vs GT 的差异是否只是检测抖动」
            for (fa, Aa), (fb, Ab) in zip(scene_det, scene_det[1:]):
                p2, g2, m2 = match(Aa, Ab)
                ctrl_rows.append(dict(part='ctrl_I_vs_I', set=tag, scene=sc, frame=f'{fa}|{fb}',
                                      n_det_I=len(Aa['boxes']), n_det_GT=len(Ab['boxes']),
                                      n_common=len(p2), n_ghost=len(g2), n_missed=len(m2)))

    # ============ Part 2: MyVideo 基线（无 GT，仅看检出位置）============
    mv_rows, mv_cand = [], []
    scs = sorted(x for x in os.listdir(rd(MYVIDEO)) if x.startswith('VID_'))
    for sc in scs:
        fs = sorted(f for f in os.listdir(rd(os.path.join(MYVIDEO, sc))) if f.endswith('.jpg'))
        if not fs:
            continue
        for i in np.unique(np.linspace(0, len(fs) - 1, min(args.myvideo_per_scene, len(fs))).astype(int)):
            fn = fs[i]
            img = cv2.imread(rd(os.path.join(MYVIDEO, sc, fn)))
            if img is None:
                continue
            H, W = img.shape[:2]
            D = _sub(frcnn_detect(frcnn, cats, img, dev, CONF_LO), CONF)
            persons = [j for j in range(len(D['boxes'])) if D['names'][j] == 'person']
            lower, ratios = [], []
            for j in persons:
                cx, cy = ctr(D['boxes'][j])
                rcx, rcy = cx / W, cy / H
                if BAND_CX[0] <= rcx <= BAND_CX[1] and rcy >= BAND_CY:
                    lower.append((rcx, rcy, float(D['scores'][j]), int(BAND_CX[0] <= rcx <= BAND_CX[1] and rcy >= BAND_CY_SENS)))
                    r = box_luma_ratio(img, D['boxes'][j])
                    if r is not None:
                        ratios.append(round(r, 3))
            mv_rows.append(dict(part='myvideo', set='myvideo', scene=sc, frame=fn,
                                n_det=len(D['boxes']), n_person=len(persons), n_person_lower=len(lower),
                                person_lower_flag=int(bool(lower)),
                                person_lower_flag_sens=int(any(x[3] for x in lower)),
                                person_lower_scores='|'.join(f'{x[2]:.2f}' for x in lower),
                                person_lower_cxcy='|'.join(f'{x[0]:.2f},{x[1]:.2f}' for x in lower),
                                box_luma_ratio_min=(min(ratios) if ratios else ''),
                                all_classes='|'.join(sorted(set(D['names'])))))
            if lower:
                mv_cand.append(dict(score=max(x[2] for x in lower), sc=sc, fn=fn, n_lower=len(lower),
                                    luma_ratio=(min(ratios) if ratios else 1.0)))
    nfl = sum(r['person_lower_flag'] for r in mv_rows)
    print(f'[myvideo] 抽样 {len(mv_rows)} 帧 / {len(scs)} 场景；中下部人形框出现在 {nfl} 帧 '
          f'({nfl/max(len(mv_rows),1):.1%})')

    # Keypoint 复核：中下部人形分最高的若干 MyVideo 帧
    mv_kp = {}
    if kprcnn is not None and mv_cand:
        ranked = sorted(mv_cand, key=lambda d: (-d['score'], d['luma_ratio']))[:args.keypoint_frames]
        for cand in ranked:
            sc, fn = cand['sc'], cand['fn']
            mv_kp[(sc, fn)] = kprcnn_detect(kprcnn, cv2.imread(rd(os.path.join(MYVIDEO, sc, fn))), dev, CONF)
        print(f'[myvideo/kp] 复核 {len(mv_kp)} 帧，其中 {sum(1 for v in mv_kp.values() if len(v["boxes"]))} 帧检出 person')

    # ============ 汇总 + 门控判决 ============
    verdict = dict(prereg=dict(conf=CONF, iou=IOU, class_aware=True, band_cy=BAND_CY,
                               scene_consistency='>=2/3 场景同方向', frame_prevalence='>=1/3 人影帧',
                               redline='jun 未使用'), data={}, gate={})
    for tag in PAIRS:
        rr = [r for r in rows if r['set'] == tag]
        scenes = sorted({r['scene'] for r in rr})
        per_scene = {}
        for sc in scenes:
            sr = [r for r in rr if r['scene'] == sc]
            gh, ms = sum(r['n_ghost'] for r in sr), sum(r['n_missed'] for r in sr)
            per_scene[sc] = dict(n_frames=len(sr), ghost=gh, missed=ms,
                                 ghost_in_omega=sum(r['n_ghost_in_omega'] for r in sr if r['n_ghost_in_omega'] != ''),
                                 omega_valid=any(r['omega_pct'] != '' for r in sr),
                                 direction=('ghost' if gh > ms else 'missed' if ms > gh else 'none'))
        ns = len(scenes)
        sg = dict(n_frames=len(rr), n_scenes=ns,
                  n_common=sum(r['n_common'] for r in rr), n_ghost=sum(r['n_ghost'] for r in rr),
                  n_missed=sum(r['n_missed'] for r in rr),
                  n_ghost_in_omega=sum(r['n_ghost_in_omega'] for r in rr if r['n_ghost_in_omega'] != ''),
                  n_omega_valid=sum(1 for r in rr if r['omega_pct'] != ''),
                  n_frames_ghost=sum(1 for r in rr if r['n_ghost'] > 0),
                  n_frames_missed=sum(1 for r in rr if r['n_missed'] > 0),
                  n_frames_any_det=sum(1 for r in rr if r['n_det_I'] + r['n_det_GT'] > 0),
                  n_ghost_person=sum(1 for r in rr for c in r['ghost_classes'].split('|') if c == 'person'),
                  ghost_class_hist=dict(Counter(c for r in rr for c in r['ghost_classes'].split('|') if c)),
                  missed_class_hist=dict(Counter(c for r in rr for c in r['missed_classes'].split('|') if c)),
                  n_ghostI_killed_by_R=sum(r['n_ghostI_killed_by_R'] for r in rr if r['n_ghostI_killed_by_R'] != ''),
                  n_ghost_R=sum(r['n_ghost_R'] for r in rr if r['n_ghost_R'] != ''),
                  n_common_R=sum(r['n_common_R'] for r in rr if r['n_common_R'] != ''),
                  n_missed_R=sum(r['n_missed_R'] for r in rr if r['n_missed_R'] != ''),
                  n_frames_kpI_person=sum(1 for r in rr if r['kp_I_person'] != '' and r['kp_I_person'] > 0),
                  n_frames_kpGT_person=sum(1 for r in rr if r['kp_GT_person'] != '' and r['kp_GT_person'] > 0),
                  per_scene=per_scene)
        # Ω 归属补充口径：ghost 框面积与 Ω 的重叠比（主口径仍是「框中心 ∈ Ω」）
        _fr = [float(x) for r in rr if r['ghost_omega_fracs'] for x in r['ghost_omega_fracs'].split('|')]
        sg['ghost_omega_overlap_mean'] = round(float(np.mean(_fr)), 3) if _fr else None
        sg['n_ghost_omega_overlap_ge030'] = int(sum(1 for x in _fr if x >= 0.30))
        sg['n_scene_ghost_majority'] = sum(1 for v in per_scene.values() if v['direction'] == 'ghost')
        sg['n_scene_missed_majority'] = sum(1 for v in per_scene.values() if v['direction'] == 'missed')
        sg['scene_consistent_ghost'] = bool(sg['n_scene_ghost_majority'] / ns >= 2 / 3 - 1e-9)
        sg['scene_consistent_missed'] = bool(sg['n_scene_missed_majority'] / ns >= 2 / 3 - 1e-9)
        sg['frame_prevalent_ghost'] = bool(sg['n_frames_ghost'] / len(rr) >= 1 / 3 - 1e-9)
        sg['frame_prevalent_missed'] = bool(sg['n_frames_missed'] / len(rr) >= 1 / 3 - 1e-9)
        # 口径 B（出现即算，不做主次比较）：场景内 ≥1 个 ghost / 漏检 的场景数
        sg['n_scene_with_geq1_ghost'] = sum(1 for v in per_scene.values() if v['ghost'] >= 1)
        sg['n_scene_with_geq1_missed'] = sum(1 for v in per_scene.values() if v['missed'] >= 1)
        sg['recurrence_ghost_B'] = bool(sg['n_scene_with_geq1_ghost'] / ns >= 2 / 3 - 1e-9)
        sg['recurrence_missed_B'] = bool(sg['n_scene_with_geq1_missed'] / ns >= 2 / 3 - 1e-9)
        sg['harm_ghost_readingB'] = bool(sg['recurrence_ghost_B'] and sg['frame_prevalent_ghost'])
        sg['harm_missed_readingB'] = bool(sg['recurrence_missed_B'] and sg['frame_prevalent_missed'])
        # 闪烁对照：I vs GT 的每帧平均差异框数 对比 I vs I（同场景相邻帧）的每对平均差异框数
        cr = [c for c in ctrl_rows if c['set'] == tag]
        sg['ctrl_n_pairs'] = len(cr)
        sg['ctrl_mean_diff_per_pair'] = round(float(np.mean([(c['n_ghost'] + c['n_missed']) / 2 for c in cr])), 3) if cr else None
        sg['mean_diff_per_frame_IGT'] = round(float(np.mean([(r['n_ghost'] + r['n_missed']) / 2 for r in rr])), 3)
        sg['harm_ghost'] = bool(sg['scene_consistent_ghost'] and sg['frame_prevalent_ghost'])
        sg['harm_missed'] = bool(sg['scene_consistent_missed'] and sg['frame_prevalent_missed'])
        sg['harm'] = bool(sg['harm_ghost'] or sg['harm_missed'])
        sg['weak_sensitivity'] = bool(sg['n_frames_any_det'] < len(rr) / 3)
        verdict['data'][tag] = sg
        print(f"[verdict/{tag}] frames={sg['n_frames']} common={sg['n_common']} ghost={sg['n_ghost']} "
              f"missed={sg['n_missed']} ghost∈Ω={sg['n_ghost_in_omega']}/{sg['n_ghost']}(Ω有效{sg['n_omega_valid']}帧 "
              f"| 重叠≥0.3: {sg['n_ghost_omega_overlap_ge030']}, 均{ sg['ghost_omega_overlap_mean']}) "
              f"场景同向: ghost {sg['n_scene_ghost_majority']}/{ns} missed {sg['n_scene_missed_majority']}/{ns} "
              f"| 帧占比: ghost {sg['n_frames_ghost']}/{len(rr)} missed {sg['n_frames_missed']}/{len(rr)} "
              f"| HARM_ghost={sg['harm_ghost']} HARM_missed={sg['harm_missed']}"
              f"{' [WEAK]' if sg['weak_sensitivity'] else ''}")
        print(f"   口径B(出现即算): ≥1ghost 场景 {sg['n_scene_with_geq1_ghost']}/{ns} → harm={sg['harm_ghost_readingB']}；"
              f"≥1漏检 场景 {sg['n_scene_with_geq1_missed']}/{ns} → harm={sg['harm_missed_readingB']}"
              f" | 闪烁对照 I-vs-I 每对差异框 {sg['ctrl_mean_diff_per_pair']}（{sg['ctrl_n_pairs']} 对）"
              f" vs I-vs-GT 每帧 {sg['mean_diff_per_frame_IGT']}")

    nf = len(mv_rows)
    fl = sum(r['person_lower_flag'] for r in mv_rows)
    lrs = [r['box_luma_ratio_min'] for r in mv_rows if r['box_luma_ratio_min'] != '']
    verdict['data']['myvideo'] = dict(
        n_frames_sampled=nf, n_scenes=len(scs), n_frames_person_lower=fl,
        rate_frames_person_lower=round(fl / max(nf, 1), 3),
        n_frames_person_lower_sens=sum(r['person_lower_flag_sens'] for r in mv_rows),
        n_frames_any_person=sum(1 for r in mv_rows if r['n_person'] > 0),
        n_frames_any_det=sum(1 for r in mv_rows if r['n_det'] > 0),
        kp_frames_checked=len(mv_kp), kp_frames_with_person=sum(1 for v in mv_kp.values() if len(v['boxes'])),
        median_box_luma_ratio=(round(float(np.median(lrs)), 3) if lrs else None),
        n_frames_box_on_dark_blob=sum(1 for x in lrs if x < 0.85))
    print(f"[verdict/myvideo] {nf} 帧中下部人形 {fl} 帧 ({fl/max(nf,1):.1%})；"
          f"暗斑占比中位 {verdict['data']['myvideo']['median_box_luma_ratio']}；"
          f"Keypoint 复核 {len(mv_kp)} 帧检出 {verdict['data']['myvideo']['kp_frames_with_person']} 帧")

    verdict['gate'] = dict(
        reading_A_scene_dominance={t: bool(verdict['data'][t]['harm']) for t in PAIRS},
        reading_B_scene_recurrence={t: bool(verdict['data'][t]['harm_ghost_readingB']
                                            or verdict['data'][t]['harm_missed_readingB']) for t in PAIRS},
        harm_overall=bool(verdict['data']['apr']['harm'] or verdict['data']['may']['harm']),
        note='伤害=跨场景一致(≥2/3)+人影帧占比≥1/3。口径A=场景内 ghost/漏检 谁多定方向（严格）；'
             '口径B=现象在该场景出现≥1次即算（出现即算）。两口径均可由 CSV 机械复算，结论对口径敏感处已注明')
    verdict['runtime_s'] = round(time.time() - t0, 1)

    # ============ 落盘：CSV / JSON ============
    for path, rr in ((args.out, rows + ctrl_rows), (args.out_myvideo, mv_rows)):
        p = rd(path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        keys = sorted({k for r in rr for k in r})
        with open(p, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=keys, restval='')
            wr.writeheader(); wr.writerows(rr)
        print(f'[out] {p} ({len(rr)} 行)')
    with open(rd(args.verdict), 'w') as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=1, default=str)
    print(f'[out] {rd(args.verdict)}')

    # ============ 落盘：可视化拼图（3 行 × 4 列）============
    rows_img = []
    for tag in PAIRS:
        cand = [s for s in state if s['tag'] == tag]
        cand.sort(key=lambda s: (-s['n_ghost'], -s['ghost_max'], -s['n_missed']))
        if not cand:
            continue
        s = cand[0]
        d, predroot = PAIRS[tag]
        I = cv2.imread(rd(os.path.join(d, s['sc'], s['fn'])))
        GT = cv2.imread(rd(os.path.join(d, s['sc'], 'GT.png')))
        Om = omega_of(I, GT)
        A, B = s['A'], s['B']
        cols = [tile(I, A['boxes'], [n + ('★ghost' if i in s['ghost'] else '') for i, n in enumerate(A['names'])],
                     [CLR['ghost'] if i in s['ghost'] else CLR['common'] for i in range(len(A['names']))],
                     list(A['scores']),
                     f'{tag} I 含人影 | ghost={len(s["ghost"])} 落Ω内='
                     f'{sum(box_in_mask(A["boxes"][i], Om["Om"])[0] for i in s["ghost"]) if Om else "NA"}',
                     omega=(Om['Om'] if Om else None))]
        cols.append(tile(GT, B['boxes'], B['names'], CLR['common'], list(B['scores']),
                         f'{tag} GT 无人影参照 | det={len(B["boxes"])}'))
        Rp = rd(os.path.join(predroot, s['key'], REPAIR_FILE))
        R = cv2.imread(Rp) if os.path.exists(Rp) else None
        if R is not None:
            if R.shape != I.shape:
                R = cv2.resize(R, (I.shape[1], I.shape[0]))
            CR = _sub(frcnn_detect(frcnn, cats, R, dev, CONF_LO), CONF)
            _, gR, _ = match(CR, B)
            cols.append(tile(R, CR['boxes'], [n + ('★ghost' if i in gR else '') for i, n in enumerate(CR['names'])],
                             [CLR['ghost'] if i in gR else CLR['common'] for i in range(len(CR['names']))],
                             list(CR['scores']), f'{tag} 修复态 R | ghost={len(gR)} (I 的 {len(s["ghost"])} 个消掉 {s["nkillR"]})'))
        else:
            cols.append(tile(np.full((640, 800, 3), 240, np.uint8), title=f'{tag} 修复态缺失', band=False))
        # 骨架证据：若选例本身无人形骨架，换本集内骨架检出最好的帧（标题标出帧名）
        kp_src = s
        if not s['kp_n']:
            alt = [x for x in state if x['tag'] == tag and x['kp_n']]
            if alt:
                kp_src = max(alt, key=lambda x: x['kp_n'])
        kp_img = I if kp_src is s else cv2.imread(rd(os.path.join(PAIRS[tag][0], kp_src['sc'], kp_src['fn'])))
        kpI = kprcnn_detect(kprcnn, kp_img, dev, CONF) if kprcnn is not None else None
        kp_title = f'{tag} Keypoint R-CNN | {kp_src["fn"]} person={len(kpI["boxes"])}' if kpI is not None else 'skip'
        if kpI is not None and len(kpI['boxes']):
            kp_title += f' kpS={float(kpI["kp_scores"].max()):.1f}'
        cols.append(tile(kp_img, None, [], [], None,
                         kp_title,
                         kps=(kpI['keypoints'] if kpI is not None and len(kpI['boxes']) else None),
                         kp_scores=(kp_person_scores(kpI['kp_scores']) if kpI is not None and len(kpI['boxes']) else None)))
        rows_img.append(hstack_pad(cols))
        print(f'[fig] {tag} 选例 {s["key"]} ghost={s["n_ghost"]}')
    # MyVideo 选例：优先「框落在暗斑上」（luma_ratio 低）且分数高的人形框
    cols = []
    for cand in sorted(mv_cand, key=lambda d: (d['luma_ratio'], -d['score']))[:2]:
        sc, fn = cand['sc'], cand['fn']
        img = cv2.imread(rd(os.path.join(MYVIDEO, sc, fn)))
        D = _sub(frcnn_detect(frcnn, cats, img, dev, CONF_LO), CONF)
        cols.append(tile(img, D['boxes'], D['names'],
                         [CLR['ghost'] if n == 'person' else CLR['common'] for n in D['names']], list(D['scores']),
                         f'MyVideo {sc[-6:]} {fn} | person={sum(1 for n in D["names"] if n == "person")}'
                         f' 框内亮度比={cand["luma_ratio"]:.2f}'))
        k = mv_kp.get((sc, fn))
        cols.append(tile(img, None, [], [], None,
                         f'MyVideo {sc[-6:]} {fn} Keypoint | person={len(k["boxes"]) if k else 0}',
                         kps=(k['keypoints'] if k and len(k['boxes']) else None),
                         kp_scores=(kp_person_scores(k['kp_scores']) if k and len(k['boxes']) else None)))
    while cols:
        rows_img.append(hstack_pad(cols[:4]))
        cols = cols[4:]
    W = max(r.shape[1] for r in rows_img)
    rows_img = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows_img]
    fig = np.vstack([np.vstack([r, np.full((8, W, 3), 255, np.uint8)]) for r in rows_img])
    fp = rd(args.fig)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    cv2.imwrite(fp, fig, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'[out] {fp} {fig.shape}')
    print(f'[done] {verdict["runtime_s"]}s  HARM={verdict["gate"]["harm_overall"]}')


if __name__ == '__main__':
    main()
