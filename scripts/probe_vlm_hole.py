#!/usr/bin/env python3
"""P3 多模态感知探针：VLM 能否"看出端倪"
T1 命名/定位：反射物体 + 5x5 格号  → 对用户 ROI 的命中率与 IoU
T2 判别：I vs GT 二选一哪张有反射  → 准确率（随机顺序）
T3 洞区语义：ROI 里被挡的是什么/背后是什么 → 与 GT 的 SegFormer 语义比对
用法：CUDA_VISIBLE_DEVICES=2 python3 scripts/probe_vlm_hole.py --model qwen --out eval/vlm_probe_qwen.csv
"""
import argparse, csv, json, os, re, sys, time
import cv2, numpy as np, torch
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 5
CELL_RE = re.compile(r'\b([A-E])([1-5])\b')

P1 = """这是一张隔着深色贴膜车窗拍摄的照片，画面上叠加了 5×5 网格（列 A–E 从左到右，行 1–5 从上到下）。
图中有些暗色或异常的东西是【玻璃反射】造成的，并不属于玻璃后面的真实场景，例如：人的身体部位/手/手机、树叶枝条、三脚架、灯影。
请你只找出【属于反射、不属于真实场景】的物体，对每个物体给出类别和它覆盖的网格格号。
严格按 JSON 数组输出，不要解释：
[{"obj":"树叶","cells":["B3","C3"]},{"obj":"三脚架","cells":["D3"]}]
如果看不出任何反射物体，输出 []。"""

P2 = """下面两张照片拍的是同一个场景、同一机位、同一曝光：
图 A 是隔着车窗直接拍的；图 B 是把车窗反光遮挡后拍的（干净透射）。
请判断：哪一张照片里存在【玻璃反射造成的多余亮斑或暗影】（不属于场景本身的东西）？
只回答一个字母：A 或 B。"""

P3 = """这张裁剪图来自隔着深色贴膜车窗拍摄的照片，其中这块区域被玻璃反射物（如人影/手/树叶/三脚架）遮挡了一部分。
请回答两点，每点一句话：
1) 遮挡这块区域的是什么？
2) 这块区域【背后真实的场景】最可能是什么？（例如：地面、墙面、箱子、桌腿……）"""


def grid_overlay(img):
    im = img.copy(); H, W = im.shape[:2]
    for j in range(1, N):
        cv2.line(im, (j*W//N, 0), (j*W//N, H), (0, 255, 255), 2)
        cv2.line(im, (0, j*H//N), (W, j*H//N), (0, 255, 255), 2)
    for r in range(N):
        for c in range(N):
            cv2.putText(im, f'{chr(65+c)}{r+1}', (c*W//N+6, r*H//N+28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 5)
            cv2.putText(im, f'{chr(65+c)}{r+1}', (c*W//N+6, r*H//N+28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    return im


def load(tag):
    from transformers import AutoProcessor
    if tag == 'qwen':
        from transformers import Qwen2_5_VLForConditionalGeneration
        repo = 'Qwen/Qwen2.5-VL-3B-Instruct'
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
        return ('qwen', m, AutoProcessor.from_pretrained(repo))
    if tag == 'smolvlm':
        from transformers import AutoModelForVision2Seq
        repo = 'HuggingFaceTB/SmolVLM-Instruct'
        m = AutoModelForVision2Seq.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
        return ('smolvlm', m, AutoProcessor.from_pretrained(repo))
    if tag == 'internvl':
        from transformers import AutoModel, AutoTokenizer
        repo = 'OpenGVLab/InternVL2-2B'
        m = AutoModel.from_pretrained(repo, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                     device_map='cuda:0').eval()
        return ('internvl', m, AutoTokenizer.from_pretrained(repo, trust_remote_code=True))
    raise SystemExit('unknown model ' + tag)


def ask(kind, m, proc, imgs, prompt, max_new=320):
    """imgs: list of BGR uint8"""
    if kind == 'internvl':
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        tf = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                        T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        px = torch.stack([tf(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in imgs]).to(torch.bfloat16).cuda()
        if len(imgs) == 1:
            q = f'<image>\n{prompt}'
        else:
            q = ''.join('<image>\n' for _ in imgs) + prompt
        with torch.no_grad():
            return m.chat(proc, px, q, dict(max_new_tokens=max_new, do_sample=False))
    msgs = [{'role': 'user', 'content': [{'type': 'image'} for _ in imgs] + [{'type': 'text', 'text': prompt}]}]
    if kind == 'qwen':
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs], return_tensors='pt').to('cuda:0')
    else:
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=text, images=[cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs], return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
    gen = out[:, inp['input_ids'].shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()


def parse_cells(txt):
    try:
        s = txt[txt.index('['): txt.rindex(']')+1]
        arr = json.loads(s)
        cells = []
        for o in arr:
            for c in (o.get('cells') or []):
                if CELL_RE.fullmatch(str(c).strip().upper()):
                    cells.append(str(c).strip().upper())
        return cells, arr
    except Exception:
        return CELL_RE.findall(txt) and [f'{a}{b}' for a, b in CELL_RE.findall(txt)] or [], []


def cellmask(shape, cells):
    H, W = shape; m = np.zeros((H, W), bool)
    for c in cells:
        col, row = ord(c[0])-65, int(c[1])-1
        m[row*H//N:(row+1)*H//N, col*W//N:(col+1)*W//N] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['qwen', 'smolvlm', 'internvl'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', default='all')
    ap.add_argument('--tasks', default='T1,T2,T3')
    a = ap.parse_args()
    J = json.load(open(f'{PROJ}/data/annot_hole_cells.json', encoding='utf-8'))
    keys = [k for k, v in J.items() if v.get('pos')]
    if a.frames != 'all':
        keys = keys[:int(a.frames)]
    print(f'[{a.model}] 帧数 {len(keys)} 任务 {a.tasks}', flush=True)
    kind, m, proc = load(a.model)
    try:
        from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
        seg = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/segformer-b0-finetuned-ade-512-512').to('cuda:0').eval()
        spr = AutoImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
        id2label = seg.config.id2label
    except Exception as e:
        seg = None; print('[warn] SegFormer 不可用（T3 参照缺失）:', e, flush=True)
    tasks = a.tasks.split(',')
    rows = []
    for i, k in enumerate(keys):
        sname, case = k.split('/')
        z = np.load(f'{PROJ}/data/gate_v2/lovo_k2/{sname}/{case}.npz')
        I, T = z['I'], z['T']; H, W = I.shape[:2]
        user = J[k]['pos']
        rec = dict(model=a.model, key=k)
        t0 = time.time()
        if 'T1' in tasks:
            try:
                txt = ask(kind, m, proc, [grid_overlay(I)], P1)
                pred, arr = parse_cells(txt)
                um = cellmask((H, W), user); pm = cellmask((H, W), pred)
                rec['T1_txt'] = txt.replace('\n', ' ')[:400]
                rec['T1_pred'] = '|'.join(pred)
                rec['T1_iou'] = round((um & pm).sum()/max((um | pm).sum(), 1), 3)
                rec['T1_prec'] = round(len([c for c in pred if c in user])/max(len(pred), 1), 3)
                rec['T1_rec'] = round(len([c for c in user if c in pred])/max(len(user), 1), 3)
            except Exception as e:
                rec['T1_txt'] = f'ERR {type(e).__name__}: {str(e)[:120]}'
        if 'T2' in tasks:
            try:
                flip = (i % 2 == 1)
                imgs = [T, I] if flip else [I, T]
                txt = ask(kind, m, proc, imgs, P2, max_new=16)
                ans = 'A' if re.search(r'\bA\b', txt.upper()[:12]) else ('B' if re.search(r'\bB\b', txt.upper()[:12]) else '?')
                ref = 'B' if flip else 'A'
                rec['T2_txt'] = txt.replace('\n', ' ')[:80]
                rec['T2_ans'] = ans
                rec['T2_correct'] = int(ans == ref)
            except Exception as e:
                rec['T2_txt'] = f'ERR {type(e).__name__}: {str(e)[:120]}'
        if 'T3' in tasks and seg is not None:
            try:
                msk = cv2.imread(f'{PROJ}/data/hole_masks_v2/{sname}__{case}.png', 0)
                ys, xs = np.nonzero(msk > 0)
                y0, y1 = max(0, ys.min()-20), min(H, ys.max()+20)
                x0, x1 = max(0, xs.min()-20), min(W, xs.max()+20)
                crop = I[y0:y1, x0:x1]
                txt = ask(kind, m, proc, [crop], P3)
                rec['T3_txt'] = txt.replace('\n', ' ')[:400]
                with torch.no_grad():
                    lg = seg(**spr(images=T[y0:y1, x0:x1], return_tensors='pt').to('cuda:0')).logits
                    p = lg.softmax(1)[0].mean((1, 2))
                    top = torch.topk(p, 3).indices.tolist()
                labs = [id2label[t] for t in top]
                rec['T3_gt_sem'] = '|'.join(labs)
                low = txt.lower()
                rec['T3_hit'] = int(any(l.lower() in low for l in labs))
            except Exception as e:
                rec['T3_txt'] = f'ERR {type(e).__name__}: {str(e)[:120]}'
        rec['sec'] = round(time.time()-t0, 1)
        rows.append(rec)
        print(f"  [{i+1}/{len(keys)}] {k} T1IoU={rec.get('T1_iou','-')} T2={rec.get('T2_correct','-')} "
              f"T3hit={rec.get('T3_hit','-')} {rec['sec']}s", flush=True)
        with open(a.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({kk for r in rows for kk in r})); w.writeheader(); w.writerows(rows)
    import numpy as np
    print(f'\n===== {a.model} 汇总（n={len(rows)}）=====')
    for kk in ('T1_iou', 'T1_prec', 'T1_rec', 'T2_correct', 'T3_hit'):
        v = [r[kk] for r in rows if kk in r and isinstance(r[kk], (int, float))]
        if v:
            print(f'  {kk:12s} 均值 {np.mean(v):.3f}  (n={len(v)})')
    print(f"[写出] {a.out}")


if __name__ == '__main__':
    main()
