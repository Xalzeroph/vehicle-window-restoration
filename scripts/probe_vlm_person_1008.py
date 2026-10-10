#!/usr/bin/env python3
"""干净版人影探针：不泄漏候选清单，强制二选一，选项顺序按帧打散。

问法刻意不列举例词（避免模型照抄 prompt 复述类别），
并把"有/没有"两个选项的呈现顺序按帧交替，抵消位置偏置。
用法：python3 scripts/probe_vlm_person_1008.py --model qwen --out eval/person_probe_qwen.csv
"""
import argparse, csv, json, os, re, time
import cv2, numpy as np, torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 不给出任何候选类别词，只问"有没有人"。选项顺序由调用方按帧决定。
Q_HAS_FIRST = """这是隔着深色贴膜车窗拍的照片。
请问画面里有没有【人】的反光——包括人的身体、头、手、手臂，或者人拿着手机的样子？
(A) 有   (B) 没有
只回答一个字母。"""
Q_NONE_FIRST = """这是隔着深色贴膜车窗拍的照片。
请问画面里有没有【人】的反光——包括人的身体、头、手、手臂，或者人拿着手机的样子？
(A) 没有   (B) 有
只回答一个字母。"""

# 追问：若有，在哪几个格（用方位词而非格号，避免"格号"这个抽象接口本身成为障碍）
Q_WHERE = """这是隔着深色贴膜车窗拍的照片。画面被平均分成 3×3 九宫格：
上左/上中/上右、中左/中中/中右、下左/下中/下右。
请判断：画面里【人的反光】（身体/头/手/手臂/拿手机的样子）出现在哪些区域？
只输出区域名，用逗号分隔；一个都没有就输出"无"。"""


def load(tag):
    from transformers import AutoProcessor
    if tag == 'qwen':
        from transformers import Qwen2_5_VLForConditionalGeneration
        repo = 'Qwen/Qwen2.5-VL-3B-Instruct'
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            repo, dtype=torch.bfloat16, device_map='cuda:0').eval()
        return ('qwen', m, AutoProcessor.from_pretrained(repo))
    if tag == 'smolvlm':
        import transformers as tf
        repo = 'HuggingFaceTB/SmolVLM-Instruct'
        cls = None
        for name in ('AutoModelForImageTextToText', 'AutoModelForVision2Seq',
                     'Idefics3ForConditionalGeneration', 'AutoModelForCausalLM'):
            cls = getattr(tf, name, None)
            if cls is not None:
                print(f'[smolvlm] 用 {name}', flush=True); break
        m = cls.from_pretrained(repo, dtype=torch.bfloat16, device_map='cuda:0').eval()
        return ('smolvlm', m, AutoProcessor.from_pretrained(repo))
    raise SystemExit('unknown ' + tag)


def ask(kind, m, proc, img, prompt, mx=32):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    if kind == 'qwen':
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[t], images=[cv2.cvtColor(img, cv2.COLOR_BGR2RGB)],
                   return_tensors='pt').to('cuda:0')
    else:
        t = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=t, images=[cv2.cvtColor(img, cv2.COLOR_BGR2RGB)],
                   return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=mx, do_sample=False)
    gen = out[:, inp['input_ids'].shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['qwen', 'smolvlm'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--gt', default='data/person_gt_1008.json',
                    help='人工判读的人影真值 {"key": "yes"/"no"/"unsure"}')
    a = ap.parse_args()

    J = json.load(open(f'{PROJ}/data/annot_hole_cells.json', encoding='utf-8'))
    keys = [k for k, v in J.items() if v.get('pos')]
    gt = json.load(open(f'{PROJ}/{a.gt}', encoding='utf-8')) if os.path.exists(f'{PROJ}/{a.gt}') else {}
    kind, m, proc = load(a.model)
    print(f'[{a.model}] 干净人影探针 n={len(keys)}', flush=True)

    rows = []
    for i, k in enumerate(keys):
        s, c = k.split('/')
        z = np.load(f'{PROJ}/data/gate_v2/lovo_k2/{s}/{c}.npz')
        I = z['I'].copy()
        has_first = (i % 2 == 0)          # 交替选项顺序，抵消位置偏置
        prompt = Q_HAS_FIRST if has_first else Q_NONE_FIRST
        t0 = time.time()
        rec = dict(model=a.model, key=k, gt=gt.get(k, ''), opt_order='有在前' if has_first else '无在前')
        txt = ask(kind, m, proc, I, prompt)
        # 解析字母
        head = txt.upper()[:10]
        ans = 'A' if re.search(r'\bA\b', head) else ('B' if re.search(r'\bB\b', head) else '?')
        if ans == '?':
            # 兜底：中文"有/没有"
            ans = 'A' if ('有' in txt[:12] and '没有' not in txt[:12]) else ('B' if '没有' in txt[:12] or '无' in txt[:6] else '?')
        has = None
        if ans == 'A':
            has = has_first
        elif ans == 'B':
            has = (not has_first)
        rec['has_txt'] = txt.replace('\n', ' ')[:60]
        rec['ans'] = ans
        rec['has_person'] = '' if has is None else int(has)
        # 追问位置（九宫格）
        loc = ask(kind, m, proc, I, Q_WHERE, mx=48)
        rec['where_txt'] = loc.replace('\n', ' ')[:80]
        rec['sec'] = round(time.time()-t0, 1)
        rows.append(rec)
        print(f"  [{i+1}/{len(keys)}] {k[:34]:<34} gt={rec['gt']:<6} 判={rec['has_person']} "
              f"({ans}) 位置={rec['where_txt'][:34]}", flush=True)
        with open(a.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({kk for r in rows for kk in r}))
            w.writeheader(); w.writerows(rows)

    # ---- 汇总：只在与人工真值可比的行上算 ----
    print(f'\n===== {a.model} 人影判定（n={len(rows)}）=====')
    cmp_rows = [r for r in rows if r['gt'] in ('yes', 'no') and r['has_person'] != '']
    if cmp_rows:
        tp = sum(1 for r in cmp_rows if r['gt'] == 'yes' and r['has_person'] == 1)
        fn = sum(1 for r in cmp_rows if r['gt'] == 'yes' and r['has_person'] == 0)
        fp = sum(1 for r in cmp_rows if r['gt'] == 'no' and r['has_person'] == 1)
        tn = sum(1 for r in cmp_rows if r['gt'] == 'no' and r['has_person'] == 0)
        n = len(cmp_rows)
        print(f'  可比帧 {n}: 真阳性 {tp}  假阴性 {fn}  假阳性 {fp}  真阴性 {tn}')
        print(f'  准确率 {(tp+tn)/n:.3f}   正类召回 {tp/max(tp+fn,1):.3f}   正类精确 {tp/max(tp+fp,1):.3f}')
        print(f'  真值分布: yes={sum(1 for r in cmp_rows if r["gt"]=="yes")} no={sum(1 for r in cmp_rows if r["gt"]=="no")}')
        base = max(sum(1 for r in cmp_rows if r['gt'] == 'yes'),
                   sum(1 for r in cmp_rows if r['gt'] == 'no')) / n
        print(f'  多数类基线 {base:.3f}  → 模型{"未超过" if (tp+tn)/n <= base else "超过"}基线')
    else:
        print('  [无人工真值] 仅输出原始判定')
    said_yes = sum(1 for r in rows if r['has_person'] == 1)
    print(f'  模型说"有人影"的帧: {said_yes}/{len(rows)}')
    print(f'  答括号顺序: 有在前 {sum(1 for r in rows if r["opt_order"]=="有在前")} 帧')
    print(f'  位置回答里答"无"的: {sum(1 for r in rows if "无" in r["where_txt"][:4])}/{len(rows)}')
    print(f'[写出] {a.out}')


if __name__ == '__main__':
    main()
