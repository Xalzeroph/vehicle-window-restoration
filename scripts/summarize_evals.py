#!/usr/bin/env python3
"""汇总 eval_iar_*_summary.json → Markdown 对照表（G1 指标 + 基线对照）。

用法:
  python3 scripts/summarize_evals.py                      # 全部 summary
  python3 scripts/summarize_evals.py --glob '*match*'     # 只看匹配轮
  python3 scripts/summarize_evals.py --out docs/表.md
无 summary 的旧评测（只有 CSV）会提示重跑评测生成。
"""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="eval_iar_*_summary.json")
    ap.add_argument("--out", default="", help="写到文件；默认 stdout")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print("没有 summary.json —— 新版 eval_iar.py 跑过评测后才会生成")
        return
    rows = []
    for f in files:
        d = json.load(open(f))
        stem = os.path.basename(f)[len("eval_iar_"):-len("_summary.json")]
        rows.append((stem, d))

    lines = ["| 配置(encoder_tag) | 测试集 | n | mask AUC | 档位 acc | naked(模型/常数) | film spread(GT) | αMAE |",
             "|---|---|---|---|---|---|---|---|"]
    for stem, d in rows:
        lines.append(
            f"| {stem} | {d['ts']} | {d['n']} | {d['auc']:.3f} | {d['acc']:.3f} | "
            f"{d['naked']:.3f} / {d['naked_const']:.3f} | {d['spread']:.3f} ({d['spread_gt']:.3f}) | "
            + "/".join(f"{d['mae_per_lv'][k]:.3f}" for k in ("1", "0.6", "0.3")) + " |")
    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as fp:
            fp.write(text + "\n")
        print(f"-> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
