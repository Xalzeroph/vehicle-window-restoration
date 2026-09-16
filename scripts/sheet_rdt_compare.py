#!/usr/bin/env python3
"""S1 结果拼图：给定测试集与档位，出 [输入 / GT / 各模型 T̂(或 R̂)] 逐场景对照表。

用法:
  python3 scripts/sheet_rdt_compare.py --set synth_rrw_v21_0913 --level alpha_0.3 \
      --tags zeroshot,pilotB_r2,freeze,plain --scenes t02,t05,t09,t10 \
      --out fig/rdt_s1_v21_t_alpha03.png
  python3 scripts/sheet_rdt_compare.py --kind r ...   # R̂ 对照（GT 换测试集 r/）
每格右下角标该图 raw PSNR（vs GT / vs R 源）。
"""
import argparse, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
NAME = {"zeroshot": "RDNet zero-shot", "pilotB_r2": "pilot B_r2 (4k)", "freeze": "S1 freeze (w)",
        "freeze_s101": "S1 freeze s101", "plain": "S1 plain", "plain_s101": "S1 plain s101"}


def psnr(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="synth_rrw_v21_0913")
    ap.add_argument("--level", default="alpha_0.3")
    ap.add_argument("--tags", default="zeroshot,pilotB_r2,freeze,plain")
    ap.add_argument("--scenes", default="t02,t05,t09,t10")
    ap.add_argument("--kind", choices=["t", "r"], default="t")
    ap.add_argument("--tile-w", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = os.path.join(PROJ, "data/test_sets", args.set)
    scenes = [s if s.endswith(".png") else s + ".png" for s in args.scenes.split(",")]
    tags = args.tags.split(",")
    font = ImageFont.truetype(FONT_PATH, 15)
    font_s = ImageFont.truetype(FONT_PATH, 13)

    tw = args.tile_w
    th = round(tw * 384 / 832)
    head, lab_w, pad = 30, 46, 4
    cols = 2 + len(tags)                       # 输入 + GT + 各模型
    W = lab_w + cols * (tw + pad) + pad
    H = head + len(scenes) * (th + pad) + pad
    sheet = Image.new("RGB", (W, H), (22, 22, 26))
    d = ImageDraw.Draw(sheet)

    headers = (["Input I"] if args.kind == "t" else ["Reflect R"]) + ["GT T" if args.kind == "t" else "R src"]
    headers += [NAME.get(t, t) for t in tags]
    for c, htxt in enumerate(headers):
        d.text((lab_w + c * (tw + pad) + pad, 8), htxt, font=font, fill=(235, 235, 235))

    for r, sc in enumerate(scenes):
        y = head + r * (th + pad)
        d.text((6, y + th // 2 - 8), os.path.splitext(sc)[0], font=font, fill=(235, 235, 235))
        if args.kind == "t":
            ref_p = os.path.join(root, "gt", sc)
            inp_p = os.path.join(root, "levels", args.level, sc)
        else:
            ref_p = os.path.join(root, "r", sc)
            inp_p = os.path.join(root, "levels", args.level, sc)
        ref = np.asarray(Image.open(ref_p).convert("RGB"))
        tiles = [(inp_p, None), (ref_p, None)]
        for t in tags:
            p = os.path.join(PROJ, "runs/pred", f"rdt_{t}", args.set, args.level,
                             os.path.splitext(sc)[0], "t_hat.png" if args.kind == "t" else "r_hat.png")
            tiles.append((p, None))
        for c, (p, _) in enumerate(tiles):
            im = Image.open(p).convert("RGB")
            v = psnr(np.asarray(im), ref)
            im = im.resize((tw, th), Image.LANCZOS)
            x = lab_w + c * (tw + pad) + pad
            sheet.paste(im, (x, y))
            d.text((x + 4, y + th - 18), f"{v:.1f}" if v != float("inf") else "inf",
                   font=font_s, fill=(255, 240, 120), stroke_width=2, stroke_fill=(0, 0, 0))

    os.makedirs(os.path.dirname(os.path.join(PROJ, args.out)), exist_ok=True)
    sheet.save(os.path.join(PROJ, args.out))
    print(f"-> {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
