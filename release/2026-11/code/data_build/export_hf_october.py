"""把十月数据集导出成可直接推到 HuggingFace 的目录结构。

目标仓库：zmy1234567890/tintedglass-rdt-synth（作者既有 HF dataset）
布局与合作者的 loader 对齐（`train/<split>/{T,R}/<id>.png` + 448×448 掩码 .npy）：

    <out>/
      README.md                      数据卡（含隐私与许可声明）
      manifest.json                  每对：id / split / 来源 / 尺寸 / 统计
      train/rdt_1020/{I,T,R,omega_png}/<id>.png     + masks/<id>.npy (448×448, uint8)
      train/rdt_1019split/...
      test/real_pair_1019/{I,T}/<id>.png            + masks/<id>.npy
      stats/summary.json             Ω 面积、I/T 里外比值等汇总

用法：
    python export_hf_october.py --out /DataDisk/zmy_code/_hf/tintedglass_october_1029 [--only test]
"""
import argparse
import json
import os
import shutil
import statistics as st

import cv2
import numpy as np

P = os.environ.get("TINTED_ROOT", "/DataDisk/zmy_code/TintedGlass_Enhance")
MASK_SIZE = 448


def ensure(d):
    os.makedirs(d, exist_ok=True)


def save_mask_npy(om_img, dst):
    m = cv2.imread(om_img, 0)
    if m is None:
        return None
    m = cv2.resize(m, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_NEAREST)
    b = (m > 127).astype(np.uint8)
    np.save(dst, b)
    return b


def export_pool(rel, split, name, out, manifest, stats, with_mask=True):
    src = os.path.join(P, rel) if os.path.isdir(os.path.join(P, rel)) else os.path.join(P, "data", rel)
    meta_path = os.path.join(src, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
    base = os.path.join(out, split, name)
    for sub in (("I", "T", "R", "masks") if with_mask else ("I", "T")):
        ensure(os.path.join(base, sub))
    n = 0
    areas = []
    for k in sorted(meta) or sorted(os.listdir(os.path.join(src, "I"))):
        key = k if k.endswith(".png") else k + ".png"
        stem = key[:-4]
        ok = True
        for sub, dst_sub in (("I", "I"), ("T", "T"), ("R", "R")):
            s = os.path.join(src, sub, key)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(base, dst_sub, key))
            elif sub in ("I", "T"):
                ok = False
        if not ok:
            continue
        if with_mask:
            om = os.path.join(src, "omega", key)
            b = save_mask_npy(om, os.path.join(base, "masks", stem + ".npy")) if os.path.exists(om) else None
            if b is not None:
                areas.append(float(b.mean()))
        d = dict(id=f"{name}/{stem}", split=split, pool=name,
                 source=(meta.get(stem, {}) or {}).get("domain", ""),
                 scene=(meta.get(stem, {}) or {}).get("scene", ""))
        manifest.append(d)
        n += 1
    if areas:
        stats[name] = dict(n=n, omega_area_median=float(st.median(areas)),
                           omega_area_p10=float(np.percentile(areas, 10)),
                           omega_area_p90=float(np.percentile(areas, 90)))
    else:
        stats[name] = dict(n=n)
    print(f"  {split}/{name}: {n} 对" + (f"  Ω 面积中位 {st.median(areas)*100:.1f}%" if areas else ""))


def export_test_panel(out, manifest, stats):
    """冻结测试面板：15 场景，每场景 I_*.png + GT.png（无 R、无掩码）。"""
    src = os.path.join(P, "data/test_sets/real_pair_1019")
    base = os.path.join(out, "test", "real_pair_1019")
    ensure(os.path.join(base, "I"))
    ensure(os.path.join(base, "T"))
    n = 0
    for sc in sorted(os.listdir(src)):
        d = os.path.join(src, sc)
        if not os.path.isdir(d):
            continue
        gts = [f for f in os.listdir(d) if f.startswith("GT")]
        Is = sorted(f for f in os.listdir(d) if f.startswith("I_"))
        if not gts or not Is:
            continue
        shutil.copy2(os.path.join(d, Is[0]), os.path.join(base, "I", sc + ".png"))
        shutil.copy2(os.path.join(d, gts[0]), os.path.join(base, "T", sc + ".png"))
        manifest.append(dict(id=f"real_pair_1019/{sc}", split="test", pool="real_pair_1019",
                             source="", scene=sc))
        n += 1
    stats["real_pair_1019"] = dict(n=n)
    print(f"  test/real_pair_1019: {n} 场景")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/DataDisk/zmy_code/_hf/tintedglass_october_1029")
    ap.add_argument("--only", default="all", choices=["all", "test", "train"])
    a = ap.parse_args()
    ensure(a.out)
    manifest, stats = [], {}

    if a.only in ("all", "test"):
        export_test_panel(a.out, manifest, stats)
    if a.only in ("all", "train"):
        export_pool("train_sets/real_pairs_1020", "train", "rdt_1020", a.out, manifest, stats)
        export_pool("train_sets/real_pairs_1019split", "train", "rdt_1019split", a.out, manifest, stats)
        export_pool("train_sets/real_pairs_1026", "train", "rdt_1026", a.out, manifest, stats)

    json.dump(dict(
        status="RESEARCH_ONLY_TINTED_GLASS_AUTOMOTIVE_WINDOW",
        n_pairs=len(manifest),
        mask_size=MASK_SIZE,
        note=("Occlusion-region masks are AUTOMATIC proposals confirmed only by eye; "
              "they are NOT verified ground truth. See LIMITATIONS_OMEGA.md in the paper release."),
        items=manifest), open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=1)
    ensure(os.path.join(a.out, "stats"))
    json.dump(stats, open(os.path.join(a.out, "stats/summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n导出完成: {a.out}  共 {len(manifest)} 对")


if __name__ == "__main__":
    main()
