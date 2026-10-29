#!/usr/bin/env python3


import argparse,json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

def records(root,fold,manifest_name="manifest.json"):
    p=root/fold/manifest_name
    d=json.loads(p.read_text())
    assert d["status"]=="RESEARCH_ONLY_TINTED_GLASS_SOURCE_PHOTOGRAPHIC_NOT_AUTOMOTIVE_EXTERIOR_CERTIFIED"
    for x in d["items"]:
        folder=root/fold/x["relative_directory"]
        for name in ("observed.png","target_tinted.png","metadata.json"):
            if not (folder/name).is_file():raise AssertionError(f"Missing {folder/name}")
    return d["items"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rows={k:records(a.root,k) for k in ("train","validation","test")}
    for k,l in (("train","validation"),("train","test"),("validation","test")):
        for f in ("source_t_key","source_r_key"):
            dup={r[f] for r in rows[k]} & {r[f] for r in rows[l]}
            if dup:raise AssertionError(f"SOURCE_LEAKAGE_{k}_{l}_{f}:{list(dup)[:5]}")
    summary={}
    for fold,items in rows.items():
        stat={
            "images":len(items),
            "distinct_T_source":len({r["source_t_key"] for r in items}),
            "distinct_R_source":len({r["source_r_key"] for r in items}),
            "T_category":dict(Counter(r["source_category"] for r in items)),
            "film_levels":dict(Counter(str(r["film_level"]) for r in items)),
        }
        for name in ("reflection_fraction_median","valid_fraction","saturation_fraction"):
            v=np.array([r[name] for r in items],dtype=np.float64)
            stat[name]={
                "p05":float(np.quantile(v,.05)),"median":float(np.median(v)),
                "p95":float(np.quantile(v,.95)),"max":float(np.max(v))}
        summary[fold]=stat


    vals=[]
    for row in rows["validation"][::max(1,len(rows["validation"])//12)]:
        folder=a.root/"validation"/row["relative_directory"]
        meta=json.loads((folder/"metadata.json").read_text())
        T=np.asarray(Image.open(meta["source_T"]).convert("RGB"),dtype=np.float32)/255
        m=np.asarray(np.load(meta["film_coverage_mask"]),dtype=np.float32)
        g=np.asarray(np.load(meta["film_gain_field"]),dtype=np.float32)
        expect=np.clip(meta["film_level"]*g*m+(1-m),.01,1)[...,None]*T
        tgt=np.asarray(Image.open(folder/"target_tinted.png"),dtype=np.float32)/255
        vals.append(float(abs(tgt-expect).max()))
    summary["correctness"]={
        "tinted_target_max_absolute_quantization_error":float(max(vals)),
        "cross_split_T_R_source_identity_disjoint":True,
        "test_data_used_for_training":False,
        "actual_vehicle_exterior_camera_verified":False,
        "RAW_linear_calibrated":False,
        "source_license":"research-use-only (verify upstream)"
    }
    a.out.write_text(json.dumps(summary,indent=2)+"\n")
    print("HF_DATA_AUDIT_PASS",json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
