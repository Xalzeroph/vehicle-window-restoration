#!/usr/bin/env python3


import argparse,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image


def read_manifest(path):
    data=json.loads(path.read_text())
    if data["status"]!="RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE":
        raise ValueError("Unknown scientific status")
    if data.get("measured_automotive_realism") or not data.get("items"):
        raise ValueError("Manifest may be mislabeled as real vehicle")
    return data


def identities(manifest):
    T=Counter();R=Counter();tg=Counter();rg=Counter();p=Counter()
    for x in manifest["items"]:
        original=x["source_pair_sha256"]
        T[original[0]]+=1;R[original[1]]+=1
        tg[x["capture_group"]]+=1;rg[x["cabin_group"]]+=1
        p[x["preset"]]+=1
    return {"outdoor_orig":T,"cabin_orig":R,
            "outdoor_groups":tg,"cabin_groups":rg,"presets":p}


def summarize(manifest):
    items=manifest["items"]
    attrs=identities(manifest)
    summary={"samples":len(items)}
    for name,counts in attrs.items():
        n=np.asarray(list(counts.values()),dtype=np.float64)
        summary[name]={
            "unique":len(counts),
            "count_min":int(n.min()),
            "count_max":int(n.max()),
            "effective_population_simpson":float(n.sum()**2/(n@n)),
            "largest_group_fraction":float(n.max()/n.sum())}
    for field in ("usable_fraction","saturation_fraction","reflection_visibility_p50",
                  "reflection_visibility_p90","gradient_content_score"):
        vals=np.array([x[field] for x in items if field in x],dtype=np.float64)
        if vals.size:
            summary[field]={"median":float(np.median(vals)),
                            "p05":float(np.quantile(vals,.05)),
                            "p95":float(np.quantile(vals,.95)),
                            "max":float(vals.max())}
    summary["by_driving_group"]=dict(attrs["outdoor_groups"])
    summary["by_cabin_group"]=dict(attrs["cabin_groups"])
    return summary,attrs


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args()
    data={k:read_manifest(args.root/k/"manifest.json") for k in ("train","validation","test")}
    ids={k:identities(v) for k,v in data.items()}


    for a,b in (("train","validation"),("train","test"),("validation","test")):
        for f in ("outdoor_orig","cabin_orig","outdoor_groups","cabin_groups"):
            overlap=ids[a][f].keys() & ids[b][f].keys()
            if overlap:
                raise AssertionError(f"SOURCE LEAKAGE {a}-{b}: {f} {list(overlap)[:5]}")
    summary={k:summarize(v)[0] for k,v in data.items()}
    report={
      "status":"SOURCE GROUP LEAKAGE AND SYNTHETIC ENGINEERING AUDIT PASSED",
      "real_automotive_reflection_certified":False,
      "paired_camera_raw_calibrated":False,
      "three_folds_sha_and_capture_disjoint":True,
      "all_samples_materialized":True,
      "source_rights":"Restricted source images and local derivatives are NOT publicly redistributable",
      "fold_summary":summary,
      "limitations":[
         "reflection textures are oblique 2D cabin camera crops, NOT mirror-registered",
         "gamma inverse of 8-bit camera images is NOT common scene-linear RAW",
         "glass/PVB thickness, curvature, coatings and camera PSF unmeasured",
         "held-out test still SYNTHETIC, not a real windshield benchmark",
         "sample count is NOT number of independently captured scenes"
      ]}
    (args.root/"three_fold_source_and_quality_audit.json").write_text(
        json.dumps(report,indent=2)+"\n")
    print("PROVENANCE_AND_LEAKAGE_AUDIT_PASS",json.dumps({
         k:{field:summary[k][field]
            for field in ("samples","outdoor_orig","cabin_orig","outdoor_groups",
                          "cabin_groups","presets","reflection_visibility_p50")}
         for k in summary},indent=2),flush=True)


if __name__=="__main__":main()
