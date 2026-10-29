#!/usr/bin/env python3


import argparse,json,statistics
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np


def classify(row,visibility_high_quantile=None):
    coverage=float(row["usable_fraction"])
    saturation=float(row["saturation_fraction"])
    visibility=float(row["reflection_visibility_p50"])
    if not all(np.isfinite(x) for x in (coverage,saturation,visibility)):
        return "engineering_reject"
    if coverage<.90 or saturation>.015:
        return "engineering_reject"
    if visibility_high_quantile is None:
        raise ValueError("Reflection-tail threshold must be learned from current candidate distribution")
    if visibility>visibility_high_quantile:
        return "high_reflection_uncertain"
    return "usable_proxy"


def tier(root,*,uncertainty_quantile=.85):
    root=Path(root)
    parent=json.loads((root/"manifest.json").read_text())
    if parent["status"]!="RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE":
        raise ValueError("Refuse automatic relabeling of unknown release status")
    items=parent["items"]
    vis=np.asarray([r["reflection_visibility_p50"] for r in items],dtype=float)
    vis=vis[np.isfinite(vis)]
    if not vis.size:raise ValueError("No optical visibility")
    threshold=float(np.quantile(vis,uncertainty_quantile))
    groups=defaultdict(list)
    for row in items:
        label=classify(row,threshold)
        groups[label].append(row)
    for label,members in groups.items():
        out=dict(parent)
        out["status"]="RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE"
        out["items"]=members
        out["parent_manifest_path"]=str(root/"manifest.json")
        out["proxy_tier"]=label
        out["not_realism_certified"]=True
        out["strong_visibility_quantile"]=uncertainty_quantile
        out["strong_visibility_threshold_estimated_from_own_synthetic_data"]=threshold
        (root/f"{label}_manifest.json").write_text(json.dumps(out,indent=2)+"\n")
    report={"status":"SYNTHETIC-DISTRIBUTION ENGINEERING PRIORITY ONLY",
            "threshold":threshold,"tier_counts":{k:len(v) for k,v in groups.items()},
            "uncertainty_quantile":uncertainty_quantile,
            "unique_driving_images":len({row["source_pair_sha256"][0] for row in items}),
            "unique_cabin_images":len({row["source_pair_sha256"][1] for row in items}),
            "capture_group_count":len({row["capture_group"] for row in items}),
            "cabin_group_count":len({row["cabin_group"] for row in items}),
            "measured_real_domain_reference_available":False,
            "rationale":"High proxy reflection can be real, but with UNREGISTERED Drive&Act cabin RGB it must NOT be presented as high-fidelity calibrated automotive glass",
            "not_certified":["real glass geometry","glass spectral optics","RAW ISP","real-domain validation"]}
    (root/"engineering_tier_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("CANDIDATE_TIERS",json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--uncertainty-quantile",type=float,default=.85)
    args=p.parse_args()
    if not 0<args.uncertainty_quantile<1:raise ValueError("Invalid quantile")
    tier(args.root,uncertainty_quantile=args.uncertainty_quantile)

if __name__=="__main__":
    main()
