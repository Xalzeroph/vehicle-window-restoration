#!/usr/bin/env python3


import argparse
import json
from pathlib import Path
from collections import Counter
from vehicle_recipe_index import _normalized_list


def curate(source_manifest, review, destination):
    source=_normalized_list(source_manifest)
    decisions=json.loads(Path(review).read_text())
    decisions=decisions["capture_groups"]
    groups={x["capture_group"] for x in source}
    missing=groups-set(decisions)
    extra=set(decisions)-groups
    if missing or extra:
        raise ValueError("All capture groups must be accounted for: missing="+str(missing)+" extra="+str(extra))
    valid_labels={"driving_eligible","exclude_static_calibration","exclude_other"}
    for group,entry in decisions.items():
        if entry.get("disposition") not in valid_labels or not entry.get("evidence"):
            raise ValueError("Every group needs explicit disposition and evidence "+group)
    approved=[row for row in source
              if decisions[row["capture_group"]]["disposition"]=="driving_eligible"]
    excluded=[row for row in source
              if decisions[row["capture_group"]]["disposition"]!="driving_eligible"]
    if not approved:
        raise ValueError("No approved sources")
    report={
        "status":"ANNOTATED DRIVING CANDIDATES ONLY; STILL NOT HDR/CAMERA-GLASS CALIBRATED",
        "source":str(source_manifest),
        "review_provenance":str(review),
        "original_images":len(source),
        "approved_images":len(approved),
        "excluded_images":len(excluded),
        "approved_capture_groups":sorted({row["capture_group"] for row in approved}),
        "excluded_capture_groups":sorted({row["capture_group"] for row in excluded}),
        "items":approved,
    }
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    Path(destination).write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--review",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print("CURATED",curate(a.source,a.review,a.out)["approved_images"])

if __name__=="__main__":main()
