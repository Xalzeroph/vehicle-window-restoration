#!/usr/bin/env python3


import argparse,json
from pathlib import Path


def select(rendered_manifest,group_review,out):
    data=json.loads(Path(rendered_manifest).read_text())
    review=json.loads(Path(group_review).read_text())["capture_groups"]
    if data["status"]!="RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE":
        raise ValueError("Unrecognized base manifest")
    kept=[]
    excluded=[]
    for row in data["items"]:
        group=row["capture_group"]
        if group.startswith("KITTI/"):
            name=group.removeprefix("KITTI/")
            decision=review[name]["disposition"]
            if decision!="driving_eligible":
                excluded.append({"id":row["id"],"group":group,
                                 "reason":review[name]["evidence"]})
                continue
        kept.append(row)
    if not kept or not excluded:
        raise ValueError("Expected nonempty candidates and auditable exclusions")
    clean=dict(data)
    clean["items"]=kept
    clean["original_image_count"]=len(data["items"])
    clean["source_screened_image_count"]=len(kept)
    clean["excluded_image_count"]=len(excluded)
    clean["exclude_manifest"]=excluded
    clean["status"]="RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE"
    clean["human_review_warning"]="Still no measured car-glass optics or geometric registration"
    Path(out).write_text(json.dumps(clean,indent=2)+"\n")
    return clean


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rendered-manifest",required=True,type=Path)
    p.add_argument("--review",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    m=select(a.rendered_manifest,a.review,a.out)
    print("SCREENED",m["source_screened_image_count"],
          "REJECTED",m["excluded_image_count"],"OUT",a.out,flush=True)


if __name__=="__main__":main()
