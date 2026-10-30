#!/usr/bin/env python3
"""Retrieve published real I/T/mask samples for human-reviewed semantic annotations.

Masks from upstream are only weak Ω proposals, NEVER semantic ground truth.
Retain exact scene identities and immutable local PNG sources.
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path
import requests

DATASET = "zmy1234567890/tintedglass-rdt-synth"
BASE = f"https://hf-mirror.com/datasets/{DATASET}/resolve/main/"
API = f"https://hf-mirror.com/api/datasets/{DATASET}"
GROUPS = ("rdt_1019split", "rdt_1026", "rdt_1020")

def download(item):
    relative, root = item
    path = root / relative
    if path.is_file() and path.stat().st_size > 1000:
        return True, relative
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(9):
        try:
            with requests.get(BASE+relative, stream=True, timeout=(12,65)) as result:
                if result.status_code == 429:
                    time.sleep(min(80,10*(attempt+1)))
                    continue
                result.raise_for_status()
                with partial.open("wb") as out:
                    for chunk in result.iter_content(1<<18):
                        if chunk:
                            out.write(chunk)
            partial.replace(path)
            return True, relative
        except Exception:
            partial.unlink(missing_ok=True)
            time.sleep(min(30,3*(attempt+1)))
    return False,relative

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--all-pairs",action="store_true")
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    data=requests.get(API,timeout=45).json()
    published={x["rfilename"] for x in data["siblings"]}
    rows=[]
    for group in GROUPS:
        all_names=sorted(x.rsplit("/",1)[1].removesuffix(".png") for x in published
            if x.startswith(f"train/{group}/I/") and x.endswith(".png"))
        by_scene={}
        for name in all_names:
            scene=re.search(r'VID_\d{8}_\d{6}',name)
            if scene:
                by_scene.setdefault(scene.group(),[]).append(name)
        for scene, names in by_scene.items():
            selected=names if args.all_pairs else [names[i] for i in sorted({0,len(names)//4,len(names)//2,3*len(names)//4,len(names)-1})]
            for name in selected:
                prefix=f"train/{group}"
                files={k:f"{prefix}/{k}/{name}{'.npy' if k=='masks' else '.png'}"
                       for k in ("I","T","masks")}
                if not all(x in published for x in files.values()):
                    continue
                rows.append(dict(id=f"{group}/{name}",group=group,scene=scene,
                        files=files,mask_kind="upstream_omega_weak_proposal",
                        label_status="unreviewed",release_revision=data["sha"]))
    jobs=[(rel,args.out) for row in rows for rel in row["files"].values()]
    print("OCT_ANNOTATION_SOURCE",len(rows),"samples",len(jobs),"files",flush=True)
    failure=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i,(ok,name) in enumerate(ex.map(download,jobs),1):
            if not ok:failure.append(name)
            if i%30==0 or i==len(jobs):
                print("OCT_SOURCE_DOWNLOAD",i,len(jobs),"fail",len(failure),flush=True)
    rows=[row for row in rows if all((args.out/x).is_file() for x in row["files"].values())]
    manifest={"dataset":DATASET,"revision":data["sha"],"label_status":"unreviewed",
              "n":len(rows),"files_failed":failure,"items":rows}
    (args.out/"source_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print("OCT_SOURCE_DONE",len(rows),"files_failed",len(failure),flush=True)

if __name__=="__main__":
    main()
