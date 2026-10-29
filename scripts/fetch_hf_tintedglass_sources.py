#!/usr/bin/env python3


import argparse,concurrent.futures,hashlib,json,os,time
from collections import Counter
from pathlib import Path
import requests
import numpy as np
from PIL import Image

CATEGORIES={"matched","nature","sir2"}
REPO="zmy1234567890/tintedglass-rdt-synth"
BASE="https://hf-mirror.com/datasets/"+REPO+"/resolve/main/train/rdt_0915"

def split(key):
    h=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big")/2**64
    return "train" if h<0.80 else ("validation" if h<0.90 else "test")

def work(job):
    base,dirpart,sid=job
    name=sid+(".png" if dirpart in ("T","R") else ".npy")
    target=base/dirpart/name
    if target.is_file():
        try:
            if dirpart in ("T","R"):
                with Image.open(target) as im:im.verify()
            else:
                x=np.load(target,mmap_mode="r")
                if x.shape!=(448,448):raise ValueError("Wrong mask shape")
            return True
        except Exception:
            target.unlink(missing_ok=True)
    target.parent.mkdir(exist_ok=True,parents=True)
    url=f"{BASE}/{dirpart}/{name}"
    for attempt in range(3):
        try:
            response=requests.get(url,stream=True,timeout=(9,45),headers={"User-Agent":"VehicleResearchAudit/1.0"})
            response.raise_for_status()
            tmp=target.with_suffix(target.suffix+".part")
            with open(tmp,"wb") as f:
                for chunk in response.iter_content(1<<18):
                    if chunk:f.write(chunk)
            if dirpart in ("T","R"):
                with Image.open(tmp) as im:im.verify()
            else:
                x=np.load(tmp,mmap_mode="r")
                if x.shape!=(448,448):raise ValueError("Wrong mask shape")
            os.replace(tmp,target)
            return True
        except Exception as exc:
            target.with_suffix(target.suffix+".part").unlink(missing_ok=True)
            if attempt==2:return str(exc)
            time.sleep(.4*(attempt+1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--train",type=int,default=320)
    ap.add_argument("--validation",type=int,default=32)
    ap.add_argument("--test",type=int,default=32)
    ap.add_argument("--workers",type=int,default=20)
    args=ap.parse_args()
    data=json.loads((args.root/"meta.json").read_text())
    scenes=data["scenes"]
    partitions={n:[] for n in ("train","validation","test")}
    for row in scenes:
        category=row["t_src"].split(":")[0]
        if category not in CATEGORIES:continue
        ta=split("T:"+row["t_src"])
        ra=split("R:"+row["r_src"])
        if ta!=ra:continue
        partitions[ta].append(dict(row,t_category=category))
    rng=np.random.default_rng(20261029)
    selected={}
    for fold,target in (("train",args.train),("validation",args.validation),("test",args.test)):
        options=partitions[fold]
        rng.shuffle(options)


        groups={k:[] for k in CATEGORIES}
        for x in options:groups[x["t_category"]].append(x)
        ids=sorted(groups)
        selected[fold]=[]
        while len(selected[fold])<target:
            changed=False
            for cat in ids:
                if groups[cat]:
                    selected[fold].append(groups[cat].pop())
                    changed=True
                    if len(selected[fold])==target:break
            if not changed:break
        print("HF_FOLD_PLAN",fold,"requested",target,"available",len(options),"chosen",len(selected[fold]),
              "source_types",dict(Counter(x["t_category"] for x in selected[fold])),flush=True)
    unique=[(args.root,key,scene["scene_id"]) for fold in selected for scene in selected[fold] for key in ("T","R","m","g")]
    start=time.monotonic()
    done=0;fail=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job,result in zip(unique,pool.map(work,unique)):
            done+=1
            if result is not True:fail.append((job[1:],result))
            if done%96==0:
                print("HF_SOURCE_TRANSFER",done,"/",len(unique),"errors",len(fail),"seconds",round(time.monotonic()-start,1),flush=True)
    if fail:print("HF_ERRORS",fail[:10],flush=True)
    recovered={}
    for fold,rows in selected.items():
        recovered[fold]=[x for x in rows if all((args.root/d/(x["scene_id"]+(".png" if d in ("T","R") else ".npy"))).is_file() for d in ("T","R","m","g"))]
    index={"source_repo":REPO,"metadata_original":str(args.root/"meta.json"),"status":"RESEARCH_ONLY_SYNTHETIC_GLASS_NOT_SPECIFIC_TO_CAR_CABIN","source_t_categories":sorted(CATEGORIES),
           "download_errors":fail,"historical_test_reserved":True,"folds":recovered}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(index,indent=2)+"\n")
    print("HF_TRANSFER_COMPLETE",{"elapsed":round(time.monotonic()-start,1),
        "scene_counts":{k:len(v) for k,v in recovered.items()},"missing":len(fail),
        "index":str(args.out)},flush=True)
if __name__=="__main__":main()
