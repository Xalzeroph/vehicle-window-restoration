#!/usr/bin/env python3


import argparse,hashlib,json,re,time,random
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import quote
import requests
import xml.etree.ElementTree as ET
from PIL import Image

BASE="https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
NS="{http://s3.amazonaws.com/doc/2006-03-01/}"

def list_files(prefix, max_pages=500):
    marker=None
    rows=[]
    sess=requests.Session()
    for page in range(max_pages):
        params={"list-type":"2","prefix":prefix,"max-keys":1000}
        if marker:params["continuation-token"]=marker
        for attempt in range(6):
            try:
                r=sess.get(BASE,params=params,timeout=45)
                r.raise_for_status()
                xml=ET.fromstring(r.content)
                break
            except Exception:
                if attempt==5:raise
                time.sleep(2**attempt)
        for obj in xml.findall(NS+"Contents"):
            key=obj.findtext(NS+"Key")
            if key.endswith(".png"):
                rows.append({"s3_key":key,"byte_size":int(obj.findtext(NS+"Size")),
                             "etag":obj.findtext(NS+"ETag").strip('"')})
        if xml.findtext(NS+"IsTruncated")!="true":break
        marker=xml.findtext(NS+"NextContinuationToken")
        if not marker:raise RuntimeError("Truncated but missing S3 continuation token")
    else:
        raise RuntimeError("Inventory paged limit reached")
    return rows

def get_file(obj,root):
    dest=root/obj["s3_key"]
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_file() and dest.stat().st_size==obj["byte_size"]:
        return obj["s3_key"],"EXISTS",hashlib.sha256(dest.read_bytes()).hexdigest()
    tmp=dest.with_suffix(".part")
    for attempt in range(6):
        try:
            with requests.get(BASE+quote(obj["s3_key"]),timeout=90,stream=True) as response:
                response.raise_for_status()
                with tmp.open("wb") as f:
                    for part in response.iter_content(chunk_size=1<<20):
                        if part:f.write(part)
            if tmp.stat().st_size != obj["byte_size"]:
                raise IOError("Unexpected S3 file size")
            with Image.open(tmp) as i:
                if i.format!="PNG" or min(i.size)<100:raise ValueError("Invalid input PNG")
                i.verify()
            tmp.replace(dest)
            return obj["s3_key"],"DOWNLOADED",hashlib.sha256(dest.read_bytes()).hexdigest()
        except Exception:
            if attempt==5:raise
            time.sleep(min(2**attempt,20))
    raise RuntimeError("No retries remaining")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--camera",default="cam_front_center")
    p.add_argument("--per-session",type=int,default=200)
    p.add_argument("--workers",type=int,default=6)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    corpus=[]
    sessionids=["20180810_150607","20190401_121727","20190401_145936"]
    for session in sessionids:
        prefix=f"camera_lidar/{session}/camera/{a.camera}/"
        items=list_files(prefix)
        if not items:raise RuntimeError("No original PNG for "+prefix)
        items.sort(key=lambda x:x["s3_key"])
        selected=[items[(len(items)-1)*i//(min(a.per_session,len(items))-1)]
                  for i in range(min(a.per_session,len(items)))] if len(items)>1 and a.per_session>1 else items[:1]
        for item in selected:
            item["capture_group"]="A2D2/"+session+"/"+a.camera
            item["dataset"]="Audi A2D2"
            item["license"]="CC BY-ND 4.0, research local derivatives, do NOT distribute derivatives"
        corpus+=selected
        print("INVENTORIED",session,"original",len(items),"selected",len(selected),flush=True)

    man=a.out/"a2d2_source_manifest.json"
    man.write_text(json.dumps({"dataset":"Audi A2D2","venue":"arXiv 2020 published",
            "research_local_only":True,"source_objects":corpus},indent=2)+"\n")
    print("INDEX",len(corpus),"possible_bytes",sum(x["byte_size"] for x in corpus),flush=True)
    successes={}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        f={pool.submit(get_file,obj,a.out):obj for obj in corpus}
        for future in as_completed(f):
            x=future
            try:
                key,status,sha=future.result()
                successes[key]=sha
                if len(successes)%20==0:print("DOWNLOADED",len(successes),"/",len(corpus),flush=True)
            except Exception as e:print("FAILURE",f[x]["s3_key"],repr(e)[:200],flush=True)
    for obj in corpus:
        if obj["s3_key"] in successes:
            obj["local_path"]=str(a.out/obj["s3_key"])
            obj["sha256"]=successes[obj["s3_key"]]
    done=[x for x in corpus if "local_path" in x]
    (a.out/"a2d2_acquired_manifest.json").write_text(
        json.dumps({"dataset":"A2D2","status":"LICENSE-RESTRICTED INTERNAL RESEARCH SOURCES",
                    "items":done,"count":len(done)},indent=2)+"\n")
    print("FINAL",len(done),"/",len(corpus),flush=True)

if __name__=="__main__":
    main()
