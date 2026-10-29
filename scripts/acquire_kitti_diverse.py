#!/usr/bin/env python3


from collections import defaultdict
import argparse,hashlib,json,io,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests,xml.etree.ElementTree as ET
from remotezip import RemoteZip
from PIL import Image

HOST="https://s3.eu-central-1.amazonaws.com/avg-kitti/"
NS="{http://s3.amazonaws.com/doc/2006-03-01/}"

def inventory():
    token=None
    results=[]
    sess=requests.Session()
    while True:
        params={"list-type":"2","prefix":"raw_data/","max-keys":1000}
        if token: params["continuation-token"]=token
        response=sess.get(HOST,params=params,timeout=45)
        response.raise_for_status()
        root=ET.fromstring(response.content)
        for obj in root.findall(NS+"Contents"):
            key=obj.findtext(NS+"Key")
            if key.endswith("_sync.zip"):
                results.append((key,int(obj.findtext(NS+"Size"))))
        if root.findtext(NS+"IsTruncated")!="true":break
        token=root.findtext(NS+"NextContinuationToken")
        if token is None:raise RuntimeError("S3 truncated without token")
    return results

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for blob in iter(lambda:f.read(1<<20),b""):h.update(blob)
    return h.hexdigest()

def fetch_case(out,case,n_frames):
    key,size=case
    url=HOST+key
    session=Path(key).parent.name
    folder=out/session
    folder.mkdir(parents=True,exist_ok=True)
    results=[]
    with RemoteZip(url) as zip:
        frames=sorted([s for s in zip.namelist() if
                       "/image_02/data/" in s and s.endswith(".png")])
        if not frames:raise RuntimeError("No KITTI camera-02 frames "+url)
        select=[frames[(len(frames)-1)*i//(min(len(frames),n_frames)-1)] for i in range(min(len(frames),n_frames))] if n_frames>1 and len(frames)>1 else frames[:1]
        for member in select:
            dst=folder/Path(member).name
            if not dst.is_file():
                for attempt in range(5):
                    try:
                        payload=zip.read(member)
                        with Image.open(io.BytesIO(payload)) as im:
                            if im.format!="PNG" or min(im.size)<100:
                                raise ValueError("Invalid original KITTI PNG")
                            im.verify()
                        dst.write_bytes(payload)
                        break
                    except Exception as e:
                        if attempt==4:raise
                        time.sleep(1+2**attempt)
            results.append({"dataset":"KITTI","conference":"CVPR2012","capture_group":session,
                            "filename":str(dst),"member":member,
                            "sha256":sha(dst),"original_zip":url,
                            "original_archive_size_bytes":size,
                            "license_note":"KITTI original official source terms, research / attribution"})
    return session,results

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",required=True,type=Path)
    a.add_argument("--traversals-per-day",type=int,default=5)
    a.add_argument("--frames-per-traversal",type=int,default=4)
    a.add_argument("--workers",type=int,default=4)
    args=a.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    by_day=defaultdict(list)
    allcases=inventory()
    for k,sz in allcases:
        by_day[Path(k).parent.name[:10]].append((k,sz))


    cases=[]
    for day in sorted(by_day):
        group=sorted(by_day[day],key=lambda x:x[1])
        cases+=group[:args.traversals_per_day]
    print("KITTI_ARCHIVE_INVENTORY",len(allcases),
          "DATES",{k:len(v) for k,v in by_day.items()},flush=True)
    print("SELECTED_TRAVERSALS",len(cases),"sample",[(Path(k).parent.name,sz) for k,sz in cases[:7]],flush=True)
    allrecords=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(fetch_case,args.out,k,args.frames_per_traversal):k for k in cases}
        for job in as_completed(jobs):
            try:
                key,records=job.result()
                allrecords+=records
                print("FETCHED_KITTI_DRIVE",key,"frames",len(records),flush=True)
            except Exception as e:
                print("FAILURE_KITTI",jobs[job][0],repr(e),flush=True)
    out=args.out/"kitti_diverse_manifest.json"
    out.write_text(json.dumps({
        "source":"KITTI original research dataset",
        "selected_policy":"smallest original ZIPs for each capture date, one independent group per traversal",
        "traversals":len(set(x["capture_group"] for x in allrecords)),
        "items":sorted(allrecords,key=lambda r:(r["capture_group"],r["filename"]))},indent=2)+"\n")
    print("FINAL",len(allrecords),len(set(x["capture_group"] for x in allrecords)),"groups",flush=True)

if __name__=="__main__":
    main()
