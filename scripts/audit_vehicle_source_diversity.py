#!/usr/bin/env python3


import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from vehicle_recipe_index import import_source


def dhash(arr):


    img=Image.fromarray(np.uint8(np.clip(arr*255,0,255)))
    gray=ImageOps.grayscale(img).resize((9,8),Image.Resampling.LANCZOS)
    x=np.asarray(gray,dtype=np.int16)
    bits=x[:,1:] > x[:,:-1]
    n=0
    for b in bits.reshape(-1):
        n=(n<<1)|int(b)
    return n


def profile(path):
    with Image.open(path) as source:
        if source.format not in ("PNG","JPEG"):
            raise ValueError("Unexpected image format")
        size=source.size
        im=source.convert("RGB")
        im.thumbnail((256,256),Image.Resampling.LANCZOS)
    x=np.asarray(im,dtype=np.float64)/255
    L=x@np.array([.2126,.7152,.0722])

    gx=np.diff(L,axis=1)
    gy=np.diff(L,axis=0)
    return {
        "width":size[0],"height":size[1],
        "bit_depth_source":8,
        "jpeg_or_png":source.format,
        "mean_display_luminance":float(L.mean()),
        "p05_display_luminance":float(np.quantile(L,.05)),
        "p95_display_luminance":float(np.quantile(L,.95)),
        "fraction_near_black":float((L<=.02).mean()),
        "fraction_near_white":float((L>=.98).mean()),
        "edge_gradient_energy":float(.5*((gx*gx).mean()+(gy*gy).mean())),
        "dhash_hex":f"{dhash(x):016x}",
    }


def summary(data):
    if not data:return None
    x=np.asarray(data,dtype=np.float64)
    return {"p10":float(np.quantile(x,.1)),"p50":float(np.median(x)),
            "p90":float(np.quantile(x,.9)),"min":float(np.min(x)),
            "max":float(np.max(x))}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--a2d2",required=True,type=Path)
    a.add_argument("--kitti",required=True,type=Path)
    a.add_argument("--cabin",required=True,type=Path)
    a.add_argument("--out",required=True,type=Path)
    args=a.parse_args()
    source=import_source(args.a2d2,"a2d2")+import_source(args.kitti,"kitti")
    source+=import_source(args.cabin,"driveact")
    allrows=[]
    for i,s in enumerate(source,1):
        actual=hashlib.sha256(Path(s.path).read_bytes()).hexdigest()
        if s.sha256 != actual:
            raise ValueError("Source bytes changed since download "+s.path)
        entry={"dataset":s.dataset,"capture_group":s.capture_group,
               "sha256":actual,"path":s.path,**profile(s.path)}
        allrows.append(entry)
        if i%100==0:print("PROFILED",i,"/",len(source),flush=True)
    group=defaultdict(list)
    for row in allrows:group[row["dataset"]].append(row)
    stats={}
    for name,items in group.items():
        stats[name]={
            "num_original_images":len(items),
            "num_original_capture_groups":len({x["capture_group"] for x in items}),
            "native_resolution_counts":dict(Counter(f'{x["width"]}x{x["height"]}' for x in items)),
            "mean_display_luminance":summary([x["mean_display_luminance"] for x in items]),
            "near_black_fraction":summary([x["fraction_near_black"] for x in items]),
            "near_white_fraction":summary([x["fraction_near_white"] for x in items]),
            "edge_energy":summary([x["edge_gradient_energy"] for x in items]),
            "dhash_unique_count":len({x["dhash_hex"] for x in items})
        }


    groups_hash={}
    close=[]
    for name,items in group.items():
        by_hash=[(int(x["dhash_hex"],16),x) for x in items]
        for i,(hx,x) in enumerate(by_hash):
            for hy,y in by_hash[i+1:]:
                hd=(hx^hy).bit_count()
                if hd<=3:
                    close.append({
                        "dataset":name,"sha_a":x["sha256"],"sha_b":y["sha256"],
                        "same_capture_group":x["capture_group"]==y["capture_group"],
                        "group_a":x["capture_group"],"group_b":y["capture_group"],
                        "path_a":x["path"],"path_b":y["path"],
                        "hamming_distance":hd})
    report={
        "source_frame_count":len(allrows),
        "unique_capture_groups_total":len({x["capture_group"] for x in allrows}),
        "exact_pixel_file_dedup":len({x["sha256"] for x in allrows}),
        "stats_by_dataset":stats,
        "visual_dhash_near_pairs_hamming_le3":len(close),
        "near_pairs_cross_capture_group":sum(not x["same_capture_group"] for x in close),
        "near_pairs_preview":close[:100],
        "cross_capture_near_pairs":[x for x in close if not x["same_capture_group"]],
        "warnings":[
            "RGB 8-bit camera JPEG/PNG images are NOT RAW sensor radiance",
            "Display luminance is not calibrated exposure and cannot prove weather/time of day",
            "Difference hash similarities are diagnostic only, not semantic video-scene labels",
            "Independent video capture groups are not guaranteed independent routes/vehicles",
            "A2D2/Drive&Act derivative redistribution must respect original source licenses",
        ]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print("SOURCE_QA",json.dumps({k:v for k,v in report.items()
         if k!="near_pairs_preview"},indent=2),flush=True)


if __name__=="__main__":main()
