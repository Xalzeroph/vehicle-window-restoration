#!/usr/bin/env python3

import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from train_vehicle_optical import VehiclePairs,build_rdnet,masked_mean


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--validation",required=True,type=Path)
    p.add_argument("--old",required=True,type=Path)
    p.add_argument("--fine",required=True,type=Path)
    p.add_argument("--classifier",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--per-capture-group",type=int,default=2)
    args=p.parse_args()
    torch.set_num_threads(4)
    ds=VehiclePairs(args.validation,crop=128,augment=False,seed=19)
    groups=defaultdict(list)
    for i,r in enumerate(ds.samples):groups[r["capture_group"]].append(i)
    selected=[]
    for ix in range(args.per_capture_group):
        for group in sorted(groups):
            if ix<len(groups[group]):selected.append(groups[group][ix])
    net,cls=build_rdnet(args.old,args.classifier,torch.device("cpu"))
    net.eval();cls.eval()
    weights=torch.load(args.fine,map_location="cpu",weights_only=False)
    with torch.inference_mode():
        base=[]
        for j in selected:
            x,t,r,v=ds[j]
            ctx=cls(x[None])
            _,outs=net(x[None],ctx,prompt=True)
            pred=outs[-1][0,3:].float().clamp(0,1).cpu()
            base.append(pred)
        net.load_state_dict(weights.get("icnn",weights.get("model",weights)),strict=True)
        fine=[]
        for j in selected:
            x,t,r,v=ds[j]
            ctx=cls(x[None])
            _,outs=net(x[None],ctx,prompt=True)
            pred=outs[-1][0,3:].float().clamp(0,1).cpu()
            fine.append(pred)
    rows=[]
    canvas=Image.new("RGB",(5*268,len(selected)*163),"white")
    draw=ImageDraw.Draw(canvas)
    to_uint=lambda arr:Image.fromarray(np.uint8(np.round(arr.numpy().transpose(1,2,0).clip(0,1)*255)))
    for row,(idx,old,new) in enumerate(zip(selected,base,fine)):
        observed,tint,_,valid=ds[idx]
        mask=valid[None]
        psnr=lambda pred:float(-10*np.log10(max(float(masked_mean(
            (pred[None]-tint[None])**2,mask)),1e-12)))
        before=psnr(old);after=psnr(new);inp=psnr(observed)
        rows.append({"sample":ds.samples[idx]["id"],
                     "capture_group":ds.samples[idx]["capture_group"],
                     "input_psnr":inp,
                     "rdnet_before":before,"rdnet_after":after,
                     "delta_after_before":after-before})
        images=[to_uint(p) for p in (observed,tint,old,new)]
        a=np.asarray(images[2],dtype=np.int16)
        b=np.asarray(images[3],dtype=np.int16)
        difference=Image.fromarray(np.uint8(np.clip(abs(a-b)*4,0,255)))
        images.append(difference)
        for col,(label,img) in enumerate(zip(
              ("observed","synthetic GT","old RDNet","new RDNet","change x4"),images)):
            img=img.resize((256,138),Image.Resampling.NEAREST)
            x=268*col;y=163*row
            canvas.paste(img,(x+3,y+21))
            draw.text((x+3,y+2),f"{label} {before:.1f}->{after:.1f}" if col==3 else label,
                      fill="black")
    args.out.mkdir(parents=True,exist_ok=False)
    canvas.save(args.out/"old_vs_retrained_synthetic_visual.jpg",quality=93)
    group_summary={}
    for group in sorted(groups):
        values=[r for r in rows if r["capture_group"]==group]
        if values:
            group_summary[group]={
                "n":len(values),
                "before_psnr_db":float(np.mean([v["rdnet_before"] for v in values])),
                "after_psnr_db":float(np.mean([v["rdnet_after"] for v in values])),
                "input_psnr_db":float(np.mean([v["input_psnr"] for v in values]))}
    (args.out/"per_capture_synthetic_results.json").write_text(json.dumps(
        {"real_windshield_gt":False,"n_unique_groups":len(group_summary),
         "n_synthetic_patches":len(selected),"rows":rows,
         "group_macro_before_psnr_db":float(np.mean([x["before_psnr_db"] for x in group_summary.values()])),
         "group_macro_after_psnr_db":float(np.mean([x["after_psnr_db"] for x in group_summary.values()])),
         "group_macro_input_psnr_db":float(np.mean([x["input_psnr_db"] for x in group_summary.values()])),
         "group_summary":group_summary},indent=2)+"\n")
    print("HUMAN_VISUAL_REVIEW_PREPARED",str(args.out),"SAMPLES",len(rows),flush=True)
    for row in rows:print("PAIR_RESULT",row,flush=True)


if __name__=="__main__":main()
