#!/usr/bin/env python3
"""Conservative semantic pseudo-annotation of the published October I/T samples.

The original Omega mask is an uncalibrated proposal. Dense class labels are
assigned only where category evidence and paired difference agree. Every
ambiguous pixel is 255 (ignore); NOT a human-verified ground-truth mask.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,binary_opening,binary_closing
import torch
import torch.nn.functional as F

LABELS={"background":0,"human_nuisance":1,"vegetation_nuisance":2,
        "other_weak_nuisance":3,"ignore":255}
PERSON=(12,)
VEG=(4,17,72)
OUT_SIZE=448

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--root",type=Path,required=True)
 p.add_argument("--model",type=Path,required=True)
 p.add_argument("--device",default="cuda:0")
 args=p.parse_args()
 sys.path.append('/data/hdd3/agent-research-runtime/vehicle-assets/lib/semantic')
 from transformers import SegformerForSemanticSegmentation,SegformerImageProcessor
 model=SegformerForSemanticSegmentation.from_pretrained(str(args.model),use_safetensors=True).to(args.device).eval()
 processor=SegformerImageProcessor.from_pretrained(str(args.model))
 records=json.loads((args.root/"source_manifest.json").read_text())["items"]
 out=args.root/"semantic_labels"
 out.mkdir(parents=True,exist_ok=True)
 annotated=[]
 for index,row in enumerate(records):
  imgs=[]
  for key in ("I","T"):
   with Image.open(args.root/row["files"][key]) as im:
    img=im.convert("RGB").resize((OUT_SIZE,OUT_SIZE),Image.Resampling.BILINEAR)
   imgs.append(img)
  array_i=np.asarray(imgs[0],dtype=np.float32)/255.
  array_t=np.asarray(imgs[1],dtype=np.float32)/255.
  with torch.inference_mode():
   batch=processor(images=imgs,return_tensors="pt").pixel_values.to(args.device)
   logits=model(batch).logits
   probabilities=F.interpolate(logits,size=(OUT_SIZE,OUT_SIZE),
     mode="bilinear",align_corners=False).softmax(1).cpu().numpy()
  person_i=probabilities[0,PERSON,:,:].sum(axis=0)
  person_t=probabilities[1,PERSON,:,:].sum(axis=0)
  veg_i=probabilities[0,VEG,:,:].sum(axis=0)
  veg_t=probabilities[1,VEG,:,:].sum(axis=0)
  omega=np.load(args.root/row["files"]["masks"],allow_pickle=False)
  if omega.shape!=(OUT_SIZE,OUT_SIZE) or omega.dtype!=np.uint8:
   raise ValueError("Unexpected upstream Omega mask resolution/dtype")
  difference=gaussian_filter(np.abs(array_i-array_t).mean(axis=2),sigma=1.)
  signed=gaussian_filter((array_t-array_i).mean(axis=2),sigma=1.)
  subject=(difference>.055)
  # Corrected reflection/occlusion nuisance, not all trees or real scene people.
  human=(person_i>.34)&(person_i-person_t>.12)&subject
  vegetation=(veg_i>.38)&(veg_i-veg_t>.14)&subject
  human=binary_closing(binary_opening(human,iterations=1),iterations=1)
  vegetation=binary_closing(binary_opening(vegetation,iterations=1),iterations=1)&~human
  # Upstream Omega is weak; use it only for other/ambiguous interference.
  other=(omega>0)&(signed>.075)&(difference>.11)&~human&~vegetation
  stable=(omega==0)&(difference<.035)
  labels=np.full((OUT_SIZE,OUT_SIZE),255,dtype=np.uint8)
  labels[stable]=0
  labels[other]=3
  labels[vegetation]=2
  labels[human]=1
  relative=Path(row["id"]+".png")
  path=out/relative
  path.parent.mkdir(parents=True,exist_ok=True)
  Image.fromarray(labels).save(path)
  counters={k:int(np.sum(labels==v)) for k,v in LABELS.items()}
  annotated.append(dict(id=row["id"],scene=row["scene"],group=row["group"],
    label_path=str(path.relative_to(args.root)),status="model_pseudo_unreviewed",
    counts=counters,
    label_recipe="ADE20K SegFormer classes + paired signed difference + conservative Omega proposal"))
  if (index+1)%10==0:
   total={k:sum(a["counts"][k] for a in annotated) for k in LABELS}
   print("ANNOTATED",index+1,len(records),total,flush=True)
  del logits,probabilities
 result={"source_model":"nvidia/segformer-b0-finetuned-ade-512-512",
   "label_kind":"model_pseudo_unreviewed",
   "caution":"not manually pixel-annotated segmentation ground truth",
   "label_schema":LABELS,
   "annotations":annotated}
 (args.root/"semantic_annotations.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
 print("ANNOTATION_COMPLETE",len(annotated),flush=True)

if __name__=="__main__":main()
