#!/usr/bin/env python3


from pathlib import Path
import json,collections
import numpy as np
from PIL import Image

A=Path("/data/hdd3/agent-research-runtime/vehicle-assets")
root=A/"runs/tintedglass_hf_rebuild_20261029"
rows=json.loads((root/"train/photo_manifest.json").read_text())["items"]
src=A/"datasets/tintedglass-rdt-synth/train/rdt_0915"
seen=set();recs=[]
for row in rows:
 key=row["id"].split("_v")[0]
 if key in seen:continue
 seen.add(key)
 T=np.asarray(Image.open(src/"T"/f"{key}.png").convert("RGB").resize((64,64)),dtype=np.float64)/255
 R=np.asarray(Image.open(src/"R"/f"{key}.png").convert("RGB").resize((64,64)),dtype=np.float64)/255
 tm=T.reshape(-1,3).mean(axis=0)
 rm=R.reshape(-1,3).mean(axis=0)
 denom=tm.mean()
 recs.append({"group":row["source_category"],"tmean":tm,"rmean":rm,
              "T_luma":float(tm@[.2126,.7152,.0722]),
              "R_luma":float(rm@[.2126,.7152,.0722]),
              "T_chroma":tm/(tm.sum()+1e-8),"R_chroma":rm/(rm.sum()+1e-8),
              "T_camera_cast":float(tm.max()/max(tm.min(),1e-6)),
              "R_camera_cast":float(rm.max()/max(rm.min(),1e-6))})
result={"n_independent_scene_recipes":len(recs),"per_category":{}}
for group in sorted(set(r["group"] for r in recs)):
 rr=[r for r in recs if r["group"]==group]
 result["per_category"][group]={
  "n":len(rr),
  "T_luma_p10_p50_p90":[float(np.quantile([z["T_luma"] for z in rr],x)) for x in (.1,.5,.9)],
  "R_luma_p10_p50_p90":[float(np.quantile([z["R_luma"] for z in rr],x)) for x in (.1,.5,.9)],
  "T_cast_p50_p90":[float(np.quantile([z["T_camera_cast"] for z in rr],x)) for x in (.5,.9)],
  "R_cast_p50_p90":[float(np.quantile([z["R_camera_cast"] for z in rr],x)) for x in (.5,.9)],
  "mean_T_chroma":[float(x) for x in np.mean([z["T_chroma"] for z in rr],axis=0)]}
out=root/"source_color_diagnostics_20261029.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print("HF_SOURCE_COLOR_DIAG",json.dumps(result),flush=True)
