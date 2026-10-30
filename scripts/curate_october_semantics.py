#!/usr/bin/env python3
"""Conservative audit correction: no unverified 'person' pixels as truth.

Carry only hand-reviewed nuisance ROIs with data-driven boundaries. These
remain weak annotations, not independently validated pixel-perfect masks.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,sobel,binary_opening,binary_closing

ROOT=Path('/data/hdd3/agent-research-runtime/vehicle-assets/annotations/oct2026_occlusion_v1')
# The reflected branches are visually identifiable in these I/GT comparisons.
# Format (relative id prefix, x0,y0,x1,y1), normalized image frame.
TREE_ROIS={
 "rdt_1019split/1019split__VID_20261019_165400__frame_000000": [(0.,0.,1.,.47)],
 "rdt_1019split/1019split__VID_20261019_165400__frame_000975": [(0.,0.,1.,.53)],
 "rdt_1019split/1019split__VID_20261019_171126__frame_001982": [(0.,0.,1.,.53)],
}
def main():
 src=json.loads((ROOT/"semantic_annotations.json").read_text())
 catalog={r["id"]:r for r in json.loads((ROOT/"source_manifest.json").read_text())["items"]}
 reviewed=[]
 for row in src["annotations"]:
  path=ROOT/row["label_path"]
  labels=np.asarray(Image.open(path),dtype=np.uint8).copy()
  # A semantic model mistook dark stationary objects for people. Do not
  # provide false hard person targets to the learnable head.
  labels[labels==1]=255
  if row["id"] in TREE_ROIS:
   pair=catalog[row["id"]]
   ims=[]
   for name in ("I","T"):
    with Image.open(ROOT/pair["files"][name]) as image:
     ims.append(np.asarray(image.convert("RGB").resize((448,448),
                        Image.Resampling.BILINEAR),dtype=np.float32)/255)
   a,b=ims
   lum=lambda c:c@[.2126,.7152,.0722]
   la,lb=lum(a),lum(b)
   signed=gaussian_filter(lb-la,.8)
   edge_delta=np.abs(sobel(la,axis=0)-sobel(lb,axis=0))+np.abs(
                        sobel(la,axis=1)-sobel(lb,axis=1))
   region=np.zeros((448,448),dtype=bool)
   for x0,y0,x1,y1 in TREE_ROIS[row["id"]]:
    region[int(y0*448):int(y1*448),int(x0*448):int(x1*448)]=True
   # A tree-shaped candidate must have strong negative silhouette structure
   # in I relative to T, and pairwise high-frequency difference.
   positive=region & (signed>.055) & (edge_delta>.025)
   positive=binary_closing(positive,iterations=1)
   labels[positive]=2
   reviewed.append(dict(id=row["id"],review="human_assessed_region_model_refined_boundary",
      candidate_pixels=int(positive.sum()),semantic="vegetation-shaped reflected contours"))
  Image.fromarray(labels).save(path)
  row["counts"]={name:int((labels==cls).sum()) for name,cls in
    [("background",0),("human_nuisance",1),("vegetation_nuisance",2),
     ("other_weak_nuisance",3),("ignore",255)]}
  row["status"]="pseudo_relabelled_with_reviewed_roi" if row["id"] in TREE_ROIS else "pseudo_unreviewed"
 src["label_kind"]="unverified_semantic_pseudo_labels_with_3_reviewed_vegetation_rois"
 src["caution"]="NOT pixel-perfect manual GT; person predictions were discarded after false-positive QA"
 src["reviewed_regions"]=reviewed
 (ROOT/"semantic_annotations.json").write_text(json.dumps(src,ensure_ascii=False,indent=2))
 print('OCT_CURATION',json.dumps(reviewed),flush=True)
 print('LABEL_CLASS_SUMMARY',{key:sum(r['counts'][key] for r in src['annotations']) for key in
       ("background","human_nuisance","vegetation_nuisance","other_weak_nuisance","ignore")},flush=True)
if __name__=="__main__":main()
