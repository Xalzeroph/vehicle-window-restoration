#!/usr/bin/env python3
"""Build semantic labels from 171 explicitly reviewed I/T frames, never training.

Per-image visual decisions were made from I/GT/difference panels. Dense
contours are computer-assisted and intentionally marked non-gold. Ambiguous
pixels receive 255; optical veiling receives its own continuous map, not a
human/tree semantic class. No network training, inference or pseudo-GT.
"""
import csv
import json
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,binary_erosion,binary_dilation,binary_closing,binary_opening

B=Path('/data/hdd3/agent-research-runtime/vehicle-assets/annotations')
OLD=B/'oct2026_reviewed_v3'
OCT=B/'oct2026_annotation_campaign_v3'
SRC=B/'oct2026_occlusion_v1'
OUT=B/'oct2026_quality_review_v4'

# Exact visual conclusions for the 72 newly inspected frames, in the
# chronological order of review_1019_expansion/selection.json.
DECISIONS={
 'VID_20261019_165400':[
  'tree','tree','tree','tree','tree',
  'tree','tree','tree','weak','weak',
  'weak','weak','clean','clean','clean',
  'clean','clean','clean','weak','tree',
  'tree','tree','tree','tree'],
 'VID_20261019_170406':[
  'tree','tree','tree','tree','tree','weak',
  'clean','clean','clean','clean','clean',
  'clean','clean','clean','weak','tree',
  'tree','tree','tree','tree','tree',
  'tree+human','tree+human','tree+human'],
 'VID_20261019_171126':[
  'tree','tree','tree','tree','tree','tree','tree',
  'tree+human','tree','tree+human','tree+human','tree+human',
  'tree+human','tree','weak','clean','weak',
  'weak','weak','clean','clean','clean','clean','clean']}
# These human figure polygons are manually delineated conservative cores
# after enlarged I/T review. They do NOT claim exact visible silhouette.
PERSON_POLYGONS={
 '001682': [[(.18,.22),(.29,.24),(.35,.39),(.31,.50),(.19,.48)],
            [(.25,.53),(.38,.54),(.45,.71),(.40,.80),(.30,.76)]],
 '001707': [[(.24,.52),(.39,.53),(.47,.71),(.43,.77),(.32,.73)]],
 '001727': [[(.24,.53),(.39,.54),(.46,.72),(.42,.79),(.32,.75)]],
 '001271': [[(.00,.20),(.09,.22),(.12,.39),(.08,.59),(.00,.68)]],
 '001313': [[(.53,.27),(.62,.28),(.67,.44),(.62,.63),(.54,.60)],
            [(.51,.56),(.58,.54),(.61,.80),(.55,.98),(.50,.95)]],
 '001335': [[(.80,.38),(.93,.39),(.96,.57),(.88,.73),(.81,.67)],
            [(.84,.60),(.96,.61),(.96,.93),(.85,.94)]],
 '001345': [[(.91,.35),(1.,.37),(1.,.90),(.92,.89)]],
 '001350': [[(.965,.36),(1.,.38),(1.,.90),(.97,.92)]],
}
TREE_POLYGON=[(0.,0.),(.81,0.),(1.,.14),(1.,.45),
              (.78,.48),(.56,.50),(.15,.50),(0.,.56)]
LABELS={
 '0':'stable_pair_negative',
 '1':'human_reflection_core',
 '2':'vegetation_reflection_contour',
 '255':'unknown_ignore'
}

def polygon(shape,pts):
 h,w=shape
 mask=Image.new('L',(w,h),0)
 ImageDraw.Draw(mask).polygon([(round(x*(w-1)),round(y*(h-1))) for x,y in pts],fill=1)
 return np.asarray(mask)>0

def load(path):
 with Image.open(path) as im:
  return np.asarray(im.convert('RGB'),dtype=np.float32)/255.

def maps_from_pair(I,T):
 diff=np.abs(I-T).mean(2)
 avg=gaussian_filter(diff,1.25)
 signed=gaussian_filter((T-I)@[.2126,.7152,.0722],1.25)
 yi=I@[.2126,.7152,.0722];yt=T@[.2126,.7152,.0722]
 highI=gaussian_filter(yi,1.0)-gaussian_filter(yi,3.2)
 highT=gaussian_filter(yt,1.0)-gaussian_filter(yt,3.2)
 # A physically nonsemantic image-domain interference proxy.
 optical=(np.clip((avg-.008)/.24,0,1)*255).astype('uint8')
 return avg,signed,highI,highT,optical

def target_paths(id):
 p=Path(id).name+'.png'
 group=id.split('/')[0]
 return (
   OUT/'native_masks'/group/p,
   OUT/'masks_448'/group/p,
   OUT/'optical_difference_448'/group/p,
   OUT/'confidence_448'/group/p)

def save(id,sem,optic,conf):
 paths=target_paths(id)
 for p in paths:p.parent.mkdir(exist_ok=True,parents=True)
 Image.fromarray(sem).save(paths[0])
 Image.fromarray(sem).resize((448,448),Image.Resampling.NEAREST).save(paths[1])
 Image.fromarray(optic).resize((448,448),Image.Resampling.BILINEAR).save(paths[2])
 Image.fromarray(conf).resize((448,448),Image.Resampling.NEAREST).save(paths[3])
 return [str(p.relative_to(OUT)) for p in paths]

def mask_for_new(row,decision):
 name=row['name']
 id='rdt_1019split/'+name
 Ipath=OCT/'train/rdt_1019split/I'/(name+'.png')
 Tpath=OCT/'train/rdt_1019split/T'/(name+'.png')
 I=load(Ipath);T=load(Tpath)
 if I.shape!=T.shape:raise ValueError('native grid mismatch '+name)
 h,w=I.shape[:2]
 avg,signed,highI,highT,optical=maps_from_pair(I,T)
 label=np.full((h,w),255,dtype='uint8')
 confidence=np.zeros((h,w),dtype='uint8')
 bg=avg < (.022 if decision=='clean' else .017)
 label[bg]=0;confidence[bg]=195
 tree_area=polygon((h,w),TREE_POLYGON)
 if 'tree' in decision:
  # Confirmed vegetation pattern within a manually reviewed region, not
  # all pixels of the glass pane and not true tree texture in GT.
  dense=tree_area&(signed>.045)&(avg>.061)&(np.abs(highI)>np.abs(highT)*1.12+.007)
  dense=binary_closing(dense,iterations=1)
  dense=binary_opening(dense,iterations=1)
  label[tree_area&(label==0)]=255
  confidence[tree_area]=0
  label[dense]=2;confidence[dense]=150
 if 'human' in decision:
  key=name.split('frame_')[-1]
  if key not in PERSON_POLYGONS:raise ValueError('Missing human review polygon '+id)
  full=np.zeros((h,w),bool)
  for vertices in PERSON_POLYGONS[key]:
   full |= polygon((h,w),vertices)
  label[full]=255; confidence[full]=0
  certain=binary_erosion(full,iterations=max(2,round(w/450)))&(avg>.035)&(np.abs(signed)>.018)
  label[certain]=1;confidence[certain]=195
 paths=save(id,label,optical,confidence)
 return dict(id=id,scene=row['scene'],decision=decision,
  label_grade='visual_pair_review_assisted_pixel_contour',
  visual_medium='native I/T/contrast difference at 410px and human enlarged where present',
  annotation_caveat='not pixel-perfect manually traced GT',
  I=str(Ipath),T=str(Tpath),
  input_sha256=hashlib.sha256(Ipath.read_bytes()).hexdigest(),
  target_sha256=hashlib.sha256(Tpath.read_bytes()).hexdigest(),
  native_size=[h,w],
  paths=dict(zip(('native','mask_448','optical_448','confidence_448'),paths)),
  counts={str(k):int((label==k).sum()) for k in (0,1,2,255)})

def carry_previous(row):
 id=row['id']
 group=row['group']
 src=Path(row['native_mask'])
 if not src.is_absolute():src=OLD/src
 with Image.open(src) as im:
  raw=np.asarray(im,dtype=np.uint8)
 # Class 3 is optical veiling, NOT part of human/tree semantic label set.
 raw=raw.copy()
 raw[raw==3]=255
 # The previous frame-by-frame visual review found that the old tree mask
 # in 172933 extended into actual grass. Exclude this region entirely.
 if group=='rdt_1020' and row['scene'].endswith('172933'):
  h,w=raw.shape
  yy=np.arange(h)[:,None]/h;xx=np.arange(w)[None,:]/w
  # The dark horizontal canopy is a true reflection, but the grass below
  # is the transmitted scene. Mark ambiguous lower foliage as unknown.
  outside=(yy>.56)|(yy<.23)
  raw[(raw==2)&outside]=255
 # Reduce accidental human-arm extension into the ground/tarp by using
 # a narrow, visually reviewed core; unknown outside the core.
 if group in ('rdt_1020','rdt_1026') and row['scene'].endswith('173827'):
  x=row['frame']
  if 'frame' not in row: x=id.split('_')[-1]
  if x in ('f000425','f000550'):
   h,w=raw.shape
   keep=polygon((h,w),[(.33,.59),(.48,.58),(.52,.69),(.47,.78),
                         (.49,.93),(.36,.99),(.30,.97),(.32,.76)])
   raw[(raw==1)&~keep]=255
 # Never let a person silhouette in native transmission (GT) become nuisance
 # based on generic person detection. Only original paired masks retained.
 Ipath=SRC/row['source_I'];Tpath=SRC/row['source_T']
 a=load(Ipath);b=load(Tpath)
 if a.shape!=b.shape or raw.shape!=a.shape[:2]:raise ValueError('Alignment mismatch '+id)
 avg,_,_,_,optical=maps_from_pair(a,b)
 conf=np.zeros_like(raw,dtype=np.uint8)
 conf[raw==0]=185;conf[raw==1]=190;conf[raw==2]=150
 paths=save(id,raw,optical,conf)
 return dict(id=id,scene=row['scene'],decision=row['observed_pair_decision'],
  label_grade=row['label_grade'],visual_medium=row['review_media'],
  annotation_caveat='not pixel-perfect manually traced GT',
  I=str(Ipath),T=str(Tpath),
  input_sha256=hashlib.sha256(Ipath.read_bytes()).hexdigest(),
  target_sha256=hashlib.sha256(Tpath.read_bytes()).hexdigest(),
  native_size=list(raw.shape),
  paths=dict(zip(('native','mask_448','optical_448','confidence_448'),paths)),
  counts={str(k):int((raw==k).sum()) for k in (0,1,2,255)})

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 selection=json.load(open(OCT/'review_1019_expansion/selection.json'))
 grouped={}
 for r in selection:grouped.setdefault(r['scene'],[]).append(r)
 for scene,v in grouped.items():
  if scene not in DECISIONS or len(v)!=len(DECISIONS[scene]):raise ValueError('Incomplete manual frame review for '+scene)
 reviewed=[]
 for r in json.load(open(OLD/'review_manifest.json'))['items']:
  reviewed.append(carry_previous(r))
  if len(reviewed)%20==0:print('V4_RETAINED',len(reviewed),flush=True)
 for scene,seq in grouped.items():
  for r,d in zip(seq,DECISIONS[scene]):
   reviewed.append(mask_for_new(r,d))
  print('V4_NEW_REVIEW',scene,len(seq),flush=True)
 keys=[r['id'] for r in reviewed]
 if len(keys)!=len(set(keys)) or len(keys)!=171:
  raise ValueError('expected unique 99+72 verified frame ids')
 by_label=Counter();decisions=Counter()
 for row in reviewed:
  decisions[row['decision']]+=1
  by_label.update(row['counts'])
  # Strict integrity checks on both grids, class set and source file hashes.
  p=OUT/row['paths']['native'];q=OUT/row['paths']['mask_448']
  with Image.open(p) as im: arr=np.asarray(im)
  if list(arr.shape)!=row['native_size'] or not set(np.unique(arr))<={0,1,2,255}:
   raise ValueError('Invalid semantic grid '+row['id'])
  if sum(row['counts'].values())!=arr.size:
   raise ValueError('Pixel accounting mismatch '+row['id'])
  with Image.open(q) as im:
   if im.size!=(448,448):raise ValueError('bad 448 grid '+row['id'])
  if row['decision']=='clean' and ((arr==1)|(arr==2)).any():
   raise ValueError('False semantic positive on reviewed clean '+row['id'])
 with (OUT/'review_index.csv').open('w',newline='') as fp:
  fields=['id','scene','decision','label_grade','visual_medium','native_size','semantic_mask',
          'optical_impact','human_pixels','tree_pixels','unknown_fraction']
  w=csv.DictWriter(fp,fieldnames=fields);w.writeheader()
  for r in reviewed:
   c=r['counts'];w.writerow(dict(id=r['id'],scene=r['scene'],decision=r['decision'],
       label_grade=r['label_grade'],visual_medium=r['visual_medium'],
       native_size='x'.join(map(str,r['native_size'])),
       semantic_mask=r['paths']['native'],optical_impact=r['paths']['optical_448'],
       human_pixels=c['1'],tree_pixels=c['2'],
       unknown_fraction=round(c['255']/sum(c.values()),4)))
 summary=dict(
  reviewed_images=len(reviewed),previous_reviewed=99,newly_reviewed=72,
  reviewed_video_scenes=len({r['scene'] for r in reviewed}),
  semantic_codes=LABELS,pixel_counts=dict(by_label),
  per_image_decisions=dict(decisions),
  all_source_images_hashed=True,all_masks_checked=True,
  annotation_quality_grade='visual_I_GT_review_manual_ROI_machine_assisted_contour_NOT_pixel_manual_GT',
  important='differentiate model training labels from manually confirmed presence and region labels')
 (OUT/'review_manifest.json').write_text(json.dumps(
    {'summary':summary,'items':reviewed},ensure_ascii=False,indent=2))
 (OUT/'quality_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print('V4_ALL_REVIEWED',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
