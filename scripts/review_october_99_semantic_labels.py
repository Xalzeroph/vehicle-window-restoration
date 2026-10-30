#!/usr/bin/env python3
"""Frame-by-frame reviewed October semantic nuisance annotations.

Per-image presence decisions were made from I/GT/difference review sheets.
Hard masks are conservative pairwise contours within those reviewed regions;
they are NOT entirely hand-traced pixel-perfect labels. Unknown is explicit.
"""
import csv,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,binary_opening,binary_closing,binary_erosion

SOURCE=Path('/data/hdd3/agent-research-runtime/vehicle-assets/annotations/oct2026_occlusion_v1')
LEGACY=Path('/data/hdd3/agent-research-runtime/vehicle-assets/annotations/oct2026_pair_review_v2')
DEST=Path('/data/hdd3/agent-research-runtime/vehicle-assets/annotations/oct2026_reviewed_v3')
# Rows below are individually inspected I/GT images. Tokens:
# tree=verified reflected foliage; human=verified camera-side human;
# optical=other glass interference, not a semantic human/tree;
# weak=weak/unresolved optical phenomenon; clean=no visible nuisance.
# Human candidates that were visually ambiguous are not treated as person GT.
DECISIONS={
('rdt_1020','172719'):[
('f000525','optical'),('f000600','optical'),('f000700','weak'),('f000775','weak'),('f000850','weak')],
('rdt_1020','172933'):[
('f000014','tree'),('f000026','tree'),('f000150','tree+human_candidate'),('f000275','tree+human_candidate'),('f000375','weak')],
('rdt_1020','173237'):[
('f000275','optical'),('f000300','optical'),('f000325','optical'),('f000350','optical'),('f000375','weak')],
('rdt_1020','173349'):[
('f000050','tree'),('f000150','tree+human_candidate'),('f000275','tree'),('f000400','tree'),('f000500','weak')],
('rdt_1020','173604'):[
('f000005','tree+equipment'),('f000017','tree+equipment'),('f000050','tree+equipment'),('f000375','weak'),('f000675','weak')],
('rdt_1020','173827'):[
('f000425','human'),('f000600','human'),('f000725','human'),('f000875','optical'),('f000975','weak')],
('rdt_1020','162203'):[
('f000000','equipment'),('f000014','equipment'),('f000028','equipment'),('f000350','weak'),('f000675','weak')],
('rdt_1020','162623'):[
('f000475','weak'),('f000550','weak'),('f000800','equipment'),('f000850','equipment'),('f000925','weak')],
('rdt_1020','162901'):[
('f000450','equipment'),('f000600','weak'),('f000650','weak'),('f000700','clean'),('f000725','clean')],
('rdt_1020','163859'):[
('f000400','equipment'),('f000825','equipment'),('f001175','equipment'),('f001500','weak'),('f001825','clean')],
('rdt_1020','164653'):[
('f000625','clean'),('f000650','clean'),('f000675','clean'),('f000700','clean')],
('rdt_1020','170534'):[
('f000152','optical'),('f000207','optical'),('f000301','optical'),('f000350','weak'),('f000406','weak')],
('rdt_1020','172600'):[('f001146','weak')],
('rdt_1026','172719'):[
('f000525','optical'),('f000600','equipment'),('f000700','weak'),('f000775','weak'),('f000850','weak')],
('rdt_1026','172933'):[
('f000300','tree'),('f000325','tree'),('f000350','weak'),('f000375','weak')],
('rdt_1026','173237'):[('f000325','optical'),('f000350','optical')],
('rdt_1026','173349'):[('f000425','tree'),('f000475','tree'),('f000500','weak')],
('rdt_1026','173604'):[
('f000005','tree+equipment'),('f000015','tree+equipment'),('f000026','tree+equipment'),
 ('f000300','tree+equipment'),('f000575','weak')],
('rdt_1026','173827'):[
('f000550','human'),('f000650','human'),('f000750','human'),('f000875','optical'),('f000950','weak')]
}
# Reviewed source-side ROIs. These are NOT claimed exact pixel contours.
ROI={
 '172933': {'tree':[(.00,.26),(.82,.22),(1.,.46),(1.,.84),(.0,.82)]},
 '173349': {'tree':[(.45,.09),(.95,.10),(1.,.75),(.40,.88),(.36,.38)]},
 '173604': {'tree':[(.56,.25),(1.,.18),(1.,.84),(.54,.90)]},
}
# Verified person in bicycle scene: conservatively draw head+torso and
# extended arm, rather than calling the dark pavement/tarp "person".
HUMAN_SEGMENTS={
 'f000425':[
 [( .36,.59),(.44,.58),(.49,.65),(.48,.74),(.41,.80),(.33,.78)],
 [( .34,.72),(.47,.73),(.50,.89),(.43,.99),(.32,.99)],
 [( .46,.71),(.64,.69),(.70,.72),(.57,.77),(.49,.77)]],
 'f000600':[
 [(.39,.72),(.47,.70),(.51,.76),(.48,.82),(.37,.84)],
 [(.36,.79),(.49,.81),(.50,.99),(.30,.99)]],
 'f000725':[
 [(.37,.72),(.48,.68),(.53,.77),(.48,.86),(.37,.85)],
 [(.34,.80),(.49,.79),(.51,.99),(.31,.99)]],
 'f000550':[
 [(.37,.61),(.48,.58),(.49,.68),(.45,.76),(.34,.77)],
 [(.33,.72),(.48,.74),(.51,.98),(.31,.98)]],
 'f000650':[
 [(.39,.72),(.47,.70),(.49,.79),(.43,.86),(.33,.84)],
 [(.33,.80),(.48,.81),(.50,.99),(.30,.99)]],
 'f000750':[
 [(.35,.72),(.48,.69),(.51,.80),(.46,.87),(.33,.85)],
 [(.33,.81),(.48,.81),(.50,.99),(.30,.99)]],
}
def poly(shape,pts):
 h,w=shape
 im=Image.new('L',(w,h),0);d=ImageDraw.Draw(im)
 d.polygon([(round(x*(w-1)),round(y*(h-1))) for x,y in pts],fill=1)
 return np.asarray(im).astype(bool)

def annotated(row,decision):
 from_scene=row['scene'][-6:]
 file_id=row['id'].rsplit('_',1)[-1]
 ip=SOURCE/row['files']['I'];tp=SOURCE/row['files']['T']
 with Image.open(ip) as f: I=np.asarray(f.convert('RGB'),dtype=np.float32)/255.
 with Image.open(tp) as f: T=np.asarray(f.convert('RGB'),dtype=np.float32)/255.
 if I.shape!=T.shape: raise ValueError('native I/T mismatch '+row['id'])
 h,w=I.shape[:2]
 labels=np.full((h,w),255,dtype=np.uint8)
 confidence=np.zeros((h,w),dtype=np.uint8)
 diff=np.abs(I-T).mean(2)
 dsm=gaussian_filter(diff,1.7)
 light=gaussian_filter((T-I)@[.2126,.7152,.0722],1.6)
 lumI=I@[.2126,.7152,.0722]
 lumT=T@[.2126,.7152,.0722]
 hiI=gaussian_filter(lumI,0.9)-gaussian_filter(lumI,3.0)
 hiT=gaussian_filter(lumT,0.9)-gaussian_filter(lumT,3.0)
 if decision=='clean':
  # Scene-level clean is not proof that every pixel is exactly identical.
  bg=dsm<.022
 else:
  bg=dsm<.018
 labels[bg]=0;confidence[bg]=200
 # Distinguish optical veil from semantic object masks, do not give large
 # background intensity regions a false vegetation/human identity.
 is_tree='tree' in decision
 is_human=decision=='human'
 is_other=decision in ('optical','equipment') or 'equipment' in decision
 if is_tree:
  roi=ROI.get(from_scene,{}).get('tree')
  if from_scene=='172933' and row['group']=='rdt_1026':
   roi=[(.65,.43),(1.,.34),(1.,.96),(.65,.94)]
  if from_scene=='173349' and row['group']=='rdt_1026':
   roi=[(.49,.08),(1.,.08),(1.,.83),(.50,.89)]
  if from_scene=='173604' and row['group']=='rdt_1026':
   roi=[(.55,.16),(1.,.17),(1.,.84),(.53,.89)]
  if roi is None:
   raise ValueError('tree review ROI required '+row['id'])
  bounded=poly((h,w),roi)
  branch=bounded & (light>.041) & (dsm>.055) & (np.abs(hiI)>np.abs(hiT)*1.1+.006)
  branch=binary_closing(branch,iterations=1)
  branch=binary_opening(branch,iterations=1)
  labels[branch]=2;confidence[branch]=155
  # Foreground glass veil under the reflected canopy, visibly changed
  # but semantically unknown: weak 'other optical' not tree.
  nuisance=bounded&(dsm>.093)&(light>.064)&~branch
  labels[nuisance]=3;confidence[nuisance]=100
  # A real tree under a reflection mask may still match T. Preserve
  # uncertainty inside the reviewed ROI wherever mask not positively supported.
  uncertain=bounded & (labels==0)
  labels[uncertain]=255;confidence[uncertain]=0

 if is_human:
  pts=HUMAN_SEGMENTS.get(file_id)
  if not pts:raise ValueError('missing reviewed human silhouette '+row['id'])
  human_roi=np.zeros((h,w),dtype=bool)
  for segment in pts:human_roi|=poly((h,w),segment)
  inner=binary_erosion(human_roi,iterations=max(1,round(w/550)))
  active=inner&(dsm>.036)&(light>.021)
  labels[human_roi]=255;confidence[human_roi]=0
  labels[active]=1;confidence[active]=205

 if is_other:
  # Only pixelwise optical-difference evidence; NEVER call it an object mask.
  other=(dsm>.085)&(np.abs(light)>.040)
  if from_scene in ('172719','173237','173827','170534'):
   if from_scene=='172719': region=poly((h,w),[(0,0),(1,0),(1,.4),(0,.4)])
   elif from_scene=='173237':region=poly((h,w),[(0,0),(1,0),(1,.52),(0,.52)])
   elif from_scene=='173827':region=poly((h,w),[(0,0),(1,0),(1,.7),(0,.7)])
   elif from_scene=='170534':region=poly((h,w),[(0,0),(1,0),(1,.48),(0,.48)])
   other &= region
  labels[other]=3;confidence[other]=95
 if decision.startswith('weak'):
  # Hard positive supervision is inappropriate for borderline examples.
  pass
 if 'candidate' in decision:
  # Candidate human silhouettes are not labelled as true human.
  candidate=poly((h,w),[(.55,.16),(1.,.1),(1.,.94),(.55,.94)])
  labels[candidate&(dsm>.035)&(labels==0)]=255
  confidence[(labels==255)]=0

 dest=DEST/'native_masks'/row['group']/(Path(row['id']).name+'.png')
 dest.parent.mkdir(parents=True,exist_ok=True)
 Image.fromarray(labels).save(dest)
 low=DEST/'masks_448'/row['group']/(Path(row['id']).name+'.png')
 low.parent.mkdir(parents=True,exist_ok=True)
 Image.fromarray(labels).resize((448,448),Image.Resampling.NEAREST).save(low)
 cp=DEST/'pixel_confidence'/row['group']/(Path(row['id']).name+'.png')
 cp.parent.mkdir(parents=True,exist_ok=True)
 Image.fromarray(confidence).save(cp)
 counts={str(k):int((labels==k).sum()) for k in (0,1,2,3,255)}
 return dict(id=row['id'],group=row['group'],scene=row['scene'],
   frame=file_id,observed_pair_decision=decision,
   sem_human_reviewed=bool(is_human),sem_tree_reviewed=bool(is_tree),
   label_grade='visual_I_T_frame_presence_manual_ROI_conservative_auto_contour',
   pixel_boundary_grade='not_full_resolution_manual_tracing',
   review_media='I_T_abs_difference_contact_sheet_each_frame; person candidates enlarged separately',
   source_I=row['files']['I'],source_T=row['files']['T'],
   image_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),
   target_sha256=hashlib.sha256(tp.read_bytes()).hexdigest(),
   native_shape=[h,w],
   native_mask=str(dest.relative_to(DEST)),
   mask_448=str(low.relative_to(DEST)),
   confidence_map=str(cp.relative_to(DEST)),counts=counts)

def main():
 DEST.mkdir(parents=True,exist_ok=True)
 src=json.load(open(SOURCE/'source_manifest.json'))['items']
 old=json.load(open(LEGACY/'review_manifest.json'))['items']
 catalog={r['id']:r for r in src}
 expect={}
 for (group,scene),rows in DECISIONS.items():
  for suffix,decision in rows:
   matches=[r for r in src if r['group']==group and r['scene'][-6:]==scene and r['id'].endswith(suffix)]
   if len(matches)!=1:raise ValueError('review decision missing/duplicate '+str((group,scene,suffix,len(matches))))
   expect[matches[0]['id']]=decision
 source_remaining={r['id'] for r in src if r['group']!='rdt_1019split'}
 if source_remaining!=set(expect):
  raise ValueError('Unreviewed input pairs: '+str(source_remaining.symmetric_difference(set(expect))))
 reviewed=[]
 # Keep the earlier fully viewed October 19 casework: no changed semantic claims.
 for r in old:
  reviewed.append(dict(
   id=r['id'],group='rdt_1019split',scene=r['scene'],
   observed_pair_decision=r['observed_pair_decision'],
   label_grade=r['review_grade'],pixel_boundary_grade='conservative_auto_contour',
   review_media='original_res_3_scene_video_side_by_side_I_T',
   source_I=r['source_I'],source_T=r['source_T'],
   native_shape=r['native_image_shape'],
   native_mask=str(LEGACY/r['semantic_native']),
   mask_448=str(LEGACY/r['semantic_448']),
   reused_prior_review=True,
   counts=r['counts']))
 for i,(key,decision) in enumerate(expect.items()):
  row=annotated(catalog[key],decision)
  reviewed.append(row)
  if i%10==0 or i==len(expect)-1:print('REVIEWED',i+1,len(expect),flush=True)
 
 with (DEST/'review_index.csv').open('w',newline='') as out:
  keys=['id','group','scene','frame','observed_pair_decision','label_grade','native_mask','mask_448',
      'review_media','source_I','source_T','counts']
  w=csv.DictWriter(out,fieldnames=keys);w.writeheader()
  for r in reviewed:w.writerow({k:json.dumps(r[k]) if k=='counts' else r.get(k,'') for k in keys})
 manifest={'dataset':'zmy1234567890/tintedglass-rdt-synth',
  'n_framewise_visually_reviewed':len(reviewed),'n_new_framewise_reviewed':len(expect),
  'source_published_revision':json.load(open(SOURCE/'source_manifest.json'))['revision'],
  'pixel_semantics':{'0':'verified_stable_negative','1':'human_reflection_core',
    '2':'vegetation_reflection_contour','3':'other_optical_difference_weak',
    '255':'ignore_ambiguous'},
  'provenance':'Per-frame I/T side-by-side visual assessment, human region polygons and edge-refined conservative contours.',
  'NOT_golden':'Most boundary contours generated from I/T differences, not hand-traced full-resolution GT.',
  'items':reviewed}
 (DEST/'review_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 total=Counter()
 for r in reviewed:total.update({k:v for k,v in r['counts'].items()})
 summary={'frames':len(reviewed),'new':len(expect),'categories':dict(Counter(r['observed_pair_decision'] for r in reviewed)),
          'pixel_counts':dict(total),'pixel_grade':'instance-level visual review + assisted contours'}
 (DEST/'quality_summary.json').write_text(json.dumps(summary,indent=2))
 print('V3_REVIEW_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
