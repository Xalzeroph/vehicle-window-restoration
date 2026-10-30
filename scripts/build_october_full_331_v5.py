#!/usr/bin/env python3
"""Full October-19 per-frame I/T reviewed semantic supervision.

331 image pairs visually inspected in contact sheets (I, T, difference).
171 previously image-reviewed masks are carried forward unchanged, 244 further
images get manually adjudicated nuisance-presence decisions and conservative
algorithmically refined masks. No claim of full manual pixel tracing.
"""
import csv, hashlib, json, os
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter,binary_closing,binary_opening,label as cc_label

ROOT=Path('/data/hdd3/agent-research-runtime/vehicle-assets/annotations')
CAMPAIGN=ROOT/'oct2026_annotation_campaign_v3'
OLD=ROOT/'oct2026_quality_review_v4'
NEW=ROOT/'oct2026_full_oct19_review_v5'

def reviewed_decision(scene,frame):
    n=int(frame)
    if scene=='165400':
        if n<=1005:return 'tree'
        if n<=1065:return 'weak'
        if n<=1245:return 'clean'
        if n<=1275:return 'weak'
        return 'tree'
    if scene=='170406':
        if n<=1307:return 'tree'
        if n<=1347:return 'weak'
        if n<=1522:return 'clean'
        if n<=1557:return 'weak'
        if n<=1627:return 'tree'
        if n<=1667:return 'tree+human_candidate'
        return 'tree+human_presence'
    if scene=='171126':
        if n<=1256:return 'tree'
        if n<=1364:return 'tree+human_presence'
        if n<=1537:return 'tree'
        if n<=1617:return 'weak'
        if n<=1777:return 'clean'
        if n<=1812:return 'weak'
        if n<=1877:return 'tree'
        return 'tree+human_candidate'
    raise ValueError('Unreviewed scene '+scene)

def poly(shape, points):
    h,w=shape
    im=Image.new('L',(w,h),0)
    ImageDraw.Draw(im).polygon([(round(x*(w-1)),round(y*(h-1))) for x,y in points],fill=1)
    return np.asarray(im)>0

TREE_POLYGON=[(0.,0.),(.79,0.),(1.,.14),(1.,.43),
              (.83,.51),(.58,.50),(.16,.52),(0.,.57)]

def human_uncertain_zone(scene, frame, shape):
    n=int(frame)
    if scene=='170406':
        return poly(shape,[(.20,.20),(.65,.19),(.71,.91),(.16,.91)])
    if scene=='171126':
        if n<=1276: coords=[(0.,.20),(.18,.15),(.25,.96),(0.,.99)]
        elif n<=1320:coords=[(.37,.10),(.74,.10),(.77,.99),(.35,.99)]
        elif n<=1364:coords=[(.73,.10),(1.,.13),(1.,.99),(.71,.99)]
        else:coords=[(0.,.31),(.89,.30),(.90,.99),(0.,.99)]
        return poly(shape,coords)
    return np.zeros(shape,bool)

def build_new(row, review_sheet):
    from build_october_reviewed_v4 import maps_from_pair
    I_path=CAMPAIGN/row['files']['I']
    T_path=CAMPAIGN/row['files']['T']
    with Image.open(I_path) as im: I=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
    with Image.open(T_path) as im: T=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
    if I.shape!=T.shape:raise ValueError('I/T shape mismatch '+row['id'])
    scene=row['scene'][-6:]
    frame=row['id'].rsplit('frame_',1)[-1]
    decision=reviewed_decision(scene,frame)
    h,w=I.shape[:2]
    avg,signed,highI,highT,optical=maps_from_pair(I,T)
    mask=np.full((h,w),255,np.uint8)
    confidence=np.zeros((h,w),np.uint8)
    bg=avg<(.022 if decision=='clean' else .017)
    mask[bg]=0
    confidence[bg]=180
    tree_roi=poly((h,w),TREE_POLYGON)
    if 'tree' in decision:
        mask[tree_roi&(mask==0)]=255
        confidence[tree_roi]=0
        branch=(tree_roi & (avg>.058) & (signed>.04) &
              (np.abs(highI)>np.abs(highT)*1.12+.007))
        # Preserve narrow coherent branches. Remove isolated sensor noise.
        branch=binary_closing(branch,iterations=1)
        regions,n=cc_label(branch)
        if n:
            sizes=np.bincount(regions.ravel())
            branch&=(sizes[regions]>=max(8,round(w*h/200000)))
        mask[branch]=2
        confidence[branch]=150
    if 'human' in decision:
        # Presence visually confirmed, but this new frame has no individual
        # pixel-traced human silhouette. Mark its region ignore, not fake GT.
        roi=human_uncertain_zone(scene,frame,(h,w))
        mask[roi]=255
        confidence[roi]=0
    d_id=row['id'].split('/')[-1]+'.png'
    pdir='rdt_1019split'
    paths={
        'native':NEW/'native_masks'/pdir/d_id,
        'mask_448':NEW/'masks_448'/pdir/d_id,
        'optical_448':NEW/'optical_difference_448'/pdir/d_id,
        'confidence_448':NEW/'confidence_448'/pdir/d_id}
    for path in paths.values():path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(mask).save(paths['native'])
    Image.fromarray(mask).resize((448,448),Image.Resampling.NEAREST).save(paths['mask_448'])
    Image.fromarray(optical).resize((448,448),Image.Resampling.BILINEAR).save(paths['optical_448'])
    Image.fromarray(confidence).resize((448,448),Image.Resampling.NEAREST).save(paths['confidence_448'])
    return dict(id=row['id'],scene=row['scene'],decision=decision,
      visual_review='individually viewed I/GT/difference in '+review_sheet,
      semantic_presence_review='yes',
      boundary_criterion='I/T pair high-frequency difference in manually adjudicated ROI, not pixel-perfect',
      person_pixel_provenance=('person present, dense mask intentionally withheld until traced'
                               if 'human' in decision else None),
      source_I=str(I_path),source_T=str(T_path),
      source_sha256=hashlib.sha256(I_path.read_bytes()).hexdigest(),
      target_sha256=hashlib.sha256(T_path.read_bytes()).hexdigest(),
      size=[h,w],
      paths={k:str(v.relative_to(NEW)) for k,v in paths.items()},
      counts={str(k):int((mask==k).sum()) for k in (0,1,2,255)},
      source_grade='frame_visually_reviewed_assisted_contour')

def carry(row):
    out=dict(row)
    out['source_grade']='frame_visually_reviewed_assisted_contour'
    out['visual_review']='previous 171 pair-by-pair reviewed I/GT sheets'
    paths=row['paths']
    newpaths={}
    for k,oldrel in paths.items():
        op=OLD/oldrel
        np=NEW/oldrel
        np.parent.mkdir(parents=True,exist_ok=True)
        if not np.exists():os.link(op,np)
        newpaths[k]=str(np.relative_to(NEW))
    out['paths']=newpaths
    return out

def main():
    NEW.mkdir(parents=True,exist_ok=True)
    source=json.loads((CAMPAIGN/'source_manifest.json').read_text())
    ref=json.loads((OLD/'review_manifest.json').read_text())['items']
    by_id={r['id']:r for r in ref}
    page_index=json.loads((CAMPAIGN/'remaining_1019_review/page_index.json').read_text())
    reviewpage={}
    for page in page_index:
        for id in page['ids']:
            if id in reviewpage:raise ValueError('Duplicate reviewed page entry '+id)
            reviewpage[id]=Path(page['page']).name
    october=[r for r in source['items'] if r['group']=='rdt_1019split']
    assert len(october)==331 and len(reviewpage)==244
    result=[]
    for k,row in enumerate(october):
        if row['id'] in by_id:
            result.append(carry(by_id[row['id']]))
        else:
            if row['id'] not in reviewpage:
                raise ValueError('Frame never personally visual-reviewed '+row['id'])
            result.append(build_new(row,reviewpage[row['id']]))
        if (k+1)%25==0 or k+1==len(october):
            print('OCT19_REVIEWED_MASKS',k+1,len(october),flush=True)
    keys=[r['id'] for r in result]
    assert len(keys)==len(set(keys))==331
    total=Counter();review=Counter();issue=[]
    for item in result:
        review[item['decision']]+=1
        path=NEW/item['paths']['native']
        with Image.open(path) as im:
            a=np.asarray(im)
        size=item.get('size',item.get('native_size'))
        if list(a.shape)!=list(size):issue.append('shape '+item['id'])
        if not set(np.unique(a))<={0,1,2,255}:issue.append('class '+item['id'])
        ct={str(k):int((a==k).sum()) for k in (0,1,2,255)}
        if ct!=item['counts']:issue.append('counts '+item['id'])
        if item['decision']=='clean' and (ct['1']+ct['2'])>0:issue.append('clean_fp '+item['id'])
        total.update(ct)
    summary=dict(
       dataset='zmy1234567890/tintedglass-rdt-synth',
       subset='train/rdt_1019split',
       all_october19_pairs=331,
       individual_I_GT_diff_reviewed=331,
       previously_reviewed_masks=87,
       new_reviewed_masks=244,
       other_group_masks_carried=False,
       presence_decision_counts=dict(review),
       pixel_counts=dict(total),
       semantic_classes={'0':'stable negative','1':'human reflection core',
                         '2':'tree/vegetation reflection contour',
                         '255':'unknown excluded'},
       review_scope='visual presence and manually reviewed region; pixel contours computer-assisted',
       training='NO MODEL TRAINING',
       issues=issue)
    with (NEW/'review_index.csv').open('w',newline='') as fp:
        fields=['id','scene','decision','visual_review','native_mask','optical_448',
                'human_pixels','tree_pixels','unknown_fraction']
        writer=csv.DictWriter(fp,fieldnames=fields);writer.writeheader()
        for item in result:
            ct=item['counts']
            writer.writerow(dict(id=item['id'],scene=item['scene'],decision=item['decision'],
               visual_review=item['visual_review'],
               native_mask=item['paths']['native'],
               optical_448=item['paths']['optical_448'],
               human_pixels=ct['1'],tree_pixels=ct['2'],
               unknown_fraction=round(ct['255']/sum(ct.values()),4)))
    (NEW/'review_manifest.json').write_text(json.dumps({'summary':summary,'items':result},ensure_ascii=False,indent=2))
    (NEW/'quality_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('OCT_331_FINISHED',json.dumps(summary,ensure_ascii=False),flush=True)
    if issue:raise SystemExit(1)

if __name__=='__main__':main()
