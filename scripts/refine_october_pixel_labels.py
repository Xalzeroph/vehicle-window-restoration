#!/usr/bin/env python3
"""Image-pair reviewed October annotations, preserving all uncertainties.

The author-verified decisions are whether and where a nuisance instance exists.
Pixel contours are conservatively refined from paired RGB differences and
differential edge structure; they are NOT claimed hand-traced GT.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter, sobel, binary_opening, binary_closing, binary_erosion

ROOT=Path('/data/hdd3/agent-research-runtime/vehicle-assets/annotations/oct2026_occlusion_v1')
OUT=Path('/data/hdd3/agent-research-runtime/vehicle-assets/annotations/oct2026_pair_review_v2')
OUT.mkdir(parents=True,exist_ok=True)

# Scene-pair visual checks: I and reference T are inspected together. Tree
# is an actual camera-side reflected canopy, never the real outdoor tree.
# These normalized polygons are restricted instance proposals, not semantic masks.
DECISIONS={
 '165400': {
    '000000':('tree', .53), '000975':('tree',.30),
    '001115':('clean',0), '001250':('weak',.12), '001385':('tree',.52)},
 '170406': {
    '000054':('tree',.52), '001342':('clean',0),
    '001477':('clean',0), '001607':('tree',.49),
    '001732':('tree+human',.51)},
 '171126': {
    '000360':('tree',.54), '001547':('clean',0),
    '001692':('weak',.13), '001837':('tree',.49),
    '001982':('tree',.53)}}
# High-confidence human-reflection ARM interior in I of frame 170406/001732,
# relative coordinates of full-resolution I (from supervised visual review).
HUMAN_ARM=[
 (.333,.544),(.365,.541),(.391,.558),(.416,.605),
 (.437,.675),(.482,.746),(.498,.778),(.464,.793),
 (.431,.767),(.405,.751),(.371,.718),(.348,.663),
 (.326,.612),(.317,.566)]
# Tree reflection always occurs above the foreground objects. ROI is
# deliberately broad; tree-shaped pixels require extra edge evidence.
TREE_ROI=[(0.,0.),(.66,0.),(.85,.12),(.90,.28),
          (.78,.41),(.55,.48),(.28,.47),(0.,.50)]

def poly_mask(size,vertices):
    w,h=size
    m=Image.new('L',(w,h),0)
    draw=ImageDraw.Draw(m)
    draw.polygon([(round(x*(w-1)),round(y*(h-1))) for x,y in vertices],fill=1)
    return np.asarray(m)>0

def rel_parts(row):
    parts=row['id'].split('frame_')
    return row['scene'][-6:],parts[-1]

def annotate(row):
    a,b=[],[]
    for side in ('I','T'):
        filename=ROOT/row['files'][side]
        with Image.open(filename) as im:
            (a if side=='I' else b).append(np.array(im.convert('RGB'),dtype=np.float32)/255)
    I=a[0];T=b[0]
    if I.shape!=T.shape:
        raise ValueError('Pair grid mismatch '+row['id'])
    h,w=I.shape[:2]
    scene,frame=rel_parts(row)
    decision,extent=DECISIONS[scene][frame]
    # A robust optical-difference map; weak residual misregistration is ignored.
    signed=gaussian_filter((I-T).mean(2),sigma=max(1.0,w/1200))
    mean_abs=gaussian_filter(np.abs(I-T).mean(2),sigma=max(1.0,w/1400))
    Yi=I@[.2126,.7152,.0722]
    Yt=T@[.2126,.7152,.0722]
    highI=gaussian_filter(Yi,1)-gaussian_filter(Yi,3)
    highT=gaussian_filter(Yt,1)-gaussian_filter(Yt,3)
    edge_d=np.abs(highI-highT)
    tree_region=poly_mask((w,h),TREE_ROI)&(np.arange(h)[:,None]<int(extent*h))
    labels=np.full((h,w),255,dtype=np.uint8)

    # Only low-difference, stable pixels are verified negative.
    stable=mean_abs<.020
    # Do not turn undetectable branches within a confirmed nuisance ROI into
    # false background negatives. Unknown is preferable to wrong supervision.
    if decision in ('tree','tree+human','weak'):
        labels[stable & ~tree_region]=0
    else:
        labels[stable]=0
    # Real nuisance class 3: reflection veiling evidence is bright/different
    # on I versus T and within an independently reviewed nuisance ROI.
    if decision in ('tree','tree+human','weak'):
        evidence=tree_region & (signed>.060) & (mean_abs>.061)
        labels[evidence]=3
        # Vegetation contours require local high-frequency structure
        # from I that is NOT explained by clean reference T.
        if decision != 'weak':
            branches=evidence & (edge_d>.020) & (np.abs(highI)>np.abs(highT)*1.14+.007)
            branches=binary_closing(branches,iterations=1)
            branches=binary_opening(branches,iterations=1)
            labels[branches]=2

    if decision=='tree+human':
        polygon=poly_mask((w,h),HUMAN_ARM)
        # Only inner arm receives hard supervision; do not conflate dark bag
        # or tree with body silhouette. Ambiguous boundary stays 255.
        labels[polygon]=255
        sure=binary_erosion(polygon,iterations=max(2,round(w/350)))
        sure &= mean_abs>.032
        labels[sure]=1

    # Preserve native I/T resolution and a separate 448-space label.
    src=Path(row['id'])
    dest=OUT/'native_masks'/src
    dest.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(labels).save(dest.with_suffix('.png'))
    dest448=OUT/'masks_448'/src
    dest448.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(labels).resize((448,448),Image.Resampling.NEAREST).save(dest448.with_suffix('.png'))
    counts={str(v):int((labels==v).sum()) for v in (0,1,2,3,255)}
    return {
      'id':row['id'],'source_group':row['group'],'scene':row['scene'],
      'frame':frame,'observed_pair_decision':decision,
      'verified_instance_categories':(['vegetation_reflection','human_reflection_arm'] if decision=='tree+human'
        else ['vegetation_reflection'] if decision=='tree' else
             ['possible_vegetation_reflection'] if decision=='weak' else []),
      'category_review':'visual_I_T_pair_review',
      'contour_provenance':'automatic_conservative_pair_difference_in_reviewed_ROI',
      'human_contour_provenance':('manually_delineated_polygon_eroded_core' if decision=='tree+human' else None),
      'reviewed_region_geometry':({
          'tree_candidate_polygon_normalized':TREE_ROI,
          'upper_fraction_limit':extent,
          'human_arm_polygon_normalized':HUMAN_ARM if decision=='tree+human' else None
        } if decision not in ('clean',) else None),
      'source_I':row['files']['I'],'source_T':row['files']['T'],
      'source_sha256':hashlib.sha256((ROOT/row['files']['I']).read_bytes()).hexdigest(),
      'native_image_shape':[h,w],
      'semantic_native':str(dest.with_suffix('.png').relative_to(OUT)),
      'semantic_448':str(dest448.with_suffix('.png').relative_to(OUT)),
      'counts':counts, 'labels':{'0':'no_semantic_nuisance_supported',
            '1':'camera_side_human_reflection','2':'camera_side_vegetation_reflection',
            '3':'other_bright_optical_nuisance','255':'ignore_unknown'},
      'review_grade':('manually_reviewed_presence_and_arm_interior' if decision=='tree+human'
           else 'manually_reviewed_presence_automatic_contour')}

def main():
    source=json.loads((ROOT/'source_manifest.json').read_text())['items']
    rows=[r for r in source if r['group']=='rdt_1019split']
    expected={r['scene'][-6:]+'_'+rel_parts(r)[1] for r in rows}
    labels={scene+'_'+frame for scene,v in DECISIONS.items() for frame in v}
    if expected!=labels:raise ValueError('Incomplete exact video/frame visual review '+str(expected.symmetric_difference(labels)))
    out=[annotate(r) for r in rows]
    manifest={'dataset':'zmy1234567890/tintedglass-rdt-synth',
        'subset':'train/rdt_1019split',
        'n_items':len(out),
        'human_vs_vegetation':'camera-side image contamination; not dark real-scene foreground',
        'meaning_of_255':'do not supervise uncertain pixels',
        'ground_truth_claim':'scene-level visual review, refined contours; NOT fully hand-traced pixel GT',
        'mask_alignment':'native mask saved with exact I/T image dimensions; 448 version is explicit nearest resize',
        'source_revision':json.loads((ROOT/'source_manifest.json').read_text())['revision'],
        'items':out}
    (OUT/'review_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('SAVED',OUT,flush=True)
    print('FRAMES',len(out),'DECISIONS', {k:sum(r['observed_pair_decision']==k for r in out) for k in ('tree','tree+human','weak','clean')})
    print('COUNT_TOTAL',{k:sum(a['counts'][k] for a in out) for k in ('0','1','2','3','255')})
    for r in out:print('SAMPLE',r['scene'][-6:],r['frame'],r['observed_pair_decision'],r['counts'])

if __name__=='__main__':main()
