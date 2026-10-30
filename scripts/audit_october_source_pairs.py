#!/usr/bin/env python3
"""Audit complete published I/T/mask corpus without claiming semantic review."""
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk=f.read(1024*1024)
            if not chunk:break
            digest.update(chunk)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--review-manifest',type=Path,required=True)
    args=p.parse_args()
    source=json.loads((args.source/'source_manifest.json').read_text())
    reviewed={r['id'] for r in json.loads(args.review_manifest.read_text())['items']}
    if len(reviewed)!=171:
        raise ValueError('Expected finalized 171 manual-image reviewed records')
    records=[]
    issues=[]
    pair_hashes=defaultdict(list)
    by_scene=Counter()
    for i,row in enumerate(source['items']):
        name=row['id']
        files={}
        for key in ('I','T','masks'):
            path=args.source/row['files'][key]
            if not path.exists():
                issues.append(f'{name}: missing {key}');continue
            files[key]=path
        if len(files)!=3:continue
        try:
            with Image.open(files['I']) as im:
                size_I=im.size
                im.verify()
            with Image.open(files['T']) as im:
                size_T=im.size
                im.verify()
            arr=np.load(files['masks'],allow_pickle=False)
            if arr.shape!=(448,448):
                issues.append(f'{name}: unexpected mask shape {arr.shape}')
            if size_I != size_T:
                issues.append(f'{name}: I/T grid misaligned {size_I}/{size_T}')
            shaI=checksum(files['I']);shaT=checksum(files['T'])
            pair_hashes[(shaI,shaT)].append(name)
            by_scene[row['scene']]+=1
            records.append(dict(id=name,scene=row['scene'],group=row['group'],
                native_size=list(size_I),
                image_sha256=shaI,target_sha256=shaT,
                upstream_mask_shape=list(arr.shape),
                upstream_mask_kind='weak Omega proposal, not semantic GT',
                visual_review_status='I/T pair visually reviewed in this turn' if name in reviewed
                                     else 'NOT independently reviewed',
                label_status='reviewed contour-assisted semantic mask' if name in reviewed
                             else 'NO verified semantic label'))
        except Exception as error:
            issues.append(f'{name}: {type(error).__name__} {error}')
        if (i+1)%100==0 or i+1==len(source['items']):
            print('OCT_SOURCE_AUDIT',i+1,len(source['items']),'issues',len(issues),flush=True)
    dup=[v for v in pair_hashes.values() if len(v)>1]
    report=dict(published_revision=source['revision'],total_expected=len(source['items']),
        valid_filetriples=len(records),reviewed=sum(r['id'] in reviewed for r in records),
        NOT_reviewed=sum(r['id'] not in reviewed for r in records),
        video_scene_count=len(by_scene),by_group=dict(Counter(r['group'] for r in records)),
        duplicate_pair_content_groups=dup,issues=issues)
    (args.source/'source_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (args.source/'audited_manifest.json').write_text(json.dumps(dict(
      summary=report,records=records),indent=2,ensure_ascii=False))
    print('OCT_SOURCE_AUDIT_DONE',json.dumps({k:v for k,v in report.items() if k not in ('duplicate_pair_content_groups','issues')},ensure_ascii=False),flush=True)
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
