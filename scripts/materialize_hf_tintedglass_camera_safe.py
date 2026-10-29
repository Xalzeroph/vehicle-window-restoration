#!/usr/bin/env python3


import argparse,json,hashlib,time
from pathlib import Path
from PIL import Image,ImageFilter
import numpy as np

LEVELS=(.30,.45,.60,.75,1.)

def to_u8(x):
    return np.rint(np.clip(x,0,1)*255).astype("uint8")

def run(root,index,photo_manifest,out,seed=20261029):
    if out.exists():raise FileExistsError(out)
    source=json.loads(index.read_text())
    filtered=json.loads(photo_manifest.read_text())
    if filtered["status"]!="RESEARCH_ONLY_TINTED_GLASS_SOURCE_PHOTOGRAPHIC_NOT_AUTOMOTIVE_EXTERIOR_CERTIFIED":
        raise ValueError("Unrecognized source manifest")
    by_sid={}
    for row in filtered["items"]:
        sid=row["id"].split("_v")[0]
        if sid in by_sid:continue
        by_sid[sid]=row
    permitted=set(by_sid)
    original=json.loads((root/"meta.json").read_text())
    clip_limit=float(original["clip_cap"])
    if not 0<clip_limit<1:raise ValueError("Invalid published source clip cap")
    out.mkdir(parents=True)
    items=[];times=time.monotonic()
    for i,sid in enumerate(sorted(permitted)):
        row=by_sid[sid]
        T=np.asarray(Image.open(root/"T"/f"{sid}.png").convert("RGB"),dtype=np.float32)/255
        R=np.asarray(Image.open(root/"R"/f"{sid}.png").convert("RGB"),dtype=np.float32)/255
        m=np.asarray(np.load(root/"m"/f"{sid}.npy"),dtype=np.float32)
        g=np.asarray(np.load(root/"g"/f"{sid}.npy"),dtype=np.float32)
        if T.shape!=(448,448,3) or R.shape!=T.shape:raise ValueError(sid)
        for v in (0,1):
            digest=hashlib.sha256(f"{sid}:{seed}:{v}".encode()).digest()
            rng=np.random.default_rng(int.from_bytes(digest[:8],"big"))
            reflection_gain=1.0 if v==0 else float(np.exp(rng.uniform(np.log(.65),np.log(1.45))))
            r=R if v==0 else np.asarray(
                Image.fromarray(to_u8(R)).filter(ImageFilter.GaussianBlur(
                    radius=float(rng.uniform(0,1.2)))),dtype=np.float32)/255
            r=r*reflection_gain
            noise=rng.normal(0,1.5/255,size=T.shape).astype(np.float32)


            curves=[np.clip(level*g*m+(1-m),.01,1.) for level in LEVELS]


            worst_alpha=np.maximum.reduce(curves)
            max_signal=np.max(worst_alpha[...,None]*T+r+noise,axis=2)
            cap_quantile=float(np.quantile(max_signal,1-clip_limit))
            exposure=float(min(1.,(1-0.5/255)/max(cap_quantile,1e-8)))
            for level,alpha in zip(LEVELS,curves):
                clean=(alpha[...,None]*T)*exposure
                sensor=(alpha[...,None]*T+r+noise)*exposure
                sat=((sensor>=1).any(axis=2))


                frac=float(sat.mean())
                if frac>clip_limit+.001:raise AssertionError((sid,v,level,frac))
                item_id=f"{sid}_v{v}_l{int(level*100):03d}"
                folder=out/"samples"/item_id
                folder.mkdir(parents=True)
                Image.fromarray(to_u8(sensor)).save(folder/"observed.png")
                Image.fromarray(to_u8(clean)).save(folder/"target_tinted.png")
                metadata={
                    "original_scene_id":sid,"original_source_repo":source["source_repo"],
                    "source_t_key":row["source_t_key"],"source_r_key":row["source_r_key"],
                    "source_category":row["source_category"],
                    "formula":"camera_exposure*(alpha*T_srgb+R_srgb+noise), common scalar camera response",
                    "calibrated_real_camera":False,
                    "post_mix_color_matching":False,"same_color_transform_for_input_target":True,
                    "published_clip_cap":clip_limit,"solved_exposure":exposure,
                    "film_level":level,"reflection_gain":reflection_gain,
                    "saturation_fraction":frac,
                    "original_T":str(root/"T"/f"{sid}.png"),
                    "original_R":str(root/"R"/f"{sid}.png"),
                    "mask_interpreted_as_film_coverage_not_window_polygon":True}
                (folder/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
                ref_luma=np.mean(r,axis=2)
                t_luma=np.mean(clean,axis=2)
                visibility=exposure*ref_luma/(exposure*ref_luma+t_luma+1e-7)
                items.append({
                    "id":item_id,"relative_directory":str(folder.relative_to(out)),
                    "source_t_key":row["source_t_key"],"source_r_key":row["source_r_key"],
                    "source_category":row["source_category"],"film_level":level,
                    "reflection_gain":reflection_gain,"camera_exposure":exposure,
                    "saturation_fraction":frac,
                    "reflection_fraction_median":float(np.median(visibility))})
        if (i+1)%32==0:print("CAMERA_SAFE_TINT_GENERATED",i+1,"/",len(by_sid),"images",len(items),"s",round(time.monotonic()-times,1),flush=True)
    result={"status":"RESEARCH_ONLY_TINTED_GLASS_SOURCE_PHOTOGRAPHIC_NOT_AUTOMOTIVE_EXTERIOR_CERTIFIED",
            "augmentation":"analytic shared scalar exposure with source clip cap",
            "source_scene_count":len(by_sid),"n_images":len(items),"items":items,
            "real_automotive_exterior_camera_verified":False}
    (out/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print("CAMERA_SAFE_TINT_COMPLETED",json.dumps({"images":len(items),"scenes":len(by_sid),"seconds":round(time.monotonic()-times,1)}),flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--source",type=Path,required=True)
    a.add_argument("--index",type=Path,required=True)
    a.add_argument("--photo-manifest",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    p=a.parse_args()
    run(p.source,p.index,p.photo_manifest,p.out)


if __name__=="__main__":main()
