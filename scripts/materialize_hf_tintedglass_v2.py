#!/usr/bin/env python3


import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter

LEVELS=(0.30,0.45,0.60,0.75,1.00)


def to_u8(x):
    return np.clip(np.rint(x*255),0,255).astype(np.uint8)

def make(root,index,fold,out,variations=2,seed=20261029):
    data=json.loads(index.read_text())
    rows=data["folds"][fold]
    if (out/"manifest.json").exists():raise FileExistsError("Refuse to overwrite existing split")
    items=[];start=time.monotonic()
    out.mkdir(parents=True,exist_ok=True)
    for j,scene in enumerate(rows):
        sid=scene["scene_id"]
        T=np.asarray(Image.open(root/"T"/f"{sid}.png").convert("RGB"),dtype=np.float32)/255
        R=np.asarray(Image.open(root/"R"/f"{sid}.png").convert("RGB"),dtype=np.float32)/255
        m=np.asarray(np.load(root/"m"/f"{sid}.npy"),dtype=np.float32)
        g=np.asarray(np.load(root/"g"/f"{sid}.npy"),dtype=np.float32)
        if T.shape!=(448,448,3) or R.shape!=T.shape or m.shape!=(448,448) or g.shape!=m.shape:raise ValueError("Source dimensions invalid")


        if min(T.min(),R.min(),m.min())<0 or not np.isfinite(T).all() or not np.isfinite(R).all() or not np.isfinite(m).all() or not np.isfinite(g).all():raise ValueError("Invalid original data")
        for variant in range(variations):


            seed_bytes=hashlib.sha256(f"{sid}:{seed}:{variant}".encode()).digest()
            rng=np.random.default_rng(int.from_bytes(seed_bytes[:8],"big"))
            gain=1.0 if variant==0 else float(np.exp(rng.uniform(np.log(.65),np.log(1.45))))
            r=R if variant==0 else np.asarray(
                Image.fromarray(to_u8(R)).filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(.0,1.2)))),
                dtype=np.float32)/255
            r=r*gain


            noise=rng.normal(0,1.5/255,size=T.shape).astype(np.float32)
            for level in LEVELS:
                a=np.clip(level*g*m+(1-m),.01,1.0)
                target=a[...,None]*T
                obs_unclipped=target+r+noise
                observed=np.clip(obs_unclipped,0,1)
                valid=np.all((obs_unclipped>=0)&(obs_unclipped<1),axis=2)
                path=out/"samples"/f"{sid}_v{variant}_l{int(round(100*level)):03}"
                path.mkdir(parents=True,exist_ok=True)
                if not (path/"observed.png").is_file():
                    Image.fromarray(to_u8(observed)).save(path/"observed.png")
                if not (path/"target_tinted.png").is_file():
                    Image.fromarray(to_u8(target)).save(path/"target_tinted.png")


                meta={
                    "source_repo":data["source_repo"],"original_scene_id":sid,
                    "transmission_source":scene["t_src"],
                    "reflection_source":scene["r_src"],
                    "source_T":str(root/"T"/f"{sid}.png"),"source_R":str(root/"R"/f"{sid}.png"),
                    "film_coverage_mask":str(root/"m"/f"{sid}.npy"),
                    "film_gain_field":str(root/"g"/f"{sid}.npy"),
                    "formula":"I_srgb = clip(alpha(x)*T_srgb + gain*R_effective_srgb + noise), reflection attenuation exponent p=0",
                    "target":"alpha(x)*T_srgb, retains film tint; NOT clear-glass T",
                    "film_level":level,"reflection_gain":gain,"variant":variant,
                    "noise_std_8bit":1.5,"is_calibrated_hdr":False,
                    "outside_camera_car_interior_verified":False,
                    "license":"research-use-only; upstream source permissions may vary",
                    "valid_fraction":float(valid.mean()),"saturation_fraction":float((obs_unclipped>=1).any(axis=2).mean()),
                }
                (path/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
                ref_luma=(r@[.2126,.7152,.0722])
                tr_luma=(target@[.2126,.7152,.0722])
                frac=ref_luma/(ref_luma+tr_luma+1e-7)
                items.append({
                    "id":path.name,"relative_directory":str(path.relative_to(out)),
                    "source_t_key":scene["t_src"],"source_r_key":scene["r_src"],
                    "source_category":scene["t_category"],
                    "film_level":level,"reflection_gain":gain,
                    "reflection_fraction_median":float(np.median(frac)),
                    "valid_fraction":float(valid.mean()),"saturation_fraction":meta["saturation_fraction"]
                })
        if (j+1)%32==0:print("HF_TINTED_MATERIALIZED",fold,j+1,"/",len(rows),"out",len(items),"seconds",round(time.monotonic()-start,1),flush=True)
    manifest={"status":"RESEARCH_ONLY_TINTED_GLASS_SOURCE_PHOTOGRAPHIC_NOT_AUTOMOTIVE_EXTERIOR_CERTIFIED",
              "fold":fold,"source_repo":data["source_repo"],"film_attenuates_reflection":False,
              "original_scene_count":len(rows),"n_images":len(items),
              "not_used_for_training":fold=="test",
              "capture_geometry":"NOT_REGISTERED_OUTSIDE_CAMERA_OF_VEHICLE",
              "items":items}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("HF_TINTED_COMPLETE",fold,len(items),"from",len(rows),"real_source_research_scenes","wall_seconds",round(time.monotonic()-start,1),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--fold",choices=("train","validation","test"),required=True)
    p.add_argument("--variations",type=int,default=2)
    a=p.parse_args()
    if a.variations<=0:raise ValueError
    make(a.source,a.index,a.fold,a.out,a.variations)


if __name__=="__main__":main()
