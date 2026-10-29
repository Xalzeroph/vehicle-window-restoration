#!/usr/bin/env python3

import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
from vehicle_recipe_index import import_source,dedupe_source_images,stable_balanced_folds,RecipeIndex
from materialize_vehicle_preview import native_calibration
from build_vehicle_matched_ablation import mixture_pair_from_source_meta

def digest(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"big")

def synth_meta(recipe,preset,cal,seed):
    with Image.open(recipe["outdoor_path"]) as im:
        im=im.convert("RGB");size=(min(im.width,384),min(im.height,256))
        rgb=np.asarray(im.resize(size),dtype=np.float64)/255
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    od=np.asarray(preset["glass"]["optical_depth_rgb"],dtype=float)
    reference=linear*np.exp(-od)[None,None,:]
    exposure=.88/max(float(np.quantile(reference,.995)),1e-4)
    glass=dict(preset["glass"],normal_xyz=[.04,.12,1.],
               camera_to_glass_m=.6)
    camera=dict(exposure=exposure,wb_rgb=[1.05,1.,.98],
                full_well_e=18000.,read_noise_e=3.,
                focal_length_mm=35.,f_number=float(preset["camera_f_number"]),
                focus_distance_m=40.,reflection_distance_m=.8,
                sensor_width_mm=36.)
    return {"source_transmission":{
            "path":recipe["outdoor_path"],"sha256":recipe["outdoor_sha256"],
            "source_group":recipe["outdoor_capture_group"],
            "source_dataset":recipe["outdoor_dataset"],
            "source_rights":recipe["source_rights"][0]},
            "source_reflection":{
            "path":recipe["reflection_path"],"sha256":recipe["reflection_sha256"],
            "source_group":recipe["reflection_capture_group"],
            "source_dataset":recipe["reflection_dataset"],
            "source_rights":recipe["source_rights"][1]},
            "camera_calibration":cal,"glass":glass,"camera":camera,
            "seed":digest(recipe["recipe_id"]+"-sensor-"+str(seed)),
            "source_recipe_id":recipe["recipe_id"],
            "relative_cabin_illuminance_hypothesis":2**float(
                preset["relative_cabin_radiance_ev_HYPOTHESIS"]),
            "dataset_role":"UNVERIFIED_RESEARCH_PREVIEW_ONLY",
            "candidate_not_training_release":True}

def metrics_existing(folder):
    with np.load(folder/"radiance.npz",allow_pickle=False) as z:
        valid=z["valid_ghost"]&~z["saturated"]
        t=z["transmitted"].astype(float);r=z["reflection"].astype(float)
        lum=np.array([.2126,.7152,.0722])
        ratio=(r@lum)/((t+r)@lum+1e-8)
        return {"usable_fraction":float(valid.mean()),
                "source_coverage_fraction":float(z["valid_ghost"].mean()),
                "saturation_fraction":float(z["saturated"].mean()),
                "reflection_visibility_p50":float(np.median(ratio[valid])) if valid.any() else 0.}

def main():
    p=argparse.ArgumentParser()
    for name in ("a2d2-manifest","kitti-manifest","cabin-manifest",
                 "a2d2-calibration","kitti-calibration","presets","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--fold",choices=("train","val","test"),required=True)
    p.add_argument("--max-accepted",type=int,required=True)
    p.add_argument("--max-attempts",type=int,required=True)
    p.add_argument("--side",type=int,default=256)
    p.add_argument("--seed",type=int,default=20261015)
    args=p.parse_args()
    if args.side<=0 or args.side%32 or args.max_accepted<1 or args.max_attempts<args.max_accepted:
        raise ValueError("Invalid sampling budget")
    if (args.out/"manifest.json").exists():
        raise FileExistsError("Do not overwrite finalized corpus")
    t=dedupe_source_images(import_source(args.a2d2_manifest,"a2d2")+
                           import_source(args.kitti_manifest,"kitti"))
    r=dedupe_source_images(import_source(args.cabin_manifest,"driveact"))
    assign_t,folds_t=stable_balanced_folds(t,args.seed)
    assign_r,folds_r=stable_balanced_folds(r,args.seed)
    ts=[x for x in t if assign_t[x.capture_group]==args.fold]
    rs=[x for x in r if assign_r[x.capture_group]==args.fold]
    presets=json.loads(args.presets.read_text())
    index=RecipeIndex(ts,rs,presets,args.seed)
    calibration_a2d2=json.loads(args.a2d2_calibration.read_text())
    calibration_kitti=json.loads(args.kitti_calibration.read_text())
    lookup={x.sha256:x for x in ts}
    preset_map={x["key"]:x for x in presets}
    args.out.mkdir(parents=True,exist_ok=True)
    accepted=[];rejected=[];started=time.monotonic()
    for i in range(min(args.max_attempts,index.count)):
        if len(accepted)>=args.max_accepted:break
        entry=index.get(i);token=entry["recipe_id"]
        src=lookup[entry["outdoor_sha256"]]
        cal=native_calibration(src,calibration_a2d2,calibration_kitti)
        config=synth_meta(entry,preset_map[entry["parameter_key"]],cal,args.seed)
        folder=args.out/"samples"/token
        if folder.exists():
            existing=json.loads((folder/"metadata.json").read_text())
            if existing.get("source_recipe_id")!=token:
                raise ValueError("Incompatible preexisting result "+token)
            stats=metrics_existing(folder)
        else:
            stats=mixture_pair_from_source_meta(config,folder,"laminate",args.side,
                                               token,"stable capture group recipe index")
        reason=None
        if stats["usable_fraction"]<.85:reason="insufficient_valid_rays"
        elif stats["saturation_fraction"]>.03:reason="excessive_sensor_clipping"
        elif not np.isfinite(stats["reflection_visibility_p50"]):reason="nonfinite_visibility"
        elif stats["reflection_visibility_p50"]<.0005:reason="vanishing_reflection"
        if reason:
            rejected.append(dict(id=token,reason=reason,statistics=stats))
            destination=args.out/"rejected"/token
            destination.parent.mkdir(parents=True,exist_ok=True)
            folder.rename(destination)
            continue
        accepted.append({"id":token,"relative_directory":"samples/"+token,
                         "capture_group":entry["outdoor_capture_group"],
                         "cabin_group":entry["reflection_capture_group"],
                         "source_pair_sha256":[entry["outdoor_sha256"],entry["reflection_sha256"]],
                         "preset":entry["parameter_key"],**stats})
        if len(accepted)%100==0:
            print("ACCEPTED",len(accepted),"ATTEMPT",i+1,"SECS",
                  round(time.monotonic()-started,1),flush=True)
    report={"status":"RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE",
            "fold":args.fold,"arm":"laminate",
            "measured_automotive_realism":False,
            "source_rights":"Local research only; restricted derivatives not redistributed",
            "limitations":["CABIN 2D UNREGISTERED","NOT COMMON EXPOSURE RAW",
                           "GLASS PROPERTIES HYPOTHETICAL","NO REAL WINDSHIELD GT"],
            "source_split_groups":{"T":folds_t,"R":folds_r},
            "counts":{"accepted":len(accepted),"rejected":len(rejected),
                      "unique_outdoor_photos":len({x["source_pair_sha256"][0] for x in accepted}),
                      "unique_cabin_photos":len({x["source_pair_sha256"][1] for x in accepted}),
                      "unique_outdoor_groups":len({x["capture_group"] for x in accepted}),
                      "unique_cabin_groups":len({x["cabin_group"] for x in accepted}),
                      "unique_presets":len({x["preset"] for x in accepted}),
                      "attempted":len(accepted)+len(rejected)},
            "items":accepted}
    (args.out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.out/"rejected_manifest.json").write_text(json.dumps({"items":rejected},indent=2)+"\n")
    print("CORPUS_DONE",args.out,"COUNTS",report["counts"],"SEC",
          round(time.monotonic()-started,1),flush=True)

if __name__=="__main__":
    main()
