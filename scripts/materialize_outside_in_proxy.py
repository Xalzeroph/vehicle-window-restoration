#!/usr/bin/env python3


import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
from PIL import Image
from vehicle_recipe_index import import_source,dedupe_source_images,stable_balanced_folds
from vehicle_optics import Glass,Camera,render,srgb_to_linear
from vehicle_view_contract import TASK,TRANSMISSION,REFLECTION

def rnd(seed,key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256((str(seed)+key).encode()).digest()[:8],"big"))

def image256(path, side, rng):
    with Image.open(path) as img:
        img=img.convert("RGB")
        w,h=img.size


        scale=side/min(w,h)
        width=max(side,math.ceil(w*scale))
        height=max(side,math.ceil(h*scale))
        img=img.resize((width,height),Image.Resampling.LANCZOS)
        xx=int(rng.integers(width-side+1))
        yy=int(rng.integers(height-side+1))
        return np.asarray(img.crop((xx,yy,xx+side,yy+side)),dtype=np.float64)/255

def main():
    p=argparse.ArgumentParser()
    for name in ("a2d2-manifest","kitti-manifest","cabin-manifest","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",choices=("train","val","test"),required=True)
    p.add_argument("--samples",type=int,required=True)
    p.add_argument("--side",type=int,default=256)
    p.add_argument("--seed",type=int,default=20261015)
    args=p.parse_args()
    if args.side%32 or args.side<=0 or args.samples<=0:raise ValueError("Invalid budget")
    if (args.out/"manifest.json").exists():raise FileExistsError("Do not overwrite finished dataset")
    cabin=dedupe_source_images(import_source(args.cabin_manifest,"driveact"))
    outside=dedupe_source_images(import_source(args.a2d2_manifest,"a2d2")+
                                 import_source(args.kitti_manifest,"kitti"))
    cap,cap_groups=stable_balanced_folds(cabin,args.seed)
    road,road_groups=stable_balanced_folds(outside,args.seed)
    ts=[x for x in cabin if cap[x.capture_group]==args.fold]
    rs=[x for x in outside if road[x.capture_group]==args.fold]
    if not ts or not rs:raise ValueError("Fold lacks a source class")


    t_by_group={g:[x for x in ts if x.capture_group==g] for g in sorted({x.capture_group for x in ts})}
    r_by_group={g:[x for x in rs if x.capture_group==g] for g in sorted({x.capture_group for x in rs})}
    ta=list(t_by_group);ra=list(r_by_group)
    args.out.mkdir(parents=True,exist_ok=True)
    items=[];start=time.monotonic()
    for idx in range(args.samples):
        rng=rnd(args.seed,args.fold+str(idx))
        tgroup=ta[idx%len(ta)]
        rgroup=ra[(idx//len(ta)+idx)%len(ra)]
        t= t_by_group[tgroup][int(rng.integers(len(t_by_group[tgroup])))]
        r= r_by_group[rgroup][int(rng.integers(len(r_by_group[rgroup])))]


        token=hashlib.sha256((args.fold+t.sha256+r.sha256+str(idx)+str(args.seed)).encode()).hexdigest()[:20]
        folder=args.out/"samples"/token
        if folder.exists():raise FileExistsError("Refuse overwrite "+token)
        original_t=image256(t.path,args.side,rng)
        original_r=image256(r.path,args.side,rng)
        tlin=srgb_to_linear(original_t)
        rlin=srgb_to_linear(original_r)

        ambient_ratio=2.**float(rng.uniform(-.5,3.))
        rlin*=ambient_ratio
        n=float(rng.uniform(1.49,1.55))
        optical=np.array([rng.uniform(.035,.09),rng.uniform(.055,.13),
                          rng.uniform(.07,.19)])
        width_m=float(rng.uniform(.0035,.006))
        tilt=(float(rng.uniform(-.13,.13)),float(rng.uniform(-.23,.23)),1.)
        layers=(
          {"n":n,"thickness_m":width_m*.42,"optical_depth_rgb":(optical*.48).tolist()},
          {"n":1.48,"thickness_m":width_m*.16,"optical_depth_rgb":(optical*.04).tolist()},
          {"n":n,"thickness_m":width_m*.42,"optical_depth_rgb":(optical*.48).tolist()})
        glass=Glass(n=n,optical_depth_rgb=tuple(optical),
            normal_xyz=tilt,
            focal_px=(float(args.side*1.2),float(args.side*1.2)),
            principal_px=(args.side/2,args.side/2),
            camera_to_glass_m=float(rng.uniform(.6,1.8)),
            thickness_m=width_m,layers=layers)
        exposure=float(rng.uniform(.6,1.25))
        cam=Camera(exposure=exposure,wb_rgb=tuple(rng.uniform(.93,1.07,size=3)),
            full_well_e=18000,read_noise_e=3.,
            focal_length_mm=35.,f_number=float(rng.uniform(2.0,8.)),
            focus_distance_m=float(rng.uniform(3.,15.)),
            reflection_distance_m=float(rng.uniform(1.5,30.)),
            sensor_width_mm=36.,sensor_native_width_px=args.side)
        result=render(tlin,rlin,glass,cam,int(rng.integers(2**31)))
        folder.mkdir(parents=True)
        for name in ("observed","target_tinted","target_clear"):
            Image.fromarray(np.uint8(np.round(result[name]*255))).save(folder/(name+".png"))
        np.savez_compressed(folder/"radiance.npz",
            transmitted=result["transmitted_linear"].astype("float32"),
            reflection=result["reflected_linear"].astype("float32"),
            sensor=result["raw_sensor_linear"].astype("float32"),
            saturated=result["saturation_mask"],
            valid_ghost=result["source_coverage_mask"])
        lum=np.array([.2126,.7152,.0722])
        tl=result["transmitted_linear"]@lum;rl=result["reflected_linear"]@lum
        valid=result["source_coverage_mask"]&~result["saturation_mask"]
        vis=rl/(tl+rl+1e-8)
        tmeta={"source_group":tgroup,"sha256":t.sha256,
               "scene_role":TRANSMISSION,"capture_geometry":"illustrative_unregistered_proxy",
               "source_dataset":t.dataset,"source_path":t.path}
        rmeta={"source_group":rgroup,"sha256":r.sha256,
               "scene_role":REFLECTION,"capture_geometry":"illustrative_unregistered_proxy",
               "source_dataset":r.dataset,"source_path":r.path}
        contract={"task":TASK,"camera_side":"exterior","calibration_status":"illustrative",
                  "transmission":{"scene_role":TRANSMISSION,"capture_geometry":"illustrative_unregistered_proxy"},
                  "reflection":{"scene_role":REFLECTION,"capture_geometry":"illustrative_unregistered_proxy"}}
        meta={"capture_contract":contract,"source_transmission":tmeta,
              "source_reflection":rmeta,"camera":vars(cam),
              "optical_model":result["optical_model_diagnostics"],
              "candidate_not_training_release":True,
              "origin_side":"proxy cabin camera, NOT actual exterior through-glass capture",
              "relative_exterior_illumination_proxy":ambient_ratio,
              "source_rights":"LOCAL research; KITTI/A2D2/Drive&Act derivative terms govern",
              "physical_camera_intrinsics":"illustrative uncalibrated",
              "reflection_visibility_median":float(np.median(vis[valid])) if valid.any() else 0.}
        (folder/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
        items.append({"id":token,"relative_directory":"samples/"+token,
                      "capture_group":tgroup,"cabin_group":rgroup,
                      "inside_original_sha":t.sha256,"outside_original_sha":r.sha256,
                      "reflection_visibility_median":meta["reflection_visibility_median"],
                      "valid_fraction":float(valid.mean()),
                      "saturation_fraction":float(result["saturation_mask"].mean())})
        if (idx+1)%32==0:
            print("OUTSIDE_IN_CREATED",args.fold,idx+1,"SECS",round(time.monotonic()-start,1),flush=True)
    report={"status":"RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE",
            "task":TASK,"fold":args.fold,"source_is_exterior_view_calibrated":False,
            "limitations":["Cabin cameras inside vehicle are perspective-inaccurate proxies",
                "Outdoor reflection geometry not measured against exterior camera",
                "Inverse sRGB RGB sources are not common radiance RAW"],
            "source_group_splits":{"cabin":cap_groups,"outdoors":road_groups},
            "original_interior_count":len({x["inside_original_sha"] for x in items}),
            "original_environment_count":len({x["outside_original_sha"] for x in items}),
            "items":items}
    (args.out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print("OUTSIDE_IN_COMPLETE",args.fold,len(items),"INTERIOR",report["original_interior_count"],
          "EXTERIOR",report["original_environment_count"],"SECONDS",round(time.monotonic()-start,1),flush=True)

if __name__=="__main__":
    main()
