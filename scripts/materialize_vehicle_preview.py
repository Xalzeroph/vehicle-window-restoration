#!/usr/bin/env python3


import argparse,copy,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from vehicle_recipe_index import import_source,stable_balanced_folds
from build_vehicle_synth_dataset import make


def choose_spaced(items,n):
    if n<=0: return []
    items=sorted(items,key=lambda x:x.sha256)
    k=min(n,len(items))
    if k==1:return items[:1]
    return [items[(len(items)-1)*i//(k-1)] for i in range(k)]


def native_calibration(t,a2d2,kitti):
    if t.dataset == "Audi A2D2":
        cams=a2d2["cameras"]["front_center"]
        k=cams["CamMatrix"]
        return {
            "source":"Audi A2D2 calibrated telecamera JSON; PNG distortion validity unverified",
            "focal_px":[k[0][0],k[1][1]],
            "principal_px":[k[0][2],k[1][2]],
            "camera_model":"telecam, original lens distortion semantics UNVERIFIED"
        }
    date=t.capture_group.split("/")[-1][:10]
    if date not in kitti:
        raise ValueError("Missing official source camera intrinsics "+date)
    return kitti[date]


def config_for(t,preset,a2d2,kitti,root):
    config={
        "transmission_dataset":t.dataset,
        "transmission_rights":t.rights,
        "reflection_dataset":"Drive&Act ICCV 2019",
        "reflection_rights":"Author research-only conditions, no redistribution",
        "seed":int(root["seed"]),
        "crop_center_xy":[.5,.5],
        "glass":{
            "n":preset["glass"]["n"],
            "optical_depth_rgb":preset["glass"]["optical_depth_rgb"],
            "normal_xyz":[.04,.12,1.],
            "thickness_m":preset["glass"]["thickness_m"],
            "camera_to_glass_m":.6},
        "calibration":native_calibration(t,a2d2,kitti),
        "camera":{
            "wb_rgb":[1.05,1.,.98],
            "full_well_e":18000,"read_noise_e":3.,
            "focal_length_mm":35.,
            "f_number":preset["camera_f_number"],
            "focus_distance_m":40.,
            "reflection_distance_m":.8,
            "sensor_width_mm":36.},
        "ae_reference_level":.88,
        "selected_preset":preset["key"],
        "relative_cabin_illuminance_hypotheses":[]
    }
    return config


def build(a):
    sources=[]
    if a.a2d2_manifest:
        sources+=import_source(a.a2d2_manifest,"a2d2")
    if a.kitti_manifest:
        sources+=import_source(a.kitti_manifest,"kitti")
    cabin=import_source(a.cabin_manifest,"driveact")
    tassign,tfolds=stable_balanced_folds(sources,a.seed)
    rassign,rfolds=stable_balanced_folds(cabin,a.seed)
    tgroup=defaultdict(list);rgroup=defaultdict(list)
    for t in sources:
        if tassign[t.capture_group]==a.fold:
            tgroup[t.capture_group].append(t)
    for r in cabin:
        if rassign[r.capture_group]==a.fold:
            rgroup[r.capture_group].append(r)
    if not rgroup:
        raise ValueError("No verified cabin actor in selected fold")
    t_selected=[]
    for group,items in sorted(tgroup.items()):
        t_selected.extend(choose_spaced(items,a.outdoor_per_capture))
    r_selected=[]
    for group,items in sorted(rgroup.items()):
        r_selected.extend(choose_spaced(items,a.cabin_per_capture))
    if not t_selected or not r_selected:
        raise ValueError("No usable valid train source groups")
    cabin_group_order=sorted(rgroup)
    cabin_by_group={g:choose_spaced(rows,a.cabin_per_capture) for g,rows in rgroup.items()}
    presets=json.loads(a.presets.read_text())
    a2d2=json.loads(a.a2d2_calibration.read_text())
    kitti=json.loads(a.kitti_calibration.read_text())
    a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,t in enumerate(t_selected):


        rgroup_name=cabin_group_order[i%len(cabin_group_order)]
        group_frames=cabin_by_group[rgroup_name]
        r=group_frames[(i//len(cabin_group_order))%len(group_frames)]
        preset=presets[i%len(presets)]
        cfg=config_for(t,preset,a2d2,kitti,{"seed":a.seed})
        scale=float(2**preset["relative_cabin_radiance_ev_HYPOTHESIS"])
        key="|".join([t.sha256,r.sha256,preset["key"],str(a.seed)])
        token=hashlib.sha256(key.encode()).hexdigest()[:24]
        seed=int.from_bytes(hashlib.sha256((key+"-sensor").encode()).digest()[:8],"big")
        output=a.out/"samples"/token
        try:
            if output.exists():
                meta_file=output/"metadata.json"
                if not meta_file.is_file():
                    raise RuntimeError("Unfinished output requires explicit inspection; refusing overwrite "+str(output))
                meta=json.loads(meta_file.read_text())
                if (meta["source_transmission"]["sha256"]!=t.sha256
                        or meta["source_reflection"]["sha256"]!=r.sha256
                        or meta["seed"]!=seed
                        or meta["glass"]!=cfg["glass"]):
                    raise RuntimeError("Existing output provenance mismatch "+str(output))
            else:
                meta=make(cfg,Path(t.path),Path(r.path),output,
                          t.capture_group,r.capture_group,scale,seed)
            rows.append({"id":token,"relative_directory":"samples/"+token,
                         "capture_group":t.capture_group,
                         "cabin_group":r.capture_group,
                         "illumination_bin":i%3,
                         "source_camera_calibration":cfg["calibration"].get(
                             "source",cfg["calibration"].get("official_zip")),
                         "preset":preset["key"],**meta["metrics"]})
            print("MATERIALIZED",len(rows),"/",len(t_selected),token,
                  t.dataset,"t_group",t.capture_group,
                  "r_group",r.capture_group,
                  "native_calibration",cfg["calibration"].get("source",cfg["calibration"].get("official_zip")),
                  "visibility",round(meta["metrics"]["visibility_p50"],4),
                  flush=True)
        except Exception as e:
            print("FAILED_RECIPE",token,str(e),flush=True)
            raise
    (a.out/"manifest.json").write_text(json.dumps({
        "status":"RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE",
        "source_policy":f"{a.fold} only, capture-disjoint from other source groups",
        "geometry":"UNCALIBRATED 2D cabin source; NOT high-quality vehicle-realistic",
        "independent_outdoor_groups":len({x["capture_group"] for x in rows}),
        "independent_cabin_groups":len({x["cabin_group"] for x in rows}),
        "materialized_images":len(rows),
        "source_counts":{"transmission_images":len(t_selected),
                         "cabin_images":len(r_selected)},
        "items":rows},indent=2)+"\n")
    print("COMPLETED",len(rows),"ACTUAL_NATIVE_RES_PREVIEWS",a.out,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a2d2-manifest",type=Path,required=True)
    p.add_argument("--kitti-manifest",type=Path,required=True)
    p.add_argument("--cabin-manifest",type=Path,required=True)
    p.add_argument("--a2d2-calibration",type=Path,required=True)
    p.add_argument("--kitti-calibration",type=Path,required=True)
    p.add_argument("--presets",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--outdoor-per-capture",type=int,default=3)
    p.add_argument("--cabin-per-capture",type=int,default=2)
    p.add_argument("--seed",type=int,default=20261015)
    p.add_argument("--fold",choices=("train","val","test"),default="train")
    build(p.parse_args())


if __name__=="__main__":
    main()
