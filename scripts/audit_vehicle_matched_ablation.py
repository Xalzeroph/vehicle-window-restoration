#!/usr/bin/env python3


import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from vehicle_optics import linear_to_srgb


def scan_arm(root,fold,arm):
    directory=root/fold/arm
    m=json.loads((directory/"manifest.json").read_text())
    groups_t=set();groups_r=set();hash_t=set();hash_r=set()
    rows={}
    for rec in m["items"]:
        sample=directory/rec["relative_directory"]
        meta=json.loads((sample/"metadata.json").read_text())
        with np.load(sample/"radiance.npz",allow_pickle=False) as d:
            T=d["transmitted"].astype("float64")
            R=d["reflection"].astype("float64")
            sensor=d["sensor"].astype("float64")
            saturated=d["saturated"].astype(bool)
            valid=d["valid_ghost"].astype(bool)
        img=np.asarray(Image.open(sample/"observed.png").convert("RGB"),dtype=np.float64)/255
        tinted=np.asarray(Image.open(sample/"target_tinted.png").convert("RGB"),dtype=np.float64)/255
        wb=np.asarray(meta["camera"]["wb_rgb"])[None,None,:]
        e=float(meta["camera"]["exposure"])
        consistency_obs=float(np.max(abs(img-linear_to_srgb(sensor*wb))))
        consistency_gt=float(np.max(abs(tinted-linear_to_srgb(T*e*wb))))
        if max(consistency_obs,consistency_gt) > .5/255+1e-5:
            raise ValueError("RAW/PNG inconsistency "+rec["id"])
        if any(not np.all(np.isfinite(array)) or np.any(array<0) for array in (T,R,sensor)):
            raise ValueError("Nonfinite optical radiance "+rec["id"])
        if img.shape != T.shape or not np.any(valid&~saturated):
            raise ValueError("Invalid optical supervision "+rec["id"])
        Tmeta=meta["source_transmission"]
        Rmeta=meta["source_reflection"]
        if rec["capture_group"]!=Tmeta["source_group"] or rec["cabin_group"]!=Rmeta["source_group"]:
            raise ValueError("Provenance-group mismatch "+rec["id"])
        groups_t.add(rec["capture_group"]);groups_r.add(rec["cabin_group"])
        hash_t.add(Tmeta["sha256"]);hash_r.add(Rmeta["sha256"])
        mask=valid&~saturated
        lum=np.array([.2126,.7152,.0722])
        ratio=(R@lum)/((T+R)@lum+1e-9)
        records=dict(saturation=float(saturated.mean()),
             valid=float(mask.mean()),
             photometric_display_consistency=max(consistency_obs,consistency_gt),
             median_reflection_fraction=float(np.median(ratio[mask])),
             input_psnr=float(-10*np.log10(max(np.mean((img[mask]-tinted[mask])**2),1e-12))),
             optical_residual_bound=float(
                meta["physical_ablation"]["optical_diagnostics"]["optical_residual_energy_bound"]))
        rows[rec["id"]]={
            "optical":records,
            "source_pair":[Tmeta["sha256"],Rmeta["sha256"]],
            "camera_crop":meta["physical_ablation"]["camera_crop_xywh"],
            "seed":meta["seed"],
            "arm":meta["physical_ablation"]["arm"],
        }
    return {"rows":rows,"outdoor_groups":groups_t,"cabin_groups":groups_r,
            "outdoor_hashes":hash_t,"cabin_hashes":hash_r}


def quantile(rows,key):
    x=[r["optical"][key] for r in rows.values()]
    return {"mean":float(np.mean(x)),"median":float(np.median(x)),
            "p90":float(np.quantile(x,.9)),"max":float(np.max(x))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    args=p.parse_args()
    scans={(fold,arm):scan_arm(args.root,fold,arm)
           for fold in ("train","validation") for arm in ("slab","laminate")}
    summary={}
    for fold in ("train","validation"):
        a=scans[(fold,"slab")]
        b=scans[(fold,"laminate")]
        if a["rows"].keys()!=b["rows"].keys():
            raise AssertionError("A/B arms have different sample IDs")
        for sid in a["rows"]:
            for key in ("source_pair","camera_crop","seed"):
                if a["rows"][sid][key]!=b["rows"][sid][key]:
                    raise AssertionError("Unmatched ablation source or camera crop")
        summary[fold]={}
        for arm in ("slab","laminate"):
            x=scans[(fold,arm)]["rows"]
            summary[fold][arm]={
                "sample_count":len(x),
                "valid_fraction":quantile(x,"valid"),
                "sensor_saturation":quantile(x,"saturation"),
                "reflection_fraction":quantile(x,"median_reflection_fraction"),
                "input_vs_tinted_psnr_db":quantile(x,"input_psnr"),
                "png_reconstruction_error":quantile(x,"photometric_display_consistency"),
                "residual_optical_power_bound":quantile(x,"optical_residual_bound"),
            }
    a=scans[("train","slab")];b=scans[("validation","slab")]
    for field in ("outdoor_groups","cabin_groups","outdoor_hashes","cabin_hashes"):
        intersect=a[field]&b[field]
        if intersect:
            raise ValueError(f"Source group leakage across train/validation {field}:{intersect}")
    report={"status":"NUMERICAL/PROVENANCE VALID ONLY; AUTOMOTIVE REALISM NOT ESTABLISHED",
            "matched_pair_sources":True,
            "no_train_validation_capture_or_source_overlap":True,
            "unique_train_driving_groups":len(a["outdoor_groups"]),
            "unique_train_cabin_groups":len(a["cabin_groups"]),
            "unique_val_driving_groups":len(b["outdoor_groups"]),
            "unique_val_cabin_groups":len(b["cabin_groups"]),
            "statistics":summary,
            "limitations":["No verified RAW radiance or common exposure",
                           "Synthetic cabin reflection unregistered 2D crop",
                           "Glass multilayer parameter values are illustrative",
                           "No held-out real windshield GT, no real-world metric",
                           "Content crop might omit roadway/vehicle, not semantic-certified",
                           "Rights prohibit public redistribution of unlicensed derivatives"]}
    path=args.root/"matched_optical_quality_audit.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print("MATCHED_OPTICS_AUDIT_OK",json.dumps({
        "train_n":summary["train"]["slab"]["sample_count"],
        "val_n":summary["validation"]["slab"]["sample_count"],
        "validation_laminate":summary["validation"]["laminate"]},indent=2),flush=True)


if __name__=="__main__":main()
