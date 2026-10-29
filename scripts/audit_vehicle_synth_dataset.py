#!/usr/bin/env python3


import argparse
import json
import math
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from vehicle_optics import linear_to_srgb


def load_png(path):
    with Image.open(path) as im:
        if im.format != "PNG" or im.mode != "RGB":
            raise ValueError("Expected full-resolution lossless RGB PNG "+str(path))
        return np.asarray(im,dtype=np.float64)/255.


def quantiles(data):
    x=np.asarray(data,dtype=np.float64)
    if not len(x): return None
    return {"min":float(np.min(x)),"p10":float(np.quantile(x,.1)),
            "p50":float(np.median(x)),"p90":float(np.quantile(x,.9)),
            "max":float(np.max(x))}


def edge_variance(im):
    lum=im@np.array([.2126,.7152,.0722])

    lap=(-4*lum[1:-1,1:-1]+lum[1:-1,:-2]+lum[1:-1,2:]
         +lum[:-2,1:-1]+lum[2:,1:-1])
    return float(np.var(lap))


def sample_metrics(item,root):
    path=root/item["relative_directory"]
    meta=json.loads((path/"metadata.json").read_text())
    with np.load(path/"radiance.npz",allow_pickle=False) as arrays:
        t=np.array(arrays["transmitted"],dtype=np.float64)
        r=np.array(arrays["reflection"],dtype=np.float64)
        sensor=np.array(arrays["sensor"],dtype=np.float64)
        saturated=np.array(arrays["saturated"])
        coverage=np.array(arrays["valid_ghost"])
    o=load_png(path/"observed.png")
    tinted=load_png(path/"target_tinted.png")
    clear=load_png(path/"target_clear.png")
    if t.shape != o.shape or r.shape != o.shape or sensor.shape != o.shape:
        raise ValueError("Image/linear layers mismatch")
    if not all(np.all(np.isfinite(x)) and np.all(x>=0) for x in (t,r,sensor)):
        raise ValueError("Non-finite/negative radiances")
    wb=np.asarray(meta["camera"]["wb_rgb"],dtype=np.float64)
    exposure=float(meta["camera"]["exposure"])
    reconstructed_observed=linear_to_srgb(sensor*wb)
    reconstructed_tinted=linear_to_srgb(t*exposure*wb)
    max_delta_observed=float(np.max(np.abs(o-reconstructed_observed)))
    max_delta_tinted=float(np.max(np.abs(tinted-reconstructed_tinted)))


    max_png_quant_error=0.5/255+1e-5
    if max_delta_observed>max_png_quant_error or max_delta_tinted>max_png_quant_error:
        raise ValueError("Quantization/linearity mismatch: "+str(path))
    L=np.array([.2126,.7152,.0722])
    tl=t@L;rl=r@L
    visible=rl/(tl+rl+1e-8)
    valid=~saturated & coverage
    if not np.array_equal(np.array(meta["metrics"]["saturation_fraction"]),
                          np.array(saturated.mean())):
        raise ValueError("Saturation mask/provenance mismatch")
    if not np.array_equal(np.array(meta["metrics"]["valid_ghost_fraction"]),
                          np.array(coverage.mean())):
        raise ValueError("Geometry mask/provenance mismatch")
    rmse=math.sqrt(float(np.mean((o-tinted)**2)))
    psnr= -20*math.log10(rmse) if rmse>0 else float("inf")
    return {
        "id":item["id"],"capture_group":item["capture_group"],
        "cabin_group":item["cabin_group"],
        "illumination_bin":item["illumination_bin"],
        "width":o.shape[1],"height":o.shape[0],
        "observed_png_consistency_max_error":max_delta_observed,
        "target_tinted_png_consistency_max_error":max_delta_tinted,
        "observed_relative_luminance_mean":float(np.mean(o@L)),
        "tinted_relative_luminance_mean":float(np.mean(tinted@L)),
        "raw_sensor_saturation_fraction":float(saturated.mean()),
        "invalid_ghost_fraction":float(1-coverage.mean()),
        "reflection_visibility_p50":float(np.median(visible[valid])) if valid.any() else None,
        "reflection_visibility_p90":float(np.quantile(visible[valid],.9)) if valid.any() else None,
        "image_delta_rmse":rmse,
        "observed_vs_tinted_psnr_difficulty_not_accuracy":psnr,
        "edge_variance_observed":edge_variance(o),
        "edge_variance_tinted":edge_variance(tinted),
        "target_clear_vs_tinted_mean_abs_difference":float(np.abs(clear-tinted).mean()),
        "reflected_energy_fraction":float(rl.sum()/(rl.sum()+tl.sum()+1e-8)),
        "source_T_sha256":meta["source_transmission"]["sha256"],
        "source_R_sha256":meta["source_reflection"]["sha256"],
        "scientific_release_eligible":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batch",required=True,type=Path)
    p.add_argument("--manifest",type=Path,
                   help="Optional audited subset manifest; default full manifest.json")
    p.add_argument("--out",type=Path,help="Optional independent audit output filename")
    a=p.parse_args()
    manifest_path = a.manifest if a.manifest else a.batch/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    if manifest["status"]!="RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE":
        raise ValueError("Invalid phase")
    by_id=set();rows=[]
    for item in manifest["items"]:
        if item["id"] in by_id:raise ValueError("Duplicate sample ID")
        by_id.add(item["id"])
        rows.append(sample_metrics(item,a.batch))
    summaries={}
    for metric in ("observed_png_consistency_max_error","target_tinted_png_consistency_max_error",
                   "reflection_visibility_p50","reflection_visibility_p90",
                   "raw_sensor_saturation_fraction","invalid_ghost_fraction",
                   "reflected_energy_fraction","image_delta_rmse",
                   "observed_vs_tinted_psnr_difficulty_not_accuracy",
                   "edge_variance_observed","edge_variance_tinted"):
        summaries[metric]=quantiles([x[metric] for x in rows if x[metric] is not None])
    by_condition={}
    for binid in sorted(set(x["illumination_bin"] for x in rows)):
        arr=[x for x in rows if x["illumination_bin"]==binid]
        by_condition[str(binid)]={
            "count":len(arr),
            "visibility":quantiles([x["reflection_visibility_p50"] for x in arr]),
            "saturation":quantiles([x["raw_sensor_saturation_fraction"] for x in arr])}
    flags={
        "failed_numerical_layer_consistency":0,
        "source_scene_groups":len(set(x["capture_group"] for x in rows)),
        "reflection_video_capture_groups":len(set(x["cabin_group"] for x in rows)),
        "source_unique_images":len(set(x["source_T_sha256"] for x in rows)),
        "reflection_unique_images":len(set(x["source_R_sha256"] for x in rows)),
        "no_real_windshield_calibration":True,
        "no_registered_cabin_environment_or_camera_pose":True,
        "not_authorized_for_redistribution_or_as_high_quality_training_data":True,
        "no_real_domain_downstream_validation":True,
    }
    report={"scientific_status":"UNVERIFIED RESEARCH CANDIDATES, NOT HIGH-QUALITY VERIFIED",
            "sample_count":len(rows),"dimensions":sorted(set((x["width"],x["height"]) for x in rows)),
            "grouping":flags,"metrics":summaries,"illumination_strata":by_condition,
            "metric_interpretation":{
                "PSNR":"a measure of how much synthesized mixture differs from clean target, not model PSNR or perceptual-realism score",
                "png_consistency":"asserts save paths and linear-layer GT numerical agreement, not true glass physics",
                "visibility":"relative modeled reflected energy; unknown real scene photometry implies not calibrated",
                "edge_variance":"spatial image detail diagnostic only; meaningful quality requires real reference"},
            "items":rows}
    out=a.out if a.out else a.batch/"quality_audit.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print("AUDIT_OK",len(rows),"images",flush=True)
    print("NUMERICAL_ERROR",summaries["observed_png_consistency_max_error"],flush=True)
    print("VISIBILITY",summaries["reflection_visibility_p50"],flush=True)
    print("SATURATION",summaries["raw_sensor_saturation_fraction"],flush=True)
    print("SOURCES",flags,flush=True)
    for k,v in by_condition.items():
        print("ILLUMINATION_BIN",k,v,flush=True)
    print("RELEASE_STATUS",report["scientific_status"],flush=True)


if __name__=="__main__":
    main()
