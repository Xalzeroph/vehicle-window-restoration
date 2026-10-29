#!/usr/bin/env python3


import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from vehicle_optics import Glass, Camera, render, srgb_to_linear


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def open_img(path):
    im = Image.open(path).convert("RGB")
    return im


def align_cabin(im, shape, crop_center_x=0.5, crop_center_y=0.5):


    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("Target resolution must be positive")
    if not (0 <= crop_center_x <= 1 and 0 <= crop_center_y <= 1):
        raise ValueError("Crop center must be in [0,1]")
    im_ratio = im.width / im.height
    dst_ratio = width / height
    if im_ratio >= dst_ratio:
        tw = int(round(im.height * dst_ratio))
        x = round(crop_center_x * (im.width - tw))
        im = im.crop((x, 0, x+tw, im.height))
    else:
        th = int(round(im.width / dst_ratio))
        y = round(crop_center_y * (im.height - th))
        im = im.crop((0,y,im.width,y+th))
    return im.resize((width,height),Image.Resampling.LANCZOS)


def analytic_stats(result):
    L = np.array([0.2126,0.7152,0.0722])
    t = result["transmitted_linear"] @ L
    r = result["reflected_linear"] @ L
    vis = r/(t+r+1e-8)
    valid = result["source_coverage_mask"] & ~result["saturation_mask"]
    hist = dict(
        visibility_p10=float(np.quantile(vis[valid],.1)) if valid.any() else None,
        visibility_p50=float(np.median(vis[valid])) if valid.any() else None,
        visibility_p90=float(np.quantile(vis[valid],.9)) if valid.any() else None,
        reflection_radiance_p95=float(np.quantile(r,.95)),
        usable_fraction=float(valid.mean()),
        saturation_fraction=float(result["saturation_mask"].mean()),
        valid_ghost_fraction=float(result["source_coverage_mask"].mean()),
        defocus_sigma_px=result["defocus_sigma_px"],
        modeled_ghost_orders=result["ghost_orders"],
    )
    return hist


def make(cfg, target_file, cabin_file, out_path, train_group, cabin_group,
         relative_cabin_illuminance, seed):
    with Image.open(target_file) as im:
        t8 = np.asarray(im.convert("RGB"),dtype=np.float64)/255
    t = srgb_to_linear(t8)
    h,w=t.shape[:2]
    cabin = open_img(cabin_file)
    cabin = align_cabin(cabin, (h,w),
                        cfg["crop_center_xy"][0], cfg["crop_center_xy"][1])


    r = srgb_to_linear(np.asarray(cabin,dtype=np.float64)/255)
    r *= float(relative_cabin_illuminance)
    glass = Glass(n=cfg["glass"]["n"],
                  optical_depth_rgb=tuple(cfg["glass"]["optical_depth_rgb"]),
                  normal_xyz=tuple(cfg["glass"]["normal_xyz"]),
                  focal_px=tuple(cfg["calibration"]["focal_px"]),
                  principal_px=tuple(cfg["calibration"]["principal_px"]),
                  thickness_m=cfg["glass"]["thickness_m"],
                  camera_to_glass_m=cfg["glass"]["camera_to_glass_m"])


    tr,_,_ = __import__("vehicle_optics").coefficients(h,w,glass)
    trans = t*tr
    scene_reference=float(np.quantile(trans,.995))
    exposure=cfg["ae_reference_level"]/max(scene_reference,1e-8)
    c=Camera(exposure=exposure, wb_rgb=tuple(cfg["camera"]["wb_rgb"]),
             full_well_e=cfg["camera"]["full_well_e"],
             read_noise_e=cfg["camera"]["read_noise_e"],
             focal_length_mm=cfg["camera"]["focal_length_mm"],
             f_number=cfg["camera"]["f_number"],
             focus_distance_m=cfg["camera"]["focus_distance_m"],
             reflection_distance_m=cfg["camera"]["reflection_distance_m"],
             sensor_width_mm=cfg["camera"]["sensor_width_mm"])
    result=render(t,r,glass,c,seed)
    out_path.mkdir(parents=True,exist_ok=False)
    for key in ("observed","target_tinted","target_clear"):
        Image.fromarray(np.round(np.clip(result[key],0,1)*255).astype("uint8")).save(out_path/(key+".png"))
    np.savez_compressed(out_path/"radiance.npz",
        transmitted=result["transmitted_linear"].astype("float32"),
        reflection=result["reflected_linear"].astype("float32"),
        sensor=result["raw_sensor_linear"].astype("float32"),
        saturated=result["saturation_mask"],
        valid_ghost=result["source_coverage_mask"])
    metrics=analytic_stats(result)
    metadata={
        "dataset_role":"UNVERIFIED_RESEARCH_PREVIEW_ONLY",
        "glass_scene_model":"incoherent ideal slab, non-automotive calibrated",
        "registration":"2D cropped cabin camera; no 3D car/windshield alignment",
        "radiometry":"inverse-sRGB display RGB, exposure and cabin scale illustrative",
        "source_transmission":{"path":str(target_file),"sha256":sha(target_file),
                               "source_dataset":cfg.get("transmission_dataset","KITTI"),
                               "source_group":train_group,
                               "source_rights":cfg.get("transmission_rights","research only"),
                               "split_policy":"capture-group partition; no real benchmark overlap"},
        "source_reflection":{"path":str(cabin_file),"sha256":sha(cabin_file),
                             "source_dataset":cfg.get("reflection_dataset","Drive&Act"),
                             "source_rights":cfg.get("reflection_rights","research only"),
                             "source_group":cabin_group,
                             "capture_camera":"A-column co-driver Kinect"},
        "seed":seed,"relative_cabin_illuminance_hypothesis":relative_cabin_illuminance,
        "glass":cfg["glass"],"camera":vars(c),
        "camera_calibration":cfg["calibration"],
        "candidate_not_training_release":True,
        "reason_for_restriction":["uncalibrated color/radiance basis",
                                  "no co-registered reflection camera geometry",
                                  "Drive&Act permitted research-only; no public redistribution",
                                  "no blind real windshield domain validation"],
        "metrics":metrics,
    }
    (out_path/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--kitti-manifest",type=Path,required=True)
    parser.add_argument("--cabin-manifest",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    sources=json.loads(args.kitti_manifest.read_text())
    cabin=json.loads(args.cabin_manifest.read_text())
    cfg=json.loads(args.config.read_text())
    if not sources or not cabin or not cfg.get("relative_cabin_illuminance_hypotheses"):
        raise ValueError("Missing sources or parameter sweep")
    for p in [item["filename"] for item in sources]+[item["filename"] for item in cabin]:
        if not Path(p).exists(): raise FileNotFoundError(p)
        if any(marker in str(p).lower() for marker in
               ["test_sets","real_pair","test/","polaRGB/test"]):
            raise ValueError("Test-set leakage risk "+p)
    args.out.mkdir(parents=True,exist_ok=False)


    manifest=[]
    for i,rec in enumerate(sources):
        ref=cabin[i%len(cabin)]
        scene=rec["capture_group"]
        for j,scale in enumerate(cfg["relative_cabin_illuminance_hypotheses"]):
            key=f'{rec["sha256"]}|{ref["sha256"]}|{scale}|{cfg["seed"]}'
            stem=hashlib.sha256(key.encode()).hexdigest()[:16]
            seed=int.from_bytes(hashlib.sha256((key+"-noise").encode()).digest()[:8],"big")
            meta=make(cfg,Path(rec["filename"]),Path(ref["filename"]),
                      args.out/"samples"/stem,scene,ref["capture_group"],
                      scale,seed)
            manifest.append({"id":stem,"relative_directory":"samples/"+stem,
                             "capture_group":scene,"cabin_group":ref["capture_group"],
                             "illumination_bin":j,**meta["metrics"]})
            print("RENDERED",len(manifest),stem,"visibility",round(meta["metrics"]["visibility_p50"],3),
                  "saturation",round(meta["metrics"]["saturation_fraction"],4),flush=True)
    (args.out/"manifest.json").write_text(json.dumps({
        "status":"RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE",
        "source_counts":{"transmission_images":len(sources),"cabin_images":len(cabin)},
        "transmission_capture_groups":sorted(set(x["capture_group"] for x in sources)),
        "cabin_capture_groups":sorted(set(x["capture_group"] for x in cabin)),
        "config_sha256":sha(args.config),
        "items":manifest},indent=2)+"\n")
    print("FINISHED_CANDIDATES",len(manifest),str(args.out),flush=True)


if __name__=="__main__":
    main()
