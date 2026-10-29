#!/usr/bin/env python3


import argparse,hashlib,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
from build_vehicle_synth_dataset import align_cabin,sha
from vehicle_optics import Glass,Camera,render,srgb_to_linear
from vehicle_crop_sampling import choose_content_patch
from vehicle_optical_framing import required_source_halo,extend_crop_in_source
from vehicle_view_contract import validate_sample_metadata


def mixture_pair_from_source_meta(source_meta, output, arm, crop_side,
                                  parent_id, parent_manifest):

    validate_sample_metadata(source_meta,permit_proxy=True)
    t_src=Path(source_meta["source_transmission"]["path"])
    r_src=Path(source_meta["source_reflection"]["path"])
    if not t_src.is_file() or not r_src.is_file():
        raise FileNotFoundError("Missing actual T-interior or R-exterior capture")
    if any(marker in str(t_src).lower() for marker in ("real_pair","test_sets","/test/")):
        raise ValueError("Held-out test overlap")
    t_img=Image.open(t_src).convert("RGB")
    h,w=t_img.height,t_img.width
    if min(h,w)<crop_side:raise ValueError("Capture cannot fit research crop")
    key=source_meta["source_transmission"]["sha256"]+"|"+source_meta["source_reflection"]["sha256"]
    calibr=source_meta["camera_calibration"]
    x0,y0,crop_diagnostics=choose_content_patch(
        t_img,crop_side,calibr["principal_px"],key)
    reflection_full=align_cabin(Image.open(r_src).convert("RGB"),(h,w))
    t=np.asarray(t_img.crop((x0,y0,x0+crop_side,y0+crop_side)),dtype=np.float64)/255
    r=np.asarray(reflection_full.crop((x0,y0,x0+crop_side,y0+crop_side)),dtype=np.float64)/255
    T=srgb_to_linear(t)
    R=srgb_to_linear(r)*float(source_meta["relative_exterior_reflection_illuminance_hypothesis"])
    camera=Camera(**source_meta["camera"])
    if camera.sensor_native_width_px == 0:
        camera=replace(camera,sensor_native_width_px=w)
    cfg=source_meta["glass"]
    focal=calibr["focal_px"]
    principal=calibr["principal_px"]
    layers=()
    if arm=="laminate":
        optical_depth=np.asarray(cfg["optical_depth_rgb"],dtype=float)
        thickness=float(cfg["thickness_m"])
        layers=(
            {"n":float(cfg["n"]),"thickness_m":.42*thickness,
             "optical_depth_rgb":(.48*optical_depth).tolist()},
            {"n":1.48,"thickness_m":.16*thickness,
             "optical_depth_rgb":(.04*optical_depth).tolist()},
            {"n":float(cfg["n"]),"thickness_m":.42*thickness,
             "optical_depth_rgb":(.48*optical_depth).tolist()},
        )
    glass=Glass(n=float(cfg["n"]),
        optical_depth_rgb=tuple(cfg["optical_depth_rgb"]),
        normal_xyz=tuple(cfg["normal_xyz"]),
        focal_px=tuple(focal),
        principal_px=(float(principal[0]-x0),float(principal[1]-y0)),
        thickness_m=float(cfg["thickness_m"]),
        camera_to_glass_m=float(cfg["camera_to_glass_m"]),
        layers=layers)


    halo,halo_diagnostics=required_source_halo(
        (crop_side,crop_side),glass,camera,
        max(float(np.max(T)),float(np.max(R))))
    region,local=extend_crop_in_source((w,h),(x0,y0,crop_side),halo)
    left,top,right,bottom=region
    t_window=np.asarray(t_img.crop(region),dtype=np.float64)/255
    r_window=np.asarray(reflection_full.crop(region),dtype=np.float64)/255
    T_window=srgb_to_linear(t_window)
    R_window=srgb_to_linear(r_window)*float(
        source_meta["relative_exterior_reflection_illuminance_hypothesis"])
    glass_window=replace(glass,principal_px=(
        float(principal[0]-left),float(principal[1]-top)))
    optical=render(T_window,R_window,glass_window,camera,
                   int(source_meta["seed"]))
    u0,v0,u1,v1=local
    for field in ("observed","target_tinted","target_clear",
                  "transmitted_linear","reflected_linear",
                  "raw_sensor_linear","saturation_mask","source_coverage_mask"):
        optical[field]=optical[field][v0:v1,u0:u1].copy()
    if output.exists():
        raise FileExistsError("Refuse silent overwrite "+str(output))
    output.mkdir(parents=True)
    for key in ("observed","target_tinted","target_clear"):
        Image.fromarray(np.uint8(np.round(optical[key]*255))).save(output/(key+".png"))
    np.savez_compressed(output/"radiance.npz",
        transmitted=optical["transmitted_linear"].astype("float32"),
        reflection=optical["reflected_linear"].astype("float32"),
        sensor=optical["raw_sensor_linear"].astype("float32"),
        saturated=optical["saturation_mask"],
        valid_ghost=optical["source_coverage_mask"])
    metadata=dict(source_meta)
    metadata["camera"]=vars(camera)
    metadata["physical_ablation"]={
        "arm":arm,"source_original_manifest":parent_manifest,
        "source_original_sample_id":parent_id,
        "camera_crop_xywh":[x0,y0,crop_side,crop_side],
        "crop_quality_heuristic":crop_diagnostics,
        "optical_halo_px":halo,
        "optical_halo_diagnostics":halo_diagnostics,
        "original_source_window_xyxy":list(region),
        "only_optical_stack_differs":True,
        "glass_layers":list(layers),
        "optical_diagnostics":optical["optical_model_diagnostics"],
        "source_radiometry":"8-bit inverse sRGB proxy, NOT measured RAW",
        "geometry":"uncalibrated exterior scene reflection projection, NOT registered",
        "quality_status":"RESEARCH-ONLY ablation; not visually/physically calibrated",
    }
    metadata["renderer_source_sha256"]={
        module:sha(Path(__file__).with_name(module))
        for module in ("vehicle_optics.py","vehicle_multilayer.py",
                       "build_vehicle_matched_ablation.py",
                       "vehicle_crop_sampling.py")
    }
    metadata["candidate_not_training_release"]=True
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    luma=np.array([.2126,.7152,.0722])
    lumt=optical["transmitted_linear"]@luma
    lumr=optical["reflected_linear"]@luma
    visibility=lumr/(lumt+lumr+1e-8)
    usable=optical["source_coverage_mask"] & ~optical["saturation_mask"]
    if not np.any(usable):
        raise ValueError("No valid optical supervision")
    return {
        "source_coverage_fraction":float(optical["source_coverage_mask"].mean()),
        "saturation_fraction":float(optical["saturation_mask"].mean()),
        "usable_fraction":float(usable.mean()),
        "reflection_visibility_p50":float(np.median(visibility[usable])),
        "reflection_visibility_p90":float(np.quantile(visibility[usable],.9)),
        "gradient_content_score":float(crop_diagnostics["gradient_score"]),
        "tinted_source_sha256":sha(t_src),
        "reflection_source_sha256":sha(r_src),
    }


def mixture_pair(info, output, arm, crop_side):
    source_dir=Path(info["relative_directory"])
    source_meta=json.loads((source_dir/"metadata.json").read_text())
    return mixture_pair_from_source_meta(
        source_meta,output,arm,crop_side,
        parent_id=info["id"],parent_manifest=info["original_manifest"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",required=True,type=Path)
    parser.add_argument("--validation",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--side",type=int,default=192)
    parser.add_argument("--max-train",type=int,default=32)
    parser.add_argument("--max-validation",type=int,default=4)
    args=parser.parse_args()
    if args.side<=0 or args.side%32:raise ValueError("side must be a positive multiple of 32")
    args.out.mkdir(parents=True,exist_ok=True)
    for fold,origin,max_items in (
        ("train",args.train,args.max_train),
        ("validation",args.validation,args.max_validation)):
        data=json.loads(origin.read_text())
        if data["status"]!="RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE":
            raise ValueError("Unexpected scientifically unverified input status")
        candidates=data["items"][:max_items]
        if not candidates:raise ValueError("No source candidates")
        print("FOLD",fold,"SOURCES",len(candidates),flush=True)
        for arm in ("slab","laminate"):
            rows=[]
            for index,source in enumerate(candidates,1):
                token=source["id"]
                folder=args.out/fold/arm/"samples"/token
                record=dict(source)
                record["original_manifest"]=str(origin)
                record["relative_directory"]=str((origin.parent/source["relative_directory"]).resolve())
                if folder.exists():
                    meta_file=folder/"metadata.json"
                    if not all((folder/name).is_file() for name in (
                            "metadata.json","observed.png","target_tinted.png","radiance.npz")):
                        raise ValueError("Incomplete output, refusing silent overwrite: "+str(folder))
                    previous=json.loads(meta_file.read_text())
                    original=json.loads((Path(record["relative_directory"])/"metadata.json").read_text())
                    if (previous["physical_ablation"]["arm"]!=arm
                            or previous["physical_ablation"]["source_original_sample_id"]!=token
                            or previous["source_transmission"]["sha256"]!=original["source_transmission"]["sha256"]
                            or previous["source_reflection"]["sha256"]!=original["source_reflection"]["sha256"]):
                        raise ValueError("Existing output has incompatible provenance "+str(folder))
                    with np.load(folder/"radiance.npz",allow_pickle=False) as layers:
                        coverage=float(layers["valid_ghost"].mean())
                        saturation=float(layers["saturated"].mean())
                    diagnostics={"source_coverage_fraction":coverage,
                        "saturation_fraction":saturation,
                        "tinted_source_sha256":original["source_transmission"]["sha256"],
                        "reflection_source_sha256":original["source_reflection"]["sha256"]}
                else:
                    diagnostics=mixture_pair(record,folder,arm,args.side)
                rows.append({"id":token,"relative_directory":"samples/"+token,
                             "capture_group":source["capture_group"],
                             "cabin_group":source["cabin_group"],
                             "arm":arm,**diagnostics})
                print("PAIR",fold,arm,index,"/",len(candidates),
                      "valid",round(diagnostics["source_coverage_fraction"],4),flush=True)
            (args.out/fold/arm/"manifest.json").write_text(json.dumps({
                "status":"RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE",
                "fold":fold,
                "optical_ablation_arm":arm,
                "calibration_certified":False,
                "photometric_basis":"uncalibrated sRGB inverse; not RAW",
                "geometry_certified":False,
                "rights":"Local research only, original source licenses control derivatives",
                "items":rows},indent=2)+"\n")
    print("MATCHED_ABLATION_CREATED",str(args.out),flush=True)


if __name__=="__main__":main()
