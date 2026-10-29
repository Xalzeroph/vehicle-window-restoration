#!/usr/bin/env python3


import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from build_vehicle_synth_dataset import align_cabin
from vehicle_optics import Glass,Camera,render,srgb_to_linear


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--max-views",type=int,default=8)
    args=ap.parse_args()
    data=json.loads(args.manifest.read_text())
    samples=data["items"]
    seen=set();selected=[]
    for entry in samples:
        g=entry["capture_group"]
        if not g.startswith("KITTI/") or g in seen:continue
        seen.add(g);selected.append(entry)
        if len(selected)>=args.max_views:break
    args.out.mkdir(parents=True,exist_ok=True)
    measures=[]
    for i,row in enumerate(selected,1):
        source=args.manifest.parent/row["relative_directory"]
        m=json.loads((source/"metadata.json").read_text())
        out=args.out/row["id"]
        if out.exists():
            if not (out/"metadata.json").exists():raise ValueError("Incomplete previous run")
            measures.append(json.loads((out/"metadata.json").read_text()));continue
        with Image.open(m["source_transmission"]["path"]) as im:
            t=np.asarray(im.convert("RGB"),dtype=float)/255
        with Image.open(m["source_reflection"]["path"]) as im:
            r=np.asarray(align_cabin(im.convert("RGB"),t.shape[:2]),dtype=float)/255
        T=srgb_to_linear(t)
        R=srgb_to_linear(r)*float(m["relative_cabin_illuminance_hypothesis"])
        gc=m["glass"];d=float(gc["thickness_m"])
        depth=np.asarray(gc["optical_depth_rgb"],dtype=float)
        stack=[
           {"n":float(gc["n"]),"thickness_m":.42*d,
            "optical_depth_rgb":(.48*depth).tolist()},
           {"n":1.48,"thickness_m":.16*d,
            "optical_depth_rgb":(.04*depth).tolist()},
           {"n":float(gc["n"]),"thickness_m":.42*d,
            "optical_depth_rgb":(.48*depth).tolist()}]
        intr=m["camera_calibration"]
        glass=Glass(n=gc["n"],optical_depth_rgb=tuple(depth),
                    normal_xyz=tuple(gc["normal_xyz"]),
                    focal_px=tuple(intr["focal_px"]),
                    principal_px=tuple(intr["principal_px"]),
                    thickness_m=d,camera_to_glass_m=gc["camera_to_glass_m"],
                    layers=tuple(stack))
        started=time.monotonic()
        result=render(T,R,glass,Camera(**m["camera"]),int(m["seed"]))
        elapsed=time.monotonic()-started
        out.mkdir()
        for key in ("observed","target_tinted"):
            Image.fromarray(np.uint8(np.round(result[key]*255))).save(out/(key+".png"))
        camera_valid=result["source_coverage_mask"]&~result["saturation_mask"]
        L=np.array([.2126,.7152,.0722])
        rl=result["reflected_linear"]@L
        tl=result["transmitted_linear"]@L
        visibility=rl/(tl+rl+1e-8)
        diagnostics={
          "id":row["id"],"source_group":row["capture_group"],
          "source_outdoor_sha256":m["source_transmission"]["sha256"],
          "source_cabin_sha256":m["source_reflection"]["sha256"],
          "source_rights":[m["source_transmission"]["source_rights"],
                           m["source_reflection"]["source_rights"]],
          "model_status":"UNREGISTERED PNG PROXY",
          "resolution":[int(t.shape[1]),int(t.shape[0])],
          "render_seconds":elapsed,
          "usable_fraction":float(camera_valid.mean()),
          "visibility_median":float(np.median(visibility[camera_valid])),
          "visibility_p90":float(np.quantile(visibility[camera_valid],.9)),
          "defocus_sigma":result["defocus_sigma_px"],
          "optical_error_bound":result["optical_model_diagnostics"]["optical_residual_energy_bound"]}
        (out/"metadata.json").write_text(json.dumps(diagnostics,indent=2)+"\n")
        measures.append(diagnostics)
        print("NATIVE_OPTICS",i,len(selected),diagnostics,flush=True)
    (args.out/"high_resolution_audit.json").write_text(json.dumps(
       {"scientific_validity":"NOT REAL WINDSHIELD CALIBRATION",
        "count":len(measures),"items":measures},indent=2)+"\n")
    print("FULLRES_VISUALS",len(measures),args.out,flush=True)
if __name__=="__main__":main()
