#!/usr/bin/env python3


import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from vehicle_optics import Glass, Camera, render, srgb_to_linear, ghost, _geometry
from vehicle_view_contract import validate_capture_contract
from cabin_mirror_geometry import (
    PinholePose, GlassPanel, project_cabin_rgbd
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_radiance(path, spec, calibration_status):
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False).astype(np.float64)
        if spec.get("color_space") != "linear_rgb":
            raise ValueError("NPY input must explicitly declare linear_rgb")
        multiplier = spec.get("radiance_multiplier")
        if multiplier is None or not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("NPY radiance multiplier is required and positive")
        return array * float(multiplier)
    if calibration_status == "measured":
        raise ValueError("Measured photometry requires calibrated linear .npy, never 8-bit inverse sRGB")
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        raise ValueError("Unsupported source format")
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)/255.
    return srgb_to_linear(array)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transmission", type=Path, required=True)
    ap.add_argument("--reflection", type=Path, required=True)
    ap.add_argument("--glass-normal-map", type=Path,
                    help="Registered camera-frame windshield surface normals NPY HxWx3")
    ap.add_argument("--depth", type=Path,
                    help="Registered metric depth NPY for calibrated RGB-D reflection geometry")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())


    contract=validate_capture_contract(
        cfg.get("capture_contract"),
        permit_proxy=cfg.get("calibration_status")=="illustrative")
    status = cfg.get("calibration_status")
    if status not in ("measured", "illustrative"):
        raise ValueError("Set calibration_status: measured or illustrative")
    spec = cfg.get("sources", {})
    t_spec = spec.get("T", {})
    r_spec = spec.get("R", {})
    if status == "measured":
        for s in (t_spec,r_spec):
            if not s.get("calibration_id") or s.get("radiance_unit") != "common_relative_radiance":
                raise ValueError("Measured source requires calibration_id and common radiance units")
    t = read_radiance(args.transmission,t_spec,status)
    r = read_radiance(args.reflection,r_spec,status)
    geom = cfg.get("projection")
    surface_cfg=cfg.get("glass_surface")
    surface_normals=None
    if args.glass_normal_map is not None:
        if (not surface_cfg or not cfg["glass"].get("layers")
                or surface_cfg.get("normal_coordinate_frame")!="target_camera"
                or not surface_cfg.get("surface_id")
                or args.glass_normal_map.suffix.lower()!=".npy"):
            raise ValueError("Local glass normals require layer stack and identified camera-frame surface geometry")
        if status=="measured":
            if (surface_cfg.get("model")!="measured_mesh_normals"
                    or not surface_cfg.get("verified")
                    or not surface_cfg.get("measurement_sha256")):
                raise ValueError("Measured normals require source mesh and verified measurement evidence")
        elif surface_cfg.get("model")!="analytical_proxy":
            raise ValueError("Illustrative normals must explicitly describe analytical proxy geometry")
        if (cfg["glass"].get("reflection_mode")!="equirectangular"
                and r_spec.get("geometry_status")!="curvature_registered_projection"):
            raise ValueError("Curved glass requires curvature-registered reflection field, not 2D crop")
        surface_normals=np.load(args.glass_normal_map,allow_pickle=False)
        if surface_normals.shape != t.shape:
            raise ValueError("Local glass normal map must match target image geometry")
        if geom is not None:
            raise ValueError("Flat-panel RGBD mirror projection cannot substitute curved-surface ray tracing")
    elif surface_cfg is not None:
        raise ValueError("Glass surface metadata provided without registered normal map")
    if geom is not None:
        if (geom.get("type") != "calibrated_rgbd_mirror"
                or not geom.get("calibration_verified")
                or geom.get("depth_unit") != "meters_optical_axis"
                or not geom.get("depth_rgb_registered")
                or not geom.get("calibration_id")
                or not args.depth
                or args.depth.suffix.lower() != ".npy"
                or status != "measured"):
            raise ValueError("RGB-D projection requires verified metric, RGB-aligned calibrated geometry")
        source=PinholePose(np.asarray(geom["source_intrinsic"]),
                           np.asarray(geom["source_camera_to_world"]))
        target=PinholePose(np.asarray(geom["target_intrinsic"]),
                           np.asarray(geom["target_camera_to_world"]))
        panel=GlassPanel(**geom["glass_panel"])
        depth=np.load(args.depth,allow_pickle=False)
        if depth.shape != r.shape[:2]:
            raise ValueError("Depth must register exactly to source RGB resolution")
        model_normal=np.asarray(cfg["glass"]["normal_xyz"],dtype=np.float64)
        model_normal/=np.linalg.norm(model_normal)
        panel_normal=np.asarray(panel.normal_xyz,dtype=np.float64)
        panel_normal/=np.linalg.norm(panel_normal)
        normal_target=target.camera_to_world[:3,:3].T@panel_normal
        if np.dot(normal_target,model_normal)<.9999:
            raise ValueError("Glass Fresnel normal inconsistent with calibrated world panel")
        if not np.allclose(target.intrinsic[0,0],cfg["glass"]["focal_px"][0],rtol=1e-7) or not np.allclose(target.intrinsic[1,1],cfg["glass"]["focal_px"][1],rtol=1e-7):
            raise ValueError("Glass Fresnel focal lengths differ from target calibrated geometry")
        r,geom_valid=project_cabin_rgbd(r,depth,source,target,panel,t.shape[:2])
    else:
        if args.depth is not None:
            raise ValueError("Depth provided without a calibrated geometric projection")
        geom_valid=np.ones(t.shape[:2],dtype=bool)
    glass=Glass(**cfg["glass"])
    result = render(t,r,glass,Camera(**cfg["camera"]),int(cfg["seed"]),
                    reflection_valid_mask=geom_valid,
                    normal_map=surface_normals)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.out}")
    args.out.mkdir(parents=True,exist_ok=True)
    for key in ("observed","target_tinted","target_clear"):
        Image.fromarray(np.round(result[key]*255).astype("uint8")).save(args.out/(key+".png"))
    np.savez_compressed(args.out/"linear_layers.npz",
                        transmitted=result["transmitted_linear"].astype("float32"),
                        reflected=result["reflected_linear"].astype("float32"),
                        raw_sensor=result["raw_sensor_linear"].astype("float32"),
                        saturated=result["saturation_mask"],
                        source_covered=result["source_coverage_mask"],
                        cabin_geom_covered=geom_valid)


    tl = result["transmitted_linear"]
    rl = result["reflected_linear"]
    luma = np.array([.2126,.7152,.0722])
    T = tl @ luma
    R = rl @ luma
    valid = result["source_coverage_mask"] & ~result["saturation_mask"]
    visible = R/(T+R+1e-8)
    report = {
        "model":result["optical_model_diagnostics"]["model"],
        "optical_model_diagnostics":result["optical_model_diagnostics"],
        "config": cfg,
        "capture_contract":cfg["capture_contract"],
        "capture_contract_verification":contract,
        "renderer_source_sha256":{
            module:digest(Path(__file__).with_name(module))
            for module in ("vehicle_optics.py","vehicle_multilayer.py",
                           "synthesize_vehicle_window.py")
        },
        "source_sha256": {"T":digest(args.transmission),"R":digest(args.reflection),
                          **({"depth":digest(args.depth)} if args.depth else {}),
                          **({"glass_normal_map":digest(args.glass_normal_map)}
                             if args.glass_normal_map else {})},
        "glass_surface":surface_cfg,
        "geometry_projection_valid_fraction":float(geom_valid.mean()),
        "source_path": {"T":str(args.transmission),"R":str(args.reflection)},
        "source_provenance": {"T":t_spec,"R":r_spec},
        "saturation_fraction": float(result["saturation_mask"].mean()),
        "source_coverage_fraction": float(result["source_coverage_mask"].mean()),
        "usable_fraction": float(valid.mean()),
        "reflection_visibility_median": float(np.median(visible[valid])) if np.any(valid) else None,
        "reflection_visibility_p90": float(np.quantile(visible[valid], .9)) if np.any(valid) else None,
        "defocus_sigma_px": result["defocus_sigma_px"],
        "ghost_orders": result["ghost_orders"],
        "limits": [
            "sRGB inversion is not camera RAW, cannot undo JPEG ISP",
            "source images must share a common radiance/exposure reference",
            "unregistered 2D source is NOT physically registered exterior reflection geometry; RGBD mode requires verified camera frames",
            "calibrated RGBD projection still cannot recover occluded or unseen cabin surfaces",
            "planar RGB multilayer (when layers supplied) omits curvature, coating interference, and true spectral dispersion",
            "local normal maps use tangent-plane optics, not exact curved-shell ray intersections",
            "pixel parallax of rear ghost is first-order, not full lens ray tracing",
            "no empirical realism guarantee without held-out genuine windshield captures",
        ],
        "targets": {
            "target_tinted":"reflection-free, with glass tint and attenuation",
            "target_clear":"reflection-free, virtual untinted scene radiance",
        },
    }

    reasons=[]
    if status != "measured":
        reasons.append("Source radiance/camera optics only illustrative")
    if not contract["geometrically_registered"]:
        reasons.append("Exterior-facing scene uses unregistered illustrative source geometry")
    for role,src in (("T",t_spec),("R",r_spec)):
        if src.get("source_split") != "train":
            reasons.append(role+": not a verified train-only split")
        if not src.get("license_verified_for_derivatives",False):
            reasons.append(role+": derivative-use authorization not verified")
        if not src.get("capture_group"):
            reasons.append(role+": capture-scene group absent; cannot prove disjointness")
    if geom is None and r_spec.get("geometry_status") not in (
            "calibrated_projection","curvature_registered_projection"):
        reasons.append("External environment reflection lacks measured geometric registration")
    if geom is not None and not np.all(geom_valid):
        reasons.append("Reflected exterior scene has unobserved/occluded pixels; require valid-pixel training mask")
    if surface_cfg is not None:
        if surface_cfg.get("model") != "measured_mesh_normals":
            reasons.append("Surface normals are analytical proxy, not measured windshield shape")
        if not surface_cfg.get("measurement_sha256"):
            reasons.append("No measured glass surface mesh evidence")

    report["scientific_provenance_gate"] = {
        "eligible_for_further_real_domain_validation": len(reasons)==0,
        "reasons": reasons,
        "not_a_realism_certificate": True}
    (args.out/"metadata.json").write_text(json.dumps(report,indent=2)+"\n")
    print("SYNTH_OK",args.out,"usable",round(report["usable_fraction"],4),
          "visible",report["reflection_visibility_median"])


if __name__ == "__main__":
    main()
