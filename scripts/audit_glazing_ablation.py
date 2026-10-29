#!/usr/bin/env python3


import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from vehicle_optics import linear_to_srgb


def png(path):
    return np.asarray(Image.open(path).convert("RGB"),dtype=np.float64)/255.


def measure(directory):
    metadata=json.loads((directory/"metadata.json").read_text())
    with np.load(directory/"linear_layers.npz",allow_pickle=False) as z:
        t=np.array(z["transmitted"],dtype=np.float64)
        r=np.array(z["reflected"],dtype=np.float64)
        sensor=np.array(z["raw_sensor"],dtype=np.float64)
        covered=np.array(z["source_covered"])
        saturated=np.array(z["saturated"])
    obs=png(directory/"observed.png")
    tinted=png(directory/"target_tinted.png")
    clear=png(directory/"target_clear.png")
    camera=metadata["config"]["camera"]
    wb=np.asarray(camera["wb_rgb"])
    exposure=camera["exposure"]
    if t.shape!=obs.shape or r.shape!=obs.shape:
        raise ValueError("Numerical/image shape mismatch")
    reconstruction=linear_to_srgb(sensor*wb)
    reference=linear_to_srgb(t*exposure*wb)
    error=float(np.max(np.abs(obs-reconstruction)))
    tinted_error=float(np.max(abs(tinted-reference)))
    quantization_bound=0.5/255+1e-6
    if error>quantization_bound or tinted_error>quantization_bound:
        raise AssertionError("Stored scene-linear layers disagree with output PNG")
    if np.any(t<0) or np.any(r<0) or np.any(sensor<0):
        raise AssertionError("Negative radiance")
    luminance_weights=np.array([.2126,.7152,.0722])
    lt=t@luminance_weights
    lr=r@luminance_weights
    ratio=lr/(lt+lr+1e-8)
    usable=covered & ~saturated
    return {
        "model":metadata["model"],
        "pixel_count":int(t.shape[0]*t.shape[1]),
        "observed_png_reconstruction_max_error":error,
        "tinted_png_reconstruction_max_error":tinted_error,
        "quantization_consistency_passed":True,
        "valid_pixel_fraction":float(usable.mean()),
        "saturation_fraction":float(saturated.mean()),
        "reflection_energy_fraction":float(lr.sum()/(lt.sum()+lr.sum()+1e-8)),
        "reflection_visibility_median":float(np.median(ratio[usable])) if usable.any() else None,
        "reflection_visibility_p90":float(np.quantile(ratio[usable],.9)) if usable.any() else None,
        "optical_model_diagnostics":metadata["optical_model_diagnostics"],
        "scientific_provenance_gate":metadata["scientific_provenance_gate"],
        "observed":obs,"tinted":tinted,"clear":clear,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--single",type=Path,required=True)
    parser.add_argument("--laminate",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    x=measure(a.single);y=measure(a.laminate)
    if x["observed"].shape!=y["observed"].shape:
        raise ValueError("Cannot compare images from different dimensions")
    delta=np.abs(x["observed"]-y["observed"])
    report={
      "status":"PASS FOR NUMERICAL CONSISTENCY; NOT A REAL-GLASS VALIDATION",
      "single":{k:v for k,v in x.items() if not isinstance(v,np.ndarray)},
      "laminate":{k:v for k,v in y.items() if not isinstance(v,np.ndarray)},
      "model_difference":{
        "observed_rgb_mean_abs_difference":float(np.mean(delta)),
        "observed_rgb_max_abs_difference":float(np.max(delta)),
        "any_rgb_changed_pixel_fraction":float(np.mean(np.any(delta>0,axis=-1))),
        "target_tinted_mean_abs_difference":float(np.mean(np.abs(x["tinted"]-y["tinted"]))),
        "target_clear_mean_abs_difference":float(np.mean(np.abs(x["clear"]-y["clear"]))),
      },
      "limitations":[
        "Input road/cabin images are not common-exposure RAW radiance",
        "Cabin viewpoint is unregistered with the windshield camera",
        "Glass is locally plane-parallel and lacks measured material spectra",
        "Not tested against independent real measured windshield photographs",
        "No visual-realism certification follows from numerical reconstruction",
      ]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    print("OPTICAL_ABLATION_AUDIT",a.out)
    print("DIFFERENCES",report["model_difference"])
    print("GLASS_GATE",report["laminate"]["scientific_provenance_gate"])


if __name__=="__main__":main()
