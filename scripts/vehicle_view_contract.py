


from __future__ import annotations

TASK = "outside_camera_viewing_vehicle_interior"
TRANSMISSION = "vehicle_interior_behind_glass"
REFLECTION = "exterior_environment_in_front_of_glass"

_T_CAPTURE = {"recorded_from_vehicle_exterior",
              "geometrically_registered_external_camera",
              "illustrative_unregistered_proxy"}
_R_CAPTURE = {"recorded_outdoor_environment",
              "geometrically_registered_outdoor_environment",
              "illustrative_unregistered_proxy"}


def validate_capture_contract(contract, *, permit_proxy=False):
    if not isinstance(contract, dict):
        raise ValueError("Missing exterior-to-interior camera/scene contract")
    if contract.get("task") != TASK:
        raise ValueError("Wrong reflection-removal task: expected OUTSIDE camera viewing INSIDE vehicle")
    if contract.get("camera_side") != "exterior":
        raise ValueError("Target imaging camera must be on the EXTERIOR side of the glass")
    transmission=contract.get("transmission")
    reflection=contract.get("reflection")
    if not isinstance(transmission,dict) or transmission.get("scene_role") != TRANSMISSION:
        raise ValueError("T must be vehicle INTERIOR seen through the window")
    if not isinstance(reflection,dict) or reflection.get("scene_role") != REFLECTION:
        raise ValueError("R must be the reflected EXTERIOR environment (sky, buildings, trees)")
    tc=transmission.get("capture_geometry")
    rc=reflection.get("capture_geometry")
    if tc not in _T_CAPTURE or rc not in _R_CAPTURE:
        raise ValueError("Missing or unsupported acquisition-side geometry provenance")
    if not permit_proxy and ("illustrative_unregistered_proxy" in (tc,rc)):
        raise ValueError("Unregistered proxy imagery cannot be certified exterior-to-interior training data")
    if ("illustrative_unregistered_proxy" in (tc,rc)
            and contract.get("calibration_status") == "measured"):
        raise ValueError("Unregistered source cannot claim measured automotive calibration")
    return {"is_exterior_to_interior":True,
            "geometrically_registered":(
                tc!="illustrative_unregistered_proxy" and
                rc!="illustrative_unregistered_proxy"),
            "training_realism_certified":False}


def validate_sample_metadata(metadata, *, permit_proxy=False):
    contract=metadata.get("capture_contract")
    checked=validate_capture_contract(contract,permit_proxy=permit_proxy)
    t=metadata.get("source_transmission",{})
    r=metadata.get("source_reflection",{})
    if t.get("scene_role") != TRANSMISSION or r.get("scene_role") != REFLECTION:
        raise ValueError("Sample T/R source roles contradict exterior-to-interior contract")
    if t.get("capture_geometry") != contract["transmission"]["capture_geometry"]:
        raise ValueError("T source capture provenance differs from camera contract")
    if r.get("capture_geometry") != contract["reflection"]["capture_geometry"]:
        raise ValueError("R source capture provenance differs from camera contract")
    return checked
