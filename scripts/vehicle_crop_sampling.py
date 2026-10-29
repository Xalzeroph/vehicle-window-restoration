


import hashlib,math
import numpy as np
from PIL import Image


def choose_content_patch(rgb_image,side,principal_xy,source_key):
    if not isinstance(rgb_image,Image.Image):
        raise TypeError("Require original RGB capture")
    width,height=rgb_image.size
    if side<=0 or min(width,height)<side:
        raise ValueError("Invalid patch size")
    cx,cy=principal_xy
    if not np.all(np.isfinite([cx,cy])):
        raise ValueError("Calibrated camera principal point required")

    count_x=max(2,math.ceil(width/side))
    count_y=max(2,math.ceil(height/side))


    y_lower=np.clip(math.ceil(cy-side/2),0,height-side)
    xs=np.unique(np.rint(np.linspace(0,width-side,count_x)).astype(int))
    ys=np.unique(np.rint(np.linspace(y_lower,height-side,count_y)).astype(int))
    im=np.asarray(rgb_image.convert("L"),dtype=np.float32)/255.
    grad=np.zeros_like(im)
    grad[1:,:]+=abs(np.diff(im,axis=0))
    grad[:,1:]+=abs(np.diff(im,axis=1))

    prefix=np.pad(grad,((1,0),(1,0))).cumsum(axis=0).cumsum(axis=1)
    entries=[]
    for y in ys:
        for x in xs:
            amount=prefix[y+side,x+side]-prefix[y,x+side]-prefix[y+side,x]+prefix[y,x]
            edge=float(amount/(side*side))
            entries.append((edge,int(y),int(x)))
    if not entries:
        raise ValueError("No valid patch geometry")

    entries.sort(key=lambda row:(-row[0],row[1],row[2]))
    ceiling=max(1,math.ceil(len(entries)/3))
    digest=hashlib.sha256(source_key.encode()).digest()
    choice=int.from_bytes(digest[:8],"big")%ceiling
    edge,y,x=entries[choice]
    return x,y,{
        "policy":"lower_optical_field_content_heuristic_NOT semantic",
        "gradient_score":edge,
        "candidate_count":len(entries),
        "content_rank":choice+1,
        "image_principal_px":[float(cx),float(cy)],
        "no_semantic_ground_truth":True,
    }
