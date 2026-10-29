


from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Layer:
    n: float
    thickness_m: float
    optical_depth_rgb: tuple


def _stack(layers):
    out=[]
    for item in layers:
        layer=Layer(**item) if isinstance(item,dict) else item
        if not isinstance(layer,Layer):
            raise ValueError("Expected physical Layer objects")
        od=np.asarray(layer.optical_depth_rgb,dtype=np.float64)
        if (not np.isfinite(layer.n) or layer.n<1
                or not np.isfinite(layer.thickness_m) or layer.thickness_m<=0
                or od.shape!=(3,) or not np.all(np.isfinite(od))
                or np.any(od<0)):
            raise ValueError("Invalid passive glazing layer")
        out.append(layer)
    if not out:raise ValueError("Missing glazing layers")
    return tuple(out)


def _ray_coefficients(shape,glass,layers,normal_map=None):
    h,w=shape
    f=np.asarray(glass.focal_px,dtype=np.float64)
    pp=np.asarray(glass.principal_px,dtype=np.float64)
    n=np.asarray(glass.normal_xyz,dtype=np.float64)
    if (h<=0 or w<=0 or f.shape!=(2,) or np.any(f<=0)
            or not np.all(np.isfinite(f))
            or pp.shape!=(2,) or not np.all(np.isfinite(pp))
            or n.shape!=(3,) or not np.all(np.isfinite(n))
            or np.linalg.norm(n)<1e-10
            or not np.isfinite(glass.camera_to_glass_m)
            or glass.camera_to_glass_m<=0):
        raise ValueError("Invalid glass/camera geometry")
    n=n/np.linalg.norm(n)
    if normal_map is not None:
        nfield=np.asarray(normal_map,dtype=np.float64)
        if (nfield.shape!=(h,w,3) or not np.all(np.isfinite(nfield))
                or np.any(np.linalg.norm(nfield,axis=-1)<=1e-9)):
            raise ValueError("Local windshield normal field has invalid shape/values")
        nfield=nfield/np.linalg.norm(nfield,axis=-1,keepdims=True)


        if np.any(np.sum(nfield*n,axis=-1)<=0):
            raise ValueError("Windshield local normal points backwards")
    else:
        nfield=n
    y,x=np.mgrid[:h,:w]
    ray=np.stack(((x-pp[0])/f[0],(y-pp[1])/f[1],
                  np.ones_like(x)),axis=-1)
    ray/=np.linalg.norm(ray,axis=-1,keepdims=True)
    cos0=np.sum(ray*nfield,axis=-1)
    if np.any(cos0<=1e-4):
        raise ValueError("Invalid back-facing incident rays")
    transverse=ray-cos0[...,None]*nfield
    sin2=np.clip(1-cos0*cos0,0,1)
    ns=(1.,)+tuple(layer.n for layer in layers)+(1.,)
    cosines=[np.sqrt(1-sin2/(nr*nr)) for nr in ns]
    refl_s=[];refl_p=[]
    for i in range(len(ns)-1):
        ni,nj=ns[i:i+2]
        ci,cj=cosines[i:i+2]
        refl_s.append((((ni*ci-nj*cj)/(ni*ci+nj*cj))**2)[...,None])
        refl_p.append((((nj*ci-ni*cj)/(nj*ci+ni*cj))**2)[...,None])
    atten=[];shifts=[]
    for k,layer in enumerate(layers,1):
        cos=cosines[k]
        atten.append(np.exp(-np.asarray(layer.optical_depth_rgb)[None,None,:]
                            /cos[...,None]))
        lateral=(layer.thickness_m/cos)[...,None]*(transverse/ns[k])
        shifts.append((f[0]*lateral[...,0]/glass.camera_to_glass_m,
                       f[1]*lateral[...,1]/glass.camera_to_glass_m))
    return (refl_s,refl_p),atten,shifts


def trace_paths(shape,glass,*,weight_tolerance=1e-11,
                residual_tolerance=5e-8,normal_map=None):

    layers=_stack(glass.layers)
    powers,atten,shifts=_ray_coefficients(shape,glass,layers,normal_map)
    nlay=len(layers)
    reflection={};transmission={}
    counted_reflection=0;counted_transmission=0
    lossbound=0.
    for pol in (0,1):
        fresnel=powers[pol]
        stack=[(0,1,np.ones((*shape,3),dtype=np.float64)*.5,
                (0,)*nlay,0)]
        while stack:
            medium,direction,weight,counts,bounces=stack.pop()
            if np.max(weight)<weight_tolerance:
                lossbound+=float(np.max(weight))
                continue
            k=medium if direction>0 else medium-1
            f=fresnel[k]
            for next_direction,factor,reflected in (
                (-direction,f,True),(direction,1-f,False)):
                medium_next=medium if reflected else medium+direction
                coeff=weight*factor
                if np.max(coeff)<weight_tolerance:
                    lossbound+=float(np.max(coeff))
                    continue
                nb=bounces+int(reflected)
                if medium_next==0 and next_direction<0:
                    counted_reflection+=1
                    if counts in reflection:reflection[counts]+=coeff
                    else:reflection[counts]=coeff
                    continue
                if medium_next==nlay+1 and next_direction>0:
                    counted_transmission+=1
                    if counts in transmission:transmission[counts]+=coeff
                    else:transmission[counts]=coeff
                    continue
                if not 1<=medium_next<=nlay:
                    raise AssertionError("Unphysical ray state")
                nc=list(counts)
                nc[medium_next-1]+=1
                stack.append((medium_next,next_direction,
                              coeff*atten[medium_next-1],tuple(nc),nb))
            if len(stack)>512 or counted_reflection+counted_transmission>4096:
                return _trace_paths_dynamic(shape,glass,powers,atten,shifts,
                                            weight_tolerance,residual_tolerance,
                                            normal_map)
    if lossbound>residual_tolerance:


        narrowed=weight_tolerance * residual_tolerance/(2*lossbound)
        if narrowed<=np.finfo(np.float64).eps or narrowed>=weight_tolerance:
            raise ValueError("Optical energy bound cannot be satisfied at machine precision")
        return trace_paths(shape,glass,weight_tolerance=narrowed,
                           residual_tolerance=residual_tolerance,
                           normal_map=normal_map)
    t=sum(transmission.values(),np.zeros((*shape,3)))
    r=sum(reflection.values(),np.zeros((*shape,3)))
    if np.min(t)<-1e-12 or np.min(r)<-1e-12 or np.max(t+r)>1+5e-8:
        raise AssertionError("Nonpassive optical paths")
    return {
        "reflection_paths":[(value,counts,0) for counts,value in reflection.items()],
        "transmission_paths":[(value,counts,0) for counts,value in transmission.items()],
        "R":r,"T":t,"shifts":shifts,
        "residual_energy_bound":lossbound,
        "path_count_r":counted_reflection,"path_count_t":counted_transmission,
        "path_groups_r":len(reflection),"path_groups_t":len(transmission),
        "interfaces":nlay+1,
        "transport_algorithm":"direct path enumeration",
        "max_simultaneous_states":len(stack) if "stack" in locals() else 0}


def compose_multilayer(t,r,glass,blur_fn,warp_fn,sigma,reflection_valid_mask=None,
                       *,exposure=1.,full_well_e=1e10,normal_map=None):

    hh,ww=t.shape[:2]


    if (not np.isfinite(exposure) or exposure<=0
            or not np.isfinite(full_well_e) or full_well_e<=0):
        raise ValueError("Invalid radiometric conversion")
    luminance_bound=max(float(np.max(t)),float(np.max(r)),1e-15)
    tolerance=0.25/(exposure*full_well_e*luminance_bound)
    tolerance=min(tolerance,5e-8)
    layers_count=len(_stack(glass.layers))
    atomic_tol=tolerance / ((layers_count+1)**2)
    tr=trace_paths((hh,ww),glass,
                   weight_tolerance=atomic_tol,
                   residual_tolerance=tolerance,normal_map=normal_map)
    blurred=blur_fn(r,sigma)
    if reflection_valid_mask is None:
        reflection_valid=np.ones((hh,ww),dtype=bool)
    else:
        reflection_valid=np.asarray(reflection_valid_mask)
        if reflection_valid.shape!=(hh,ww) or reflection_valid.dtype!=np.bool_:
            raise ValueError("Geometry reflection mask must be HxW boolean")
        reflection_valid=reflection_valid.copy()
    blur_valid=(blur_fn(reflection_valid.astype(np.float64)[...,None],
                        sigma)[...,0]>=1-1e-8)
    transmission=np.zeros_like(t)
    reflection=np.zeros_like(r)
    valid=np.ones((hh,ww),dtype=bool)
    yy,xx=np.mgrid[:hh,:ww]


    for paths,source,result,mask in (
        (tr["reflection_paths"],blurred,reflection,blur_valid),
        (tr["transmission_paths"],t,transmission,None)):
        for coeff,counts,_ in paths:
            dx=np.zeros((hh,ww));dy=np.zeros((hh,ww))
            for traversals,(sx,sy) in zip(counts,tr["shifts"]):
                if traversals:
                    dx+=traversals*sx
                    dy+=traversals*sy
            warped=warp_fn(source,dx,dy) if any(counts) else source
            result+=coeff*warped
            if np.max(coeff)>1e-7:
                valid &= ((xx-dx>=0)&(xx-dx<=ww-1)
                          &(yy-dy>=0)&(yy-dy<=hh-1))
                if mask is not None:
                    projected=warp_fn(mask.astype(np.float64)[...,None],
                                      dx,dy)[...,0]
                    valid &= projected>=1-1e-8
    return transmission,reflection,valid,tr


def _trace_paths_dynamic(shape,glass,powers,atten,shifts,
                         weight_tolerance,residual_tolerance,normal_map):


    from collections import defaultdict
    import os
    nlay=len(atten)
    reflection={};transmission={}
    lost=0.
    count_ref=0;count_trans=0
    max_states=0
    for pol in (0,1):
        fresnel=powers[pol]
        frontier={(0,1,(0,)*nlay):
                  np.ones((*shape,3),dtype=np.float64)*.5}
        while frontier:
            nxt={}
            for (medium,direction,counts),weight in frontier.items():
                if np.max(weight)<weight_tolerance:
                    lost+=float(np.max(weight))
                    continue
                interface=medium if direction>0 else medium-1
                f=fresnel[interface]
                for nd,coefficient,reflect in (
                    (-direction,f,True),(direction,1-f,False)):
                    dest=medium if reflect else medium+direction
                    amount=weight*coefficient
                    if np.max(amount)<weight_tolerance:
                        lost+=float(np.max(amount))
                        continue
                    if dest==0 and nd<0:
                        count_ref+=1
                        if counts in reflection:reflection[counts]+=amount
                        else:reflection[counts]=amount
                        continue
                    if dest==nlay+1 and nd>0:
                        count_trans+=1
                        if counts in transmission:transmission[counts]+=amount
                        else:transmission[counts]=amount
                        continue
                    if not 1<=dest<=nlay:
                        raise AssertionError("Impossible optical medium transition")
                    entered=list(counts)
                    entered[dest-1]+=1
                    key=(dest,nd,tuple(entered))
                    amount=amount*atten[dest-1]
                    if key in nxt:nxt[key]+=amount
                    else:nxt[key]=amount
            frontier=nxt
            max_states=max(max_states,len(nxt))
            if frontier:


                page=os.sysconf("SC_PAGE_SIZE")
                remaining=os.sysconf("SC_AVPHYS_PAGES")*page
                bytes_per_state=int(np.prod(shape)*3*8)
                projected=(len(frontier)+len(reflection)+len(transmission))*bytes_per_state
                if projected>remaining*.1:
                    raise MemoryError("Optical dynamic frontier exceeds adaptive safe RAM share")
    if lost>residual_tolerance:
        narrowed=weight_tolerance*residual_tolerance/(2*lost)
        if narrowed<=np.finfo(np.float64).eps or narrowed>=weight_tolerance:
            raise ValueError("Dynamic optical residual cannot converge at machine precision")
        return _trace_paths_dynamic(shape,glass,powers,atten,shifts,
                                    narrowed,residual_tolerance,normal_map)
    T=sum(transmission.values(),np.zeros((*shape,3),dtype=np.float64))
    R=sum(reflection.values(),np.zeros((*shape,3),dtype=np.float64))
    if np.min(T)<-1e-12 or np.min(R)<-1e-12 or np.max(R+T)>1+5e-8:
        raise AssertionError("Nonpassive merged optical states")
    return {
        "reflection_paths":[(power,counts,0) for counts,power in reflection.items()],
        "transmission_paths":[(power,counts,0) for counts,power in transmission.items()],
        "R":R,"T":T,"shifts":shifts,"residual_energy_bound":lost,
        "path_count_r":count_ref,"path_count_t":count_trans,
        "path_groups_r":len(reflection),"path_groups_t":len(transmission),
        "interfaces":nlay+1,
        "transport_algorithm":"merged-state dynamic transport",
        "max_simultaneous_states":max_states,
    }
