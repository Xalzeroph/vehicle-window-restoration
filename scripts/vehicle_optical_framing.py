


from dataclasses import replace
import math
import numpy as np
from vehicle_optics import _defocus_sigma_px
from vehicle_multilayer import trace_paths


def required_source_halo(core_shape,glass,camera,source_peak_radiance):
    if len(core_shape)!=2 or min(core_shape)<=0:
        raise ValueError("Invalid source crop shape")
    if not np.isfinite(source_peak_radiance) or source_peak_radiance<0:
        raise ValueError("Invalid peak radiance")
    sigma=_defocus_sigma_px(camera,core_shape[1])
    support=math.ceil(2*sigma+.5) if sigma>=.25 else 0
    tolerance=min(.25/(camera.exposure*camera.full_well_e*
                        max(source_peak_radiance,1e-15)),5e-8)
    max_shift=0.
    if glass.layers:
        atomic=tolerance/((len(glass.layers)+1)**2)
        traced=trace_paths(core_shape,glass,
                           weight_tolerance=atomic,residual_tolerance=tolerance)
        for path in (traced["reflection_paths"]+traced["transmission_paths"]):
            _,counts,_=path
            dx=np.zeros(core_shape,dtype=np.float64)
            dy=np.zeros(core_shape,dtype=np.float64)
            for visits,(sx,sy) in zip(counts,traced["shifts"]):
                if visits:
                    dx+=visits*sx
                    dy+=visits*sy
            max_shift=max(max_shift,float(np.max(np.abs(dx))),
                          float(np.max(np.abs(dy))))
        path_tail=traced["residual_energy_bound"]
    else:
        from vehicle_optics import _geometry
        rs,rp,a,sx,sy=_geometry(*core_shape,glass)
        motion=max(float(np.max(np.abs(sx))),float(np.max(np.abs(sy))))
        for fresnel in (rs,rp):
            path=.5*(1-fresnel[...,None])**2*fresnel[...,None]*a*a
            ratio=fresnel[...,None]**2*a*a
            order=1
            while np.max(path)>tolerance:
                max_shift=max(max_shift,motion*order)
                path*=ratio
                order+=1
                if order>1000:
                    raise ValueError("Unbounded slab ghost extent")
        path_tail=tolerance
    return math.ceil(support+max_shift+1),{
       "circular_aperture_support_px":support,
       "max_retained_ray_offset_px":max_shift,
       "modeled_sensor_sigma_px":sigma,
       "halo_derivation":"all retained optical paths + exact disk PSF + bilinear interpolation",
       "physical_path_tail_upper_bound":path_tail}


def extend_crop_in_source(full_size,crop,halo):
    width,height=full_size
    x,y,side=crop
    if (min(width,height,side)<=0 or x<0 or y<0 or
            x+side>width or y+side>height or halo<0):
        raise ValueError("Invalid source crop/halo geometry")
    left=max(0,x-halo);top=max(0,y-halo)
    right=min(width,x+side+halo);bottom=min(height,y+side+halo)
    region=(left,top,right,bottom)
    local=(x-left,y-top,x-left+side,y-top+side)
    return region,local
