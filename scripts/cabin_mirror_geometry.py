


from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PinholePose:
    intrinsic: np.ndarray
    camera_to_world: np.ndarray


@dataclass(frozen=True)
class GlassPanel:
    center_xyz: tuple
    normal_xyz: tuple
    horizontal_xyz: tuple
    width_m: float
    height_m: float


def _camera(p):
    k=np.asarray(p.intrinsic,dtype=np.float64)
    pose=np.asarray(p.camera_to_world,dtype=np.float64)
    if (k.shape!=(3,3) or pose.shape!=(4,4)
            or not np.all(np.isfinite(k)) or not np.all(np.isfinite(pose))
            or np.any(np.diag(k)[:2]<=0)
            or np.max(np.abs(k[2]-[0,0,1]))>1e-9
            or np.max(np.abs(pose[3]-[0,0,0,1]))>1e-9):
        raise ValueError("Invalid calibrated pinhole camera")
    rot=pose[:3,:3]
    if (np.max(np.abs(rot.T@rot-np.eye(3)))>1e-5
            or abs(np.linalg.det(rot)-1)>1e-5):
        raise ValueError("Extrinsic rotation must be proper SO(3)")
    return k,pose


def _panel(p):
    center=np.asarray(p.center_xyz,dtype=np.float64)
    n=np.asarray(p.normal_xyz,dtype=np.float64)
    u=np.asarray(p.horizontal_xyz,dtype=np.float64)
    vals=np.array([*center,*n,*u,p.width_m,p.height_m])
    if (center.shape!=(3,) or n.shape!=(3,) or u.shape!=(3,)
            or not np.all(np.isfinite(vals)) or min(p.width_m,p.height_m)<=0
            or np.linalg.norm(n)==0 or np.linalg.norm(u)==0):
        raise ValueError("Invalid vehicle windshield plane")
    n=n/np.linalg.norm(n)
    u=u-n*(u@n)
    if np.linalg.norm(u)<=1e-8:
        raise ValueError("Horizontal axis must lie in glass plane")
    u=u/np.linalg.norm(u)
    v=np.cross(n,u)
    return center,n,u,v


def project_cabin_rgbd(rgb_linear,depth_m,source_camera,target_camera,
                       glass_panel,target_shape):


    rgb=np.asarray(rgb_linear,dtype=np.float64)
    depth=np.asarray(depth_m,dtype=np.float64)
    if (rgb.ndim!=3 or rgb.shape[-1]!=3
            or depth.shape!=rgb.shape[:2]
            or not np.all(np.isfinite(rgb)) or np.any(rgb<0)):
        raise ValueError("RGB and registered depth must have matching geometry")
    kh,source=_camera(source_camera)
    kt,target=_camera(target_camera)
    center,n,u,v=_panel(glass_panel)
    h,w=target_shape
    if h<=0 or w<=0:
        raise ValueError("Invalid output dimensions")
    valid=np.isfinite(depth)&(depth>0)
    ys,xs=np.nonzero(valid)
    if len(xs)==0:
        return np.zeros((h,w,3)),np.zeros((h,w),dtype=bool)
    z=depth[ys,xs]

    srcpix=np.stack((xs,ys,np.ones_like(xs)),axis=0).astype(np.float64)
    ray=np.linalg.solve(kh,srcpix)
    world=(source[:3,:3]@(ray*z)+source[:3,3,None]).T

    mirror=world-2*((world-center)@n)[:,None]*n[None,:]
    tgt=(mirror-target[:3,3])@target[:3,:3]
    ztarget=tgt[:,2]
    good=ztarget>1e-6
    projected=(kt@tgt.T).T
    px=projected[:,0]/np.where(good,projected[:,2],1)
    py=projected[:,1]/np.where(good,projected[:,2],1)

    eye=target[:3,3]
    direction=(mirror-eye)
    denom=direction@n
    s=((center-eye)@n)/np.where(np.abs(denom)>1e-12,denom,1)
    hit=eye+s[:,None]*direction
    rel=hit-center
    good &= ((s>0)&(s<1)&(np.abs(rel@u)<=glass_panel.width_m/2)
             &(np.abs(rel@v)<=glass_panel.height_m/2))


    xl=np.floor(px).astype(np.int64)
    yl=np.floor(py).astype(np.int64)
    ax=px-xl;ay=py-yl
    depthbuf=np.full(h*w,np.inf,dtype=np.float64)
    layers=[]
    for oy in (0,1):
        for ox in (0,1):
            xx=xl+ox; yy=yl+oy
            ok=good&(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            idx=(yy[ok]*w+xx[ok]).astype(np.intp)
            zs=ztarget[ok]
            weights=((ax if ox else 1-ax)*(ay if oy else 1-ay))[ok]
            np.minimum.at(depthbuf,idx,zs)
            layers.append((idx,zs,weights,rgb[ys[ok],xs[ok]]))
    canvas=np.zeros((h*w,3),dtype=np.float64)
    mass=np.zeros(h*w,dtype=np.float64)
    for idx,zs,weights,colors in layers:
        visible=np.abs(zs-depthbuf[idx])<=np.maximum(1e-4,depthbuf[idx]*.002)
        take=idx[visible]
        ww=weights[visible]
        np.add.at(canvas,take,ww[:,None]*colors[visible])
        np.add.at(mass,take,ww)
    mask=mass>1e-10
    canvas[mask]/=mass[mask,None]
    return canvas.reshape(h,w,3),mask.reshape(h,w)
