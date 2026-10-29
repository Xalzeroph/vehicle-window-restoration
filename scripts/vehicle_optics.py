


from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Glass:
    n: float
    optical_depth_rgb: tuple
    normal_xyz: tuple
    focal_px: tuple
    principal_px: tuple
    ghost_shift_px: tuple = (0., 0.)
    thickness_m: float = 0.004
    camera_to_glass_m: float = 0.6
    reflection_mode: str = "preprojected"
    rotation_camera_to_env: tuple = ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
    layers: tuple = ()


@dataclass(frozen=True)
class Camera:
    exposure: float
    wb_rgb: tuple
    full_well_e: float
    read_noise_e: float
    focal_length_mm: float = 0.
    f_number: float = 0.
    focus_distance_m: float = 0.
    reflection_distance_m: float = 0.
    sensor_width_mm: float = 0.
    sensor_native_width_px: float = 0.


def srgb_to_linear(image):
    x = np.asarray(image, dtype=np.float64)
    if not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("sRGB must be finite in [0,1]")
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055)**2.4)


def linear_to_srgb(image):
    x = np.clip(image, 0, 1)
    return np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)


def _geometry(h, w, g):
    if h <= 0 or w <= 0:
        raise ValueError("Empty image")
    if not np.isfinite(g.n) or g.n <= 1:
        raise ValueError("Invalid index of refraction")
    od = np.asarray(g.optical_depth_rgb, dtype=np.float64)
    n = np.asarray(g.normal_xyz, dtype=np.float64)
    if (od.shape != (3,) or np.any(od < 0) or not np.all(np.isfinite(od))
            or n.shape != (3,) or not np.all(np.isfinite(n))
            or np.linalg.norm(n) <= 0):
        raise ValueError("Invalid optical depth or normal")
    if (not np.all(np.isfinite(g.focal_px)) or min(g.focal_px) <= 0
            or not np.all(np.isfinite(g.principal_px))
            or not np.all(np.isfinite(g.ghost_shift_px))
            or not np.isfinite(g.thickness_m) or g.thickness_m <= 0
            or not np.isfinite(g.camera_to_glass_m) or g.camera_to_glass_m <= 0):
        raise ValueError("Invalid glass dimensions or camera intrinsics")
    n = n / np.linalg.norm(n)
    yy, xx = np.mgrid[:h, :w]
    ray = np.stack(((xx-g.principal_px[0])/g.focal_px[0],
                    (yy-g.principal_px[1])/g.focal_px[1],
                    np.ones_like(xx)), axis=-1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    cos_i = np.sum(ray * n, axis=-1)
    if np.any(cos_i <= 1e-4):
        raise ValueError("Plane normal must face camera rays (no grazing/back-facing rays)")
    cos_t = np.sqrt(1 - (1-cos_i**2)/g.n**2)
    rs = ((cos_i-g.n*cos_t)/(cos_i+g.n*cos_t))**2
    rp = ((g.n*cos_i-cos_t)/(g.n*cos_i+cos_t))**2
    attenuation = np.exp(-od[None,None,:] / cos_t[...,None])


    internal = ray/g.n + (cos_t-cos_i/g.n)[...,None]*n
    lateral = internal - cos_t[...,None]*n
    meters = (2*g.thickness_m / cos_t)[...,None] * lateral

    shift_x = g.focal_px[0] * meters[...,0]/g.camera_to_glass_m + g.ghost_shift_px[0]
    shift_y = g.focal_px[1] * meters[...,1]/g.camera_to_glass_m + g.ghost_shift_px[1]
    return rs, rp, attenuation, shift_x, shift_y


def _coefficients_from_paths(rs, rp, a):


    def paths(f):
        f = f[...,None]
        denominator = 1 - f*f*a*a
        t = (1-f)**2 * a / denominator
        r_front = np.broadcast_to(f, t.shape)
        r_back = (1-f)**2 * f*a*a / denominator
        return t, r_front, r_back
    s = paths(rs)
    p = paths(rp)
    return tuple(.5*(x+y) for x,y in zip(s,p))


def coefficients(height, width, g):
    rs,rp,a,_,_ = _geometry(height,width,g)
    t,rf,rb = _coefficients_from_paths(rs,rp,a)
    if np.any(t+rf+rb > 1+1e-10) or np.any(t < 0):
        raise AssertionError("Violation of passive slab energy conservation")
    return t,rf,rb


def ghost(image, dx, dy):


    h,w,_ = image.shape
    yy,xx = np.mgrid[:h,:w]
    xx = xx - dx
    yy = yy - dy
    valid = (xx >= 0) & (yy >= 0) & (xx <= w-1) & (yy <= h-1)
    xx = np.clip(xx,0,w-1)
    yy = np.clip(yy,0,h-1)
    x0 = np.floor(xx).astype(int)
    y0 = np.floor(yy).astype(int)
    x1 = np.minimum(x0+1,w-1)
    y1 = np.minimum(y0+1,h-1)
    ax = (xx-x0)[...,None]
    ay = (yy-y0)[...,None]
    val = (1-ay)*((1-ax)*image[y0,x0]+ax*image[y0,x1]) + \
          ay*((1-ax)*image[y1,x0]+ax*image[y1,x1])
    return np.where(valid[...,None],val,0)


def _gaussian_blur(image, sigma):
    if sigma < .25:
        return image

    radius = math.ceil(4*sigma)
    grid = np.arange(-radius,radius+1,dtype=np.float64)
    kernel = np.exp(-.5*(grid/sigma)**2)
    kernel /= kernel.sum()
    result = image
    for axis in (0,1):
        pads = [(0,0)]*3
        pads[axis] = (radius,radius)
        extended = np.pad(result,pads,mode="reflect")
        result = sum(k*np.take(extended,indices=range(radius+j,
                  radius+j+result.shape[axis]),axis=axis)
                     for j,k in zip(grid.astype(int),kernel))
    return result


def _aperture_disk_blur(image, sigma):


    image=np.asarray(image,dtype=np.float64)
    if image.ndim!=3 or not np.all(np.isfinite(image)):
        raise ValueError("Defocus expects finite HxWxC radiance")
    if not np.isfinite(sigma) or sigma<0:
        raise ValueError("Invalid defocus radius")
    if sigma<.25:
        return image
    disk_radius=2.*sigma
    support=math.ceil(disk_radius+.5)
    yy,xx=np.mgrid[-support:support+1,-support:support+1]

    sub=(np.arange(8)+.5)/8-.5
    aperture=np.zeros_like(xx,dtype=np.float64)
    for sy in sub:
        for sx in sub:
            aperture+=((xx+sx)**2+(yy+sy)**2<=disk_radius**2)
    aperture/=aperture.sum()

    pad=support
    extended=np.pad(image,((pad,pad),(pad,pad),(0,0)),mode="reflect")
    kh,kw=aperture.shape
    eh,ew=extended.shape[:2]
    fshape=(eh+kh-1,ew+kw-1)
    filter_freq=np.fft.rfft2(aperture,s=fshape)
    result=np.empty_like(image)
    y0=pad+kh//2;x0=pad+kw//2
    for channel in range(image.shape[-1]):
        freq=np.fft.rfft2(extended[...,channel],s=fshape)
        full=np.fft.irfft2(freq*filter_freq,s=fshape)
        result[...,channel]=full[y0:y0+image.shape[0],x0:x0+image.shape[1]]
    return np.maximum(result,0.)


def _defocus_sigma_px(camera,width):
    fields = (camera.focal_length_mm,camera.f_number,camera.focus_distance_m,
              camera.reflection_distance_m,camera.sensor_width_mm)
    if all(v == 0 for v in fields):
        return 0.
    if (not np.all(np.isfinite(fields)) or min(fields) <= 0):
        raise ValueError("Defocus requires all five positive measured camera parameters")
    f = camera.focal_length_mm
    distance = camera.reflection_distance_m*1e3
    focused = camera.focus_distance_m*1e3
    if distance <= f or focused <= f:
        raise ValueError("Object and focus distances must exceed focal length")
    image_distance = f*distance/(distance-f)
    focused_distance = f*focused/(focused-f)

    coc_mm = (f/camera.f_number)*abs(image_distance-focused_distance)/image_distance


    native_width=camera.sensor_native_width_px
    if native_width == 0:
        native_width=width
    if (not np.isfinite(native_width) or native_width<width):
        raise ValueError("Original sensor width in pixels must cover current crop")
    coc_px = coc_mm * native_width / camera.sensor_width_mm

    return coc_px/4


def render(t, r, glass, camera, seed, reflection_valid_mask=None,
           normal_map=None):

    t = np.asarray(t,dtype=np.float64)
    r = np.asarray(r,dtype=np.float64)
    if t.ndim != 3 or t.shape[2] != 3:
        raise ValueError("Transmission must be HxWx3 linear radiance")
    if glass.reflection_mode == "equirectangular":
        r = project_reflected_environment(r, *t.shape[:2], glass,
                                          normal_map=normal_map)
    elif glass.reflection_mode != "preprojected":
        raise ValueError("Unknown reflection projection mode")
    if t.shape != r.shape:
        raise ValueError("Pre-projected reflection must have same resolution as T")
    if any(not np.all(np.isfinite(v)) or np.any(v<0) for v in (t,r)):
        raise ValueError("Scene radiance must be finite and nonnegative")
    wb = np.asarray(camera.wb_rgb,dtype=np.float64)
    if (wb.shape != (3,) or not np.all(np.isfinite(wb)) or np.any(wb <= 0)
            or not np.all(np.isfinite([camera.exposure,camera.full_well_e,
                                        camera.read_noise_e]))
            or camera.exposure <= 0 or camera.full_well_e <= 0
            or camera.read_noise_e < 0):
        raise ValueError("Invalid camera photometry")
    h,w,_ = t.shape
    sigma = _defocus_sigma_px(camera,w)
    if glass.layers:
        from vehicle_multilayer import compose_multilayer
        transmitted,reflected,coverage,transport=compose_multilayer(
            t,r,glass,_aperture_disk_blur,ghost,sigma,reflection_valid_mask,
            exposure=camera.exposure,full_well_e=camera.full_well_e,
            normal_map=normal_map)
        orders=max((sum(counts) for _,counts,_ in
                    transport["reflection_paths"]),default=0)
        model_diagnostics={
            "model":"incoherent planar multilayer glass",
            "interfaces":transport["interfaces"],
            "reflection_paths":transport["path_count_r"],
            "transmission_paths":transport["path_count_t"],
            "reflection_path_groups":transport["path_groups_r"],
            "transmission_path_groups":transport["path_groups_t"],
            "transport_algorithm":transport["transport_algorithm"],
            "max_simultaneous_states":transport["max_simultaneous_states"],
            "optical_residual_energy_bound":transport["residual_energy_bound"],
            "transmission_parallax_mode":"thin-plane pinhole approximation",
            "defocus_psf":"ideal thin-lens circular aperture, uncalibrated",
        }
    else:
        if normal_map is not None:
            raise ValueError("Calibrated normal field requires the multilayer optical path model")
        h,w,_ = t.shape
        rs,rp,a,sx,sy = _geometry(h,w,glass)
        tr,rf,rb = _coefficients_from_paths(rs,rp,a)
        if np.any(tr+rf+rb > 1+1e-10):
            raise AssertionError("Non-passive coating")
        transmitted = tr*t
        reflected_source = _aperture_disk_blur(r,sigma)
        reflected = rf*reflected_source
        coverage = np.ones((h,w),dtype=bool)


        contributions = [
            ((1-rs[...,None])**2 * rs[...,None] *a*a)*.5,
            ((1-rp[...,None])**2 * rp[...,None] *a*a)*.5
        ]
        ratios = [rs[...,None]**2 *a*a,rp[...,None]**2 *a*a]
        total_b = np.zeros_like(reflected)
        order = 0
        while True:
            for pol in (0,1):
                total_b += contributions[pol]*ghost(reflected_source,
                                                    sx*(order+1),sy*(order+1))

            tail = max(float(np.max(contributions[j]*ratios[j]/
                              (1-ratios[j]))) for j in (0,1))
            if tail <= 1e-9:
                break
            for j in (0,1):
                contributions[j] *= ratios[j]
            order += 1
            if order > 1000:
                raise ValueError("Grazing interface exceeds numeric series validity")
        reflected += total_b
        yy,xx = np.mgrid[:h,:w]
        coverage &= ((xx-sx >= 0) & (xx-sx <= w-1)
                     & (yy-sy >= 0) & (yy-sy <= h-1))
        orders=order+1
        if reflection_valid_mask is not None:
            mask=np.asarray(reflection_valid_mask)
            if mask.shape!=(h,w) or mask.dtype!=np.bool_:
                raise ValueError("Reflection geometry validity mask must be HxW boolean")
            safe=_aperture_disk_blur(mask.astype(np.float64)[...,None],sigma)[...,0]>=1-1e-8
            coverage &= safe
            for k in range(1,orders+1):
                coverage &= ghost(safe.astype(np.float64)[...,None],
                                  sx*k,sy*k)[...,0]>=1-1e-8
        model_diagnostics={
            "model":"incoherent planar single slab",
            "interfaces":2,
            "reflection_paths":2*orders,
            "transmission_paths":1,
            "optical_residual_energy_bound":1e-9,
            "defocus_psf":"ideal thin-lens circular aperture, uncalibrated",
        }
    rng = np.random.default_rng(seed)
    raw_ideal = (transmitted+reflected)*camera.exposure
    expected_e = raw_ideal*camera.full_well_e

    electrons = rng.poisson(np.minimum(expected_e,camera.full_well_e*2))
    electrons = np.clip(electrons+rng.normal(0,camera.read_noise_e,t.shape),
                        0,camera.full_well_e)
    raw_sensor = electrons/camera.full_well_e
    noisy_linear = raw_sensor*wb
    mixed = linear_to_srgb(noisy_linear)
    tinted = linear_to_srgb(transmitted*camera.exposure*wb)
    clear = linear_to_srgb(t*camera.exposure*wb)
    saturated = np.any((expected_e>=camera.full_well_e) |
                       (noisy_linear>=1),axis=-1)
    return dict(observed=mixed,target_tinted=tinted,target_clear=clear,
                transmitted_linear=transmitted,reflected_linear=reflected,
                raw_sensor_linear=raw_sensor,
                saturation_mask=saturated,source_coverage_mask=coverage,
                defocus_sigma_px=float(sigma),ghost_orders=orders,
                optical_model_diagnostics=model_diagnostics)


def project_reflected_environment(env, h, w, glass,normal_map=None):


    env = np.asarray(env,dtype=np.float64)
    if (env.ndim != 3 or env.shape[2] != 3
            or env.shape[1] != 2*env.shape[0]
            or not np.all(np.isfinite(env)) or np.any(env < 0)):
        raise ValueError("Cabin envmap must be finite nonnegative linear RGB 2:1")
    rotation = np.asarray(glass.rotation_camera_to_env,dtype=np.float64)
    if (rotation.shape != (3,3) or not np.all(np.isfinite(rotation))
            or np.linalg.norm(rotation@rotation.T-np.eye(3))>1e-5
            or abs(np.linalg.det(rotation)-1)>1e-5):
        raise ValueError("Camera/environment rotation must be proper orthonormal 3x3")
    n=np.asarray(glass.normal_xyz,dtype=np.float64)
    n=n/np.linalg.norm(n)
    if normal_map is not None:
        n=np.asarray(normal_map,dtype=np.float64)
        if (n.shape!=(h,w,3) or not np.all(np.isfinite(n))
                or np.any(np.linalg.norm(n,axis=-1)<=1e-9)):
            raise ValueError("Invalid local reflection normal map")
        n=n/np.linalg.norm(n,axis=-1,keepdims=True)
    y,x=np.mgrid[:h,:w]
    d=np.stack(((x-glass.principal_px[0])/glass.focal_px[0],
                (y-glass.principal_px[1])/glass.focal_px[1],
                np.ones_like(x)),axis=-1)
    d/=np.linalg.norm(d,axis=-1,keepdims=True)
    if np.any(np.sum(d*n,axis=-1)<=1e-4):
        raise ValueError("View rays outside reflected plane")
    bounced=d-2*np.sum(d*n,axis=-1)[...,None]*n
    b=bounced@rotation.T

    latitude=np.arcsin(np.clip(b[...,1],-1,1))
    azimuth=np.arctan2(b[...,0],b[...,2])
    u=(azimuth/(2*np.pi)+.5)*env.shape[1]-.5
    v=(.5-latitude/np.pi)*env.shape[0]-.5
    x0=np.floor(u).astype(np.int64)
    y0=np.floor(v).astype(np.int64)
    ax=(u-x0)[...,None]
    ay=(v-y0)[...,None]
    x0%=env.shape[1]
    x1=(x0+1)%env.shape[1]
    y0=np.clip(y0,0,env.shape[0]-1)
    y1=np.minimum(y0+1,env.shape[0]-1)
    return (1-ay)*((1-ax)*env[y0,x0]+ax*env[y0,x1]) + \
           ay*((1-ax)*env[y1,x0]+ax*env[y1,x1])
