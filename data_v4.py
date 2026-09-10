# -*- coding: utf-8 -*-
"""data_v4.py - Complete physics pipeline for car window imaging.
Generates unlimited synthetic training data in real-time.
Physics: dual-pane Fresnel, spectral lighting, multi-layer reflection.
Output: (I, T_gt, R_cumulative, has_refl=True)
"""
import os, random, math, time
import numpy as np
from PIL import Image
BASE = os.path.dirname(os.path.abspath(__file__))
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from dataclasses import dataclass
# from typing import Any  # not used
# Physics constants
N_GLASS = 1.52
N_PVB = 1.48
N_AIR = 1.0003
GLASS_T_MM = 2.1
PVB_T_MM = 0.76
ABSORP = 0.04
# CCT table: temp -> (R, G, B) multipliers
_CCT = {
    2000: (1.0, 0.181, 0.0),
    3000: (1.0, 0.539, 0.239),
    4000: (1.0, 0.713, 0.476),
    5000: (1.0, 0.852, 0.739),
    5500: (1.0, 0.888, 0.834),
    6500: (0.994, 0.900, 0.865),
}
def cct_rgb(temp):
    """Continuous color temperature -> RGB multipliers (interpolated)."""
    temps = sorted(_CCT.keys())
    if temp <= temps[0]: return _CCT[temps[0]]
    if temp >= temps[-1]: return _CCT[temps[-1]]
    for i in range(len(temps)-1):
        t0, t1 = temps[i], temps[i+1]
        if t0 <= temp <= t1:
            f = (temp-t0)/(t1-t0)
            r0,g0,b0 = _CCT[t0]
            r1,g1,b1 = _CCT[t1]
            return (r0+(r1-r0)*f, g0+(g1-g0)*f, b0+(b1-b0)*f)
    return _CCT[temps[-1]]
def fresnel_R(cos_t, R0):
    """Schlick Fresnel: R(theta)=R0+(1-R0)*(1-cos)^5."""
    c = max(0.0, min(1.0, cos_t))
    return R0 + (1.0-R0)*(1.0-c)**5
def interface_R(n1, n2, cos_t, ch=1):
    """Interface Fresnel with wavelength dispersion (ch: 0=R, 1=G, 2=B)."""
    do = {0: -0.008, 1: 0.0, 2: 0.010}
    n1c = n1 + do.get(ch, 0.0)
    n2c = n2 + do.get(ch, 0.0)
    R0 = ((n1c-n2c)/(n1c+n2c))**2
    return fresnel_R(cos_t, R0)
@dataclass
class GlassParams:
    R_iface: list  # 4x3 list of floats (was np.ndarray)
    T_total: list  # 3 floats
    ghost_off: list  # 3x2 list of floats
    ghost_int: float
def dual_pane(cos_t, device=None):
    """Compute 4-interface Fresnel for dual-pane glass. Returns GPU tensors when device is set."""
    R = [[0.0]*3 for _ in range(4)]
    for ch in range(3):
        R[0][ch] = interface_R(N_AIR, N_GLASS, cos_t, ch)
        R[1][ch] = interface_R(N_GLASS, N_PVB, cos_t, ch)
        R[2][ch] = interface_R(N_PVB, N_GLASS, cos_t, ch)
        R[3][ch] = interface_R(N_GLASS, N_AIR, cos_t, ch)
    T = [1.0]*3
    for c in range(3):
        t = (1-R[0][c])*(1-ABSORP)*(1-R[1][c])*(1-ABSORP*0.3)*(1-R[2][c])*(1-ABSORP)*(1-R[3][c])
        T[c] = max(0.7, min(0.96, t))
    theta = math.acos(max(0.01, min(0.99, cos_t)))
    sin_t = math.sin(theta)
    thick = 2*GLASS_T_MM + PVB_T_MM
    off_mag = thick*0.15*sin_t/max(0.1, math.sqrt(N_GLASS**2-sin_t**2))*4
    off_mag = max(0.5, min(8.0, off_mag))
    ang = random.uniform(0, 2*math.pi)
    go = [[0.0, 0.0] for _ in range(3)]
    for ch in range(3):
        cf = 1.0 + (ch-1)*0.03
        go[ch][0] = math.cos(ang)*off_mag*cf
        go[ch][1] = math.sin(ang)*off_mag*cf
    ghost_int = R[3][1] * random.uniform(0.3, 0.6)
    if device is not None:
        return GlassParams(
            torch.tensor(R, device=device, dtype=torch.float32),  # [4,3]
            torch.tensor(T, device=device, dtype=torch.float32),  # [3]
            torch.tensor(go, device=device, dtype=torch.float32),  # [3,2]
            torch.tensor(ghost_int, device=device))
    return GlassParams(R, T, go, ghost_int)  # lists or tensors
class ImagingDegradation:
    """Complete imaging degradation pipeline."""
    _kernel_cache = {}
    @staticmethod
    def defocus_blur(img, radius, device="cpu"):
        """Gaussian defocus blur (depth-dependent) with kernel caching."""
        if radius < 0.3:
            return img
        k = int(2*math.ceil(2*radius)+1)
        if k < 3:
            return img
        key = (k, round(radius, 2), device)
        if key not in ImagingDegradation._kernel_cache:
            grid = torch.arange(k, dtype=torch.float32, device=device) - k//2
            k1d = torch.exp(-grid**2/(2*max(0.1,radius)**2))
            k1d = k1d/k1d.sum()
            k2d = k1d[:,None] @ k1d[None,:]
            kernel = k2d[None,None,:,:].repeat(3,1,1,1)
            ImagingDegradation._kernel_cache[key] = kernel
        kernel = ImagingDegradation._kernel_cache[key]
        pad = k//2
        padded = F.pad(img[None], (pad,pad,pad,pad), mode="reflect")
        return F.conv2d(padded, kernel, groups=3)[0]
    @staticmethod
    def chromatic_aberration(img, shift, device="cpu"):
        """Chromatic aberration: RGB channel shifts."""
        if abs(shift) < 0.5:
            return img
        dx, dy = int(round(shift*0.7)), int(round(shift*0.7))
        if abs(dx) == 0 and abs(dy) == 0:
            return img
        result = img.clone()
        _, H, W = img.shape
        if abs(dx) < W and abs(dy) < H:
            result[0] = torch.roll(img[0], shifts=(dy,dx), dims=(0,1))
            result[2] = torch.roll(img[2], shifts=(-dy,-dx), dims=(0,1))
        return result
    @staticmethod
    def sensor_noise(img, iso):
        """Poisson-Gaussian noise: I_noisy = I + sqrt(I/iso)*e1 + e2."""
        if iso < 100:
            return img
        shot = torch.sqrt(img/iso+1e-8) * torch.randn_like(img)
        read_std = 0.01*(iso/100.0)**0.7
        read = torch.randn_like(img)*read_std
        quant = torch.rand_like(img)/255.0
        return (img + shot*0.7 + read*0.2 + quant*0.1).clamp(0,1)
    @staticmethod
    def vignetting(H, W, strength, device="cpu"):
        """cos^4 radial falloff vignetting."""
        y, x = torch.meshgrid(
            torch.linspace(-1,1,H,device=device),
            torch.linspace(-1,1,W,device=device),
            indexing="ij")
        cx, cy = random.uniform(-0.3,0.3), random.uniform(-0.3,0.3)
        r2_off = (x-cx)**2+(y-cy)**2
        return (1-strength*0.7*r2_off).clamp(0.2,1.0)[None,:,:]
    @staticmethod
    def apply_isp(img):
        """ISP: sRGB gamma + tone mapping + quantization."""
        mask = img <= 0.0031308
        srgb = torch.where(mask, 12.92*img, 1.055*(img.clamp(min=1e-8)**(1/2.4))-0.055)
        tm = srgb/(srgb+0.2)*1.2
        return (tm*255).round()/255.0
    @staticmethod
    def env_degradation(img, device="cpu"):
        """Environmental degradation: rain/fog/dust."""
        r = random.random()
        C, H, W = img.shape
        result = img.clone()
        if r < 0.15:
            fog = random.uniform(0.05, 0.25)
            fc = torch.ones(C,H,W,device=device)*random.uniform(0.7,0.95)
            result = result*(1-fog)+fc*fog
        elif r < 0.25:
            for _ in range(random.randint(3,15)):
                cx = random.randint(0, W-1); cy = random.randint(0, H-1)
                length = random.randint(10, 40)
                intensity = random.uniform(0.15, 0.4)
                angle = random.uniform(-0.3, 0.3)
                t_vals = torch.arange(length, device=device).float()
                px = (cx + t_vals * angle).long().clamp(0, W-1)
                py = (cy + t_vals).long().clamp(0, H-1)
                result[:, py, px] = result[:, py, px] * (1 - intensity) + intensity
        elif r < 0.32:
            for _ in range(random.randint(2,8)):
                cx, cy = random.randint(0,W-1), random.randint(0,H-1)
                radius = random.uniform(1,8)
                intensity = random.uniform(0.1,0.4)
                yg, xg = torch.meshgrid(torch.arange(H,device=device),
                                        torch.arange(W,device=device), indexing="ij")
                dist2 = (xg-cx)**2+(yg-cy)**2
                mask = torch.exp(-dist2/(2*radius**2))
                val = random.uniform(0.4, 0.6)
                result = result * (1 - mask * intensity) + mask * intensity * val
        return result.clamp(0,1)
class ReflectionCompositor:
    """Multi-layer reflection compositing: 2-3 layers + ghost."""
    @staticmethod
    def apply(T_gt, R_tex, gp, device="cpu"):
        """Composite reflections onto clean background. Returns (I, T_gt, R_cum).
        Fully GPU-optimized: no Python-in-loop GPU-to-CPU syncs.
        """
        T_gt = T_gt.to(device)
        R_tex = R_tex.to(device)
        C, H, W = T_gt.shape
        n_layers = random.randint(2, 3)
        R_cum = torch.zeros_like(T_gt)
        ghost_int = gp.ghost_int
        # Ensure GPU tensors (handle CPU list from dual_pane CPU path)
        if not hasattr(gp.R_iface, "mean"):
            gp.R_iface = torch.tensor(gp.R_iface, device=device, dtype=torch.float32)
            gp.T_total = torch.tensor(gp.T_total, device=device, dtype=torch.float32)
            gp.ghost_off = torch.tensor(gp.ghost_off, device=device, dtype=torch.float32)
            gp.ghost_int = torch.tensor(gp.ghost_int, device=device)
            ghost_int = gp.ghost_int
        R0_mean = gp.R_iface[0].mean()
        R3_mean = gp.R_iface[3].mean()
        ch_mult = torch.tensor([1.03, 1.0, 0.97], device=device).view(1, 3, 1, 1)
        for layer in range(n_layers):
            Rl = R_tex.clone()
            if layer == 0:
                beta = R0_mean * random.uniform(0.8, 1.5)
                blur = random.uniform(0.5, 2.0)
                contrast = random.uniform(0.5, 0.9)
            elif layer == 1:
                beta = R3_mean * random.uniform(0.5, 1.0)
                blur = random.uniform(2.0, 5.0)
                contrast = random.uniform(0.3, 0.7)
            else:
                beta = R0_mean * random.uniform(0.8, 1.5)
                blur = random.uniform(4.0, 10.0)
                contrast = random.uniform(0.2, 0.5)
            Rl = ImagingDegradation.defocus_blur(Rl, blur, device)
            lm = Rl.mean(dim=[1,2], keepdim=True)
            Rl = (Rl-lm)*contrast+lm
            beta_ch = (beta * ch_mult).view(3, 1, 1)
            Rl = Rl*beta_ch
            if layer > 0:
                dx = random.randint(-6, 6)
                dy = random.randint(-6, 6)
                if abs(dx) > 0 or abs(dy) > 0:
                    Rl = torch.roll(Rl, shifts=(dy,dx), dims=(1,2))
            R_cum = R_cum + Rl
        if ghost_int > 0.01:
            ghost = R_tex.clone()
            for ch in range(3):
                dx = int(round(float(gp.ghost_off[ch][0])))
                dy = int(round(float(gp.ghost_off[ch][1])))
                if abs(dx) > 0 or abs(dy) > 0:
                    ghost[ch] = torch.roll(ghost[ch], shifts=(dy,dx), dims=(0,1))
            ghost = ImagingDegradation.defocus_blur(ghost, random.uniform(1.0,3.0), device)
            ghost = ghost*ghost_int*random.uniform(0.3,0.7)
            R_cum = R_cum + ghost
        R_cum = R_cum.clamp(0,1)
        cct = random.uniform(2000,6500)
        darkness = float(np.random.beta(1.5,3.0)*0.7+0.05)
        cr, cg, cb = cct_rgb(cct)
        lighting = torch.tensor([cr,cg,cb], device=device, dtype=torch.float32).view(3,1,1)*darkness
        Ttot = (gp.T_total.view(3,1,1) if hasattr(gp.T_total, 'view') else torch.tensor(gp.T_total, device=device).view(3,1,1))
        cast = torch.tensor([random.uniform(0.95,1.0), random.uniform(0.97,1.0), random.uniform(0.95,1.0)], device=device).view(3,1,1)
        alpha = Ttot.sqrt()
        beta = (1.0 - alpha * max(0.1, darkness)).clamp(0.05, 0.95)
        I = T_gt * alpha * lighting * cast + R_cum * beta
        I = I.clamp(0,1)
        iso = random.choice([100,200,400,800,1600,3200])
        I = ImagingDegradation.sensor_noise(I, iso)
        I = I * ImagingDegradation.vignetting(H, W, random.uniform(0.0,0.5), device)
        I = ImagingDegradation.apply_isp(I)
        I = ImagingDegradation.env_degradation(I, device)
        I = ImagingDegradation.chromatic_aberration(I, random.uniform(-3.0,3.0), device)
        I = I.clamp(0,1)
        return I, T_gt, R_cum
class HighQualityWindowDataset(Dataset):
    """Highest quality synthetic dataset. Lazy GPU cache: no startup delay."""
    _shared_preload = {}  # class-level: {(cache_key): cpu_tensors}
    def __init__(self, real_label_dirs, texture_dirs, scene_dir=None,
                 size=96, samples_per_epoch=5000, bg_ratio=0.5,
                 use_cuda=False, augment=True):
        self.size = size
        self.samples = samples_per_epoch
        self.bg_ratio = bg_ratio
        self.augment = augment
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.to_tensor = transforms.ToTensor()
        # Scan directories (instant: ~0.1s)
        self.backgrounds = {"real": [], "scene": []}
        for d in real_label_dirs:
            if os.path.exists(d):
                for scene in sorted(os.listdir(d)):
                    sp = os.path.join(d, scene)
                    if os.path.isdir(sp):
                        lp = os.path.join(sp, "label.jpg")
                        if os.path.exists(lp):
                            try: self.backgrounds["real"].append(Image.open(lp).convert("RGB"))
                            except: pass
        if scene_dir and os.path.exists(scene_dir):
            for f in os.listdir(scene_dir):
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    try: self.backgrounds["scene"].append(Image.open(os.path.join(scene_dir,f)).convert("RGB"))
                    except: pass
        self.textures = []
        for d in texture_dirs:
            if os.path.exists(d):
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith((".jpg",".jpeg",".png")):
                        try: self.textures.append(Image.open(os.path.join(d,f)).convert("RGB"))
                        except: pass
        n_real = len(self.backgrounds["real"])
        n_scene = len(self.backgrounds["scene"])
        n_tex = len(self.textures)
        print(f"  HQQD: {n_real} real + {n_scene} scene backgrounds x {n_tex} textures -> {samples_per_epoch}/epoch @ {size}^2")
        if not self.textures:
            self.textures = self.backgrounds["scene"] or self.backgrounds["real"]
        # Lazy GPU cache: textures stay on CPU until first use
        self.use_gpu_preload = False
        self._bg_gpu_cache = {}  # {idx: GPU tensor} filled lazily
        self._tex_gpu_cache = {}
        self._gpu_size_cap = max(320, self.size * 2)
        if use_cuda and torch.cuda.is_available():
            rdirs_key = tuple(sorted(set(str(d) for d in (real_label_dirs or []))))
            tdirs_key = tuple(sorted(set(str(d) for d in (texture_dirs or []))))
            cache_key = (rdirs_key, tdirs_key, self._gpu_size_cap)
            cache_file = os.path.join(BASE, ".texture_cache", "cache_{}.pt".format(self._gpu_size_cap))
            # Load CPU cache (shared across DDP ranks via file)
            if cache_key not in HighQualityWindowDataset._shared_preload:
                if os.path.exists(cache_file):
                    try:
                        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
                        HighQualityWindowDataset._shared_preload[cache_key] = cached
                        print(f"    LazyCache: {len(cached['tex'])} tex + {len(cached['bg'])} bg loaded ({os.path.getsize(cache_file)/1e9:.1f}GB on CPU)")
                    except Exception:
                        print("    Cache corrupt, will rebuild on demand...")
            if cache_key in HighQualityWindowDataset._shared_preload:
                c = HighQualityWindowDataset._shared_preload[cache_key]
                self._bg_cpu = c["bg"]
                self._tex_cpu = c["tex"]
                self.use_gpu_preload = True
                mb = (sum(t.numel()*4/1e6 for t in self._bg_cpu) +
                      sum(t.numel()*4/1e6 for t in self._tex_cpu))
                print(f"    LazyCache: {len(self._bg_cpu)} bg + {len(self._tex_cpu)} tex ({mb:.0f}MB CPU, VRAM=0)")
            else:
                # No cache: mark raw PIL for on-demand conversion
                self._bg_raw = {"real": list(self.backgrounds["real"]), "scene": list(self.backgrounds["scene"])}
                self._tex_raw = list(self.textures)
                self._bg_cpu = []
                self._tex_cpu = []
                self.use_gpu_preload = True
                print("    LazyCache: no cache, will convert from PIL on first use")
        else:
            self.use_gpu_preload = False
    def _to_gpu_bg(self, bidx):
        """Lazy-load background to GPU."""
        if hasattr(self, "_bg_cpu") and self._bg_cpu:
            if bidx not in self._bg_gpu_cache:
                self._bg_gpu_cache[bidx] = self._bg_cpu[bidx].to(self.device, non_blocking=True)
            return self._bg_gpu_cache[bidx]
        pil = random.choice(self._bg_raw["real"] if (random.random() < self.bg_ratio and self._bg_raw["real"]) else self._bg_raw.get("scene", self._bg_raw.get("real", [])))
        return self._pil_to_tensor(pil)
    def _to_gpu_tex(self, tidx):
        """Lazy-load texture to GPU."""
        if hasattr(self, "_tex_cpu") and self._tex_cpu:
            if tidx not in self._tex_gpu_cache:
                self._tex_gpu_cache[tidx] = self._tex_cpu[tidx].to(self.device, non_blocking=True)
            return self._tex_gpu_cache[tidx]
        pil = self._tex_raw[tidx % len(self._tex_raw)]
        return self._pil_to_tensor(pil)
    def _pil_to_tensor(self, pil):
        w, h = pil.size
        r = min(self._gpu_size_cap/w, self._gpu_size_cap/h, 1.0)
        return self.to_tensor(pil).pin_memory().to(self.device, non_blocking=True)
        return self.to_tensor(pil).to(self.device)
    def _pick_bg(self):
        if not self.use_gpu_preload:
            ur = random.random() < self.bg_ratio
            bt = "real" if ur and self.backgrounds["real"] else "scene"
            if not self.backgrounds[bt]: bt = "real" if bt=="scene" else "scene"
            return random.choice(self.backgrounds[bt])
        # GPU path: pick random, lazy-load
        n = len(self._bg_cpu) if hasattr(self, "_bg_cpu") and self._bg_cpu else 0
        if n > 0:
            return self._to_gpu_bg(random.randrange(n))
        return self._to_gpu_bg(0)
    def _rand_crop(self, img, cs=None):
        if cs is None: cs = self.size
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape; cs = min(cs, h, w)
            x = random.randint(0, w-cs); y = random.randint(0, h-cs)
            return img[:, y:y+cs, x:x+cs]
        w, h = img.size
        if w < cs or h < cs:
            ratio = cs/min(w,h); img = img.resize((max(cs,int(w*ratio+5)), max(cs,int(h*ratio+5))), Image.BICUBIC)
            w, h = img.size
        x = random.randint(0, w-cs); y = random.randint(0, h-cs)
        return img.crop((x,y,x+cs,y+cs))
    def __getitem__(self, idx):
        bg = self._pick_bg()
        if self.use_gpu_preload:
            ws = int(self.size*(1.2+random.random()*0.3))
            bg_crop = self._rand_crop(bg, ws)
            T_gt = F.interpolate(bg_crop[None], size=(self.size,self.size), mode="bilinear", align_corners=False)[0]
            # Pick texture: prefer GPU-cached, fall back to CPU path
            n_tex = len(self._tex_cpu) if hasattr(self, "_tex_cpu") and self._tex_cpu else (len(self._tex_raw) if hasattr(self, "_tex_raw") else len(self.textures))
            tidx = random.randrange(n_tex)
            tex = self._to_gpu_tex(tidx)
            R_tex = F.interpolate(self._rand_crop(tex, ws)[None], size=(self.size,self.size), mode="bilinear", align_corners=False)[0]
            cos_t = random.uniform(0.17, 1.0)
            gp = dual_pane(cos_t, device=self.device)
            I, Tf, Rc = ReflectionCompositor.apply(T_gt, R_tex, gp, self.device)
            if self.augment: I, Tf, Rc = self._augment(I, Tf, Rc)
            return I, Tf, Rc, torch.tensor(True, device=self.device)
        else:
            tex = random.choice(self.textures)
            ws = int(self.size*(1.2+random.random()*0.3))
            T_gt = self.to_tensor(self._rand_crop(bg, ws))
            R_tex = self.to_tensor(self._rand_crop(tex, ws))
            cos_t = random.uniform(0.17, 1.0)
            gp = dual_pane(cos_t, device=self.device)
            I, Tf, Rc = ReflectionCompositor.apply(T_gt, R_tex, gp, self.device)
            I = F.interpolate(I[None], size=(self.size,self.size), mode="bilinear", align_corners=False)[0]
            Tf = F.interpolate(Tf[None], size=(self.size,self.size), mode="bilinear", align_corners=False)[0]
            Rc = F.interpolate(Rc[None], size=(self.size,self.size), mode="bilinear", align_corners=False)[0]
            if self.augment: I, Tf, Rc = self._augment(I, Tf, Rc)
            return I, Tf, Rc, torch.tensor(True, device=I.device)
    def _augment(self, I, T, R):
        if random.random() < 0.5: I, T, R = I.flip(-1), T.flip(-1), R.flip(-1)
        if random.random() < 0.3: I = (I*random.uniform(0.85,1.15)).clamp(0,1)
        if random.random() < 0.2:
            m = I.mean(dim=[1,2], keepdim=True)
            I = ((I-m)*random.uniform(0.8,1.2)+m).clamp(0,1)
        return I, T, R
    def __len__(self):
        return self.samples
class BenchmarkDataset(Dataset):
    """Unified benchmark dataset loader."""
    def __init__(self, root, size=96, augment=True):
        self.size = size
        self.augment = augment and "train" in root.lower()
        self.pairs = []
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((size,size))
        if not os.path.exists(root):
            print(f"  WARNING: Benchmark not found: {root}")
            return
        for subdir in ["input", "low", "I"]:
            sd = os.path.join(root, subdir)
            if os.path.exists(sd):
                for target_sub in ["target", "gt", "high", "label", "T", "GT"]:
                    td = os.path.join(root, target_sub)
                    if os.path.exists(td):
                        for f in sorted(os.listdir(sd)):
                            if f.lower().endswith((".jpg",".png")):
                                tp = os.path.join(td, f)
                                if os.path.exists(tp):
                                    self.pairs.append((os.path.join(sd,f), tp))
                        break
                break
        print(f"  Benchmark [{os.path.basename(root)}]: {len(self.pairs)} pairs")
    def __len__(self):
        return max(1, len(self.pairs))
    def _load(self, p):
        return self.to_tensor(self.resize(Image.open(p).convert("RGB")))
    def _lazy_tex(self, tidx):
        """Lazy-load a texture tensor to GPU."""
        if not hasattr(self, "_tex_raw"):
            # Cache exists: move CPU tensor to GPU lazily
            if tidx not in self._tex_gpu_cache:
                self._tex_gpu_cache[tidx] = self._tex_cpu[tidx].to(self.device, non_blocking=True)
            return self._tex_gpu_cache[tidx]
        else:
            # No cache: load from PIL on demand
            pil_img = self._tex_raw[tidx]
            gpu_size_cap = max(320, self.size * 2)
            w, h = pil_img.size
            ratio = min(gpu_size_cap/w, gpu_size_cap/h, 1.0)
            if ratio < 1.0: pil_img = pil_img.resize((int(w*ratio), int(h*ratio)), Image.BICUBIC)
            return self.to_tensor(pil_img).to(self.device)
    def __getitem__(self, idx):
        if not self.pairs:
            d = torch.zeros(3,self.size,self.size)
            return d,d,torch.zeros_like(d),torch.tensor(False)
        idx = idx % len(self.pairs)
        I = self._load(self.pairs[idx][0])
        T = self._load(self.pairs[idx][1])
        R = torch.zeros_like(T)
        return I, T, R, torch.tensor(False)
if __name__ == "__main__":
    ds = HighQualityWindowDataset(
        real_label_dirs=["dataset/train"],
        texture_dirs=["datasets/reflections", "datasets/reflection_textures"],
        scene_dir="datasets/scenes", size=96, samples_per_epoch=100, use_cuda=False)
    t0 = time.time()
    I, T, R, hr = next(iter(ds))
    t1 = time.time()
    print(f"Sample: {I.shape}, I=[{I.min():.3f},{I.max():.3f}], {t1-t0:.2f}s")
    times, im, tm, rm = [], [], [], []
    for i, (I,T,R,hr) in enumerate(ds):
        if i == 0: continue
        dt = time.time()-t0
        times.append(dt)
        im.append(I.mean().item())
        tm.append(T.mean().item())
        rm.append(R.mean().item())
        t0 = time.time()
        if i >= 49: break
    avg_t = float(np.mean(times)) if times else 0
    print(f"\n50 samples: {avg_t:.3f}s/sample ({1/max(0.001,avg_t):.0f}/s)")
    if im: print(f"  I  mean: {float(np.mean(im)):.3f}+-{float(np.std(im)):.3f}")
    if tm: print(f"  T  mean: {float(np.mean(tm)):.3f}+-{float(np.std(tm)):.3f}")
    if rm: print(f"  R  mean: {float(np.mean(rm)):.3f}+-{float(np.std(rm)):.3f}")

