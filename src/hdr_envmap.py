#!/usr/bin/env python
"""HDR Environment Map Reflection Rendering.
Provides physically-based spherical reflections using HDR equirectangular maps.
Replaces simple 2D texture overlay with curved, view-dependent reflections.
"""

import os, random, math
import torch
import torch.nn.functional as F
import torch

def load_hdr(path):
    """Load .hdr file as torch tensor [3, H, W]."""
    import imageio.v2 as imageio
    hdr = imageio.imread(path, format="HDR-FI")
    return torch.from_numpy(hdr).permute(2,0,1).float()

def sample_envmap(envmap, normal, roughness=0.0):
    """Sample HDR environment map given surface normal direction.
    envmap: [3, H, W] equirectangular HDR
    normal: [B, 3] unit surface normals (view-dependent)
    returns [B, 3] RGB color
    """
    B, C, H, W = envmap.unsqueeze(0).shape
    # Equirectangular mapping: normal (x,y,z) -> (phi, theta) -> (u, v)
    x, y, z = normal[:, 0], normal[:, 1], normal[:, 2]
    phi = torch.atan2(z, x)  # [-pi, pi]
    theta = torch.asin(y.clamp(-1, 1))  # [-pi/2, pi/2]
    u = (phi / math.pi + 1) / 2  # [0,1]
    v = (theta / (math.pi/2) + 1) / 2  # [0,1]
    # Grid sample
    grid = torch.stack([u*2-1, v*2-1], dim=1).unsqueeze(1)  # [B,1,1,2]
    sampled = F.grid_sample(
        envmap.unsqueeze(0).expand(B, -1, -1, -1),
        grid, mode="bilinear", align_corners=False
    )  # [B,3,1,1]
    return sampled.squeeze(-1).squeeze(-1)  # [B,3]

def render_spherical_reflection(T_gt, envmap_hdr, intensity=0.3):
    """Render spherical reflection on flat surface.
    Approximates curved windshield reflection by warping the envmap.
    T_gt: [B,3,H,W] clean transmission
    envmap_hdr: [3,HE,WE] HDR environment map
    returns [B,3,H,W] reflection layer
    """
    B, C, H, W = T_gt.shape
    # Create surface normal grid (approximate curved windshield)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=T_gt.device),
        torch.linspace(-1, 1, W, device=T_gt.device),
        indexing="ij"
    )
    # Parabolic curvature: z = 1 - k*(x^2+y^2)
    k_curv = 0.15 + random.random() * 0.1
    zz = 1 - k_curv * (xx**2 + yy**2)
    normals = F.normalize(torch.stack([xx, yy, zz], dim=0), dim=0)  # [3,H,W]
    
    # Sample envmap for each pixel
    normals_flat = normals.reshape(3, -1).T  # [H*W, 3]
    e, h_env, w_env = envmap_hdr.shape
    # Use grid_sample for efficient sampling
    phi = torch.atan2(normals_flat[:, 2], normals_flat[:, 0])
    theta = torch.asin(normals_flat[:, 1].clamp(-1, 1))
    u = (phi / math.pi + 1) / 2
    v = (theta / (math.pi/2) + 1) / 2
    grid = torch.stack([u*2-1, v*2-1], dim=1).view(1, H, W, 2)  # [1,H,W,2]
    
    R = F.grid_sample(
        envmap_hdr.unsqueeze(0), grid,
        mode="bilinear", align_corners=False
    )  # [1,3,H,W]
    
    # Apply intensity and gamma
    intensity_val = random.uniform(0.15, 0.45)
    R = R * intensity_val
    # Apply defocus (curved glass blur)
    k = random.choice([3, 5, 7])
    blur = torch.ones(1,1,k,k).to(R.device) / (k*k)
    R = F.conv2d(R, blur.repeat(3,1,1,1), groups=3, padding=k//2)
    
    return R.clamp(0, 1)

class HDRReflectionCompositor:
    """HDR-based reflection compositing. Provides more realistic reflections
    than simple 2D texture overlay by modeling view-dependent spherical reflection."""
    
    def __init__(self, hdr_dir="datasets/hdr_envmaps"):
        self.hdr_dir = hdr_dir
        self._envmaps = {}
        self._scan()
    
    def _scan(self):
        if not os.path.exists(self.hdr_dir):
            return
        for f in sorted(os.listdir(self.hdr_dir)):
            if f.endswith(".hdr"):
                path = os.path.join(self.hdr_dir, f)
                self._envmaps[f] = path
        print(f"HDRReflectionCompositor: {len(self._envmaps)} envmaps")
    
    def apply(self, T_gt, device="cpu"):
        """Composite HDR reflection onto clean background.
        T_gt: [3,H,W] clean transmission
        returns (I, R): [3,H,W] composited image and reflection
        """
        T_gt = T_gt.to(device)
        if not self._envmaps:
            return T_gt, torch.zeros_like(T_gt)
        # Pick random envmap
        key = random.choice(list(self._envmaps.keys()))
        envmap = self._envmaps[key]
        if isinstance(envmap, str):
            envmap = load_hdr(envmap).to(device)
            self._envmaps[key] = envmap
        R = render_spherical_reflection(T_gt.unsqueeze(0), envmap, intensity=0.3)
        I = (T_gt + R[0]).clamp(0, 1)
        return I, R[0]

    def __len__(self):
        return max(1, len(self._envmaps))

if __name__ == "__main__":
    # Test
    import time
    T = torch.rand(3, 96, 96)
    comp = HDRReflectionCompositor("datasets/hdr_envmaps")
    t0 = time.time()
    I, R = comp.apply(T)
    t1 = time.time()
    print(f"HDRComposite: T={T.shape}, I range=[{I.min():.3f},{I.max():.3f}] ({t1-t0:.3f}s)")