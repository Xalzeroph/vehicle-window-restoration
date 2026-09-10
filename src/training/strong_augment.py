"""StrongAugment ? cutting-edge data augmentation for joint low-light + reflection removal.
Combines proven techniques from image restoration, low-light enhancement, and reflection separation.
"""
import random
import torch
import torch.nn.functional as F
import numpy as np

class StrongAugment:
    """GPU-accelerated data augmentation pipeline.
    Applied AFTER dataloader, on batched GPU tensors.
    Compatible with both synthetic and real data.
    """
    
    @staticmethod
    @torch.no_grad()
    def apply(I, T, R, has_refl):
        """Apply augmentation pipeline to a batch.
        Returns augmented (I, T, R, has_refl).
        I: [B,3,H,W] input
        T: [B,3,H,W] target transmission  
        R: [B,3,H,W] target reflection
        has_refl: [B] bool flag
        """
        B = I.shape[0]
        
        # 1. Random Erasing (CutOut) ? per-sample
        I = StrongAugment._erasing(I, p=0.25)
        
        # 2. Strong color jitter ? per-sample
        I = StrongAugment._color_jitter(I, p=1.0)
        
        # 3. Gaussian noise (input only) ? per-sample  
        I = StrongAugment._gaussian_noise(I, p=0.15)
        
        # 4. Gaussian blur (input only) ? per-sample
        I = StrongAugment._gaussian_blur(I, p=0.10)
        
        # 5. MixUp ? batch-level
        I, T, R, has_refl = StrongAugment._mixup(I, T, R, has_refl, alpha=0.2, p=0.5)
        
        return I, T, R, has_refl
    
    @staticmethod
    def _erasing(I, p=0.25):
        """Random Erasing / CutOut: batch mask via random rectangles."""
        B, C, H, W = I.shape
        if random.random() >= p:
            return I
        # Generate random rectangles for all samples in batch
        eh = torch.randint(H//10, H//3+1, (B,), device=I.device)
        ew = torch.randint(W//10, W//3+1, (B,), device=I.device)
        x = torch.randint(0, W, (B,), device=I.device)
        y = torch.randint(0, H, (B,), device=I.device)
        x = x - (ew // 2)  # center the rectangle
        y = y - (eh // 2)
        x = x.clamp(0, W-1)
        y = y.clamp(0, H-1)
        fill = torch.rand(B, C, 1, 1, device=I.device) * 0.3 + 0.1
        I_out = I.clone()
        for b in range(B):
            x1, x2 = int(x[b]), int(min(x[b]+ew[b], W))
            y1, y2 = int(y[b]), int(min(y[b]+eh[b], H))
            if x2 > x1 and y2 > y1:
                I_out[b:b+1, :, y1:y2, x1:x2] = fill[b:b+1]
        return I_out
    
    @staticmethod
    def _color_jitter(I, p=1.0):
        """Strong color augmentation: batch color transforms on GPU."""
        B, C, H, W = I.shape
        if random.random() > p:
            return I
        I_out = I.clone()
        # Brightness: random per sample [0.6, 1.4]
        brightness = torch.empty(B, 1, 1, 1, device=I.device).uniform_(0.6, 1.4)
        I_out = I_out * brightness
        # Contrast: random per sample [0.6, 1.4]
        contrast = torch.empty(B, 1, 1, 1, device=I.device).uniform_(0.6, 1.4)
        m = I_out.mean(dim=[2,3], keepdim=True)
        I_out = (I_out - m) * contrast + m
        # Saturation: random per sample [0.7, 1.3], 50% chance
        if random.random() < 0.5:
            gray = I_out.mean(dim=1, keepdim=True)
            sat = torch.empty(B, 1, 1, 1, device=I.device).uniform_(0.7, 1.3)
            I_out = I_out * sat + gray * (1 - sat)
        # Color cast: random shift, 30% chance
        if random.random() < 0.3:
            shift = torch.randn(B, C, 1, 1, device=I.device) * 0.05
            I_out = (I_out + shift).clamp(0, 1)
        return I_out.clamp(0, 1)
    
    @staticmethod
    def _gaussian_noise(I, p=0.15):
        """Add Gaussian noise to input."""
        if random.random() < p:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(I) * noise_std
            I = (I + noise).clamp(0, 1)
        return I
    
    @staticmethod
    def _gaussian_blur(I, p=0.10):
        """Gaussian blur with random kernel size (direct computation, no cache)."""
        B, C, H, W = I.shape
        if not (random.random() < p and H >= 8 and W >= 8):
            return I
        k = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 2.0)
        grid = torch.arange(k, device=I.device).float() - k//2
        kernel_1d = torch.exp(-grid**2 / (2 * sigma**2 + 1e-8))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:,None] @ kernel_1d[None,:]
        kernel = kernel_2d[None, None, :, :].expand(C, 1, k, k).contiguous()
        pad = k//2
        I_pad = F.pad(I, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(I_pad, kernel, groups=C, padding=0)

    @staticmethod
    def _mixup(I, T, R, has_refl, alpha=0.2, p=0.5):
        """MixUp: convex combination of two random samples.
        Only applied when has_refl matches (both synthetic or both real).
        """
        B = I.shape[0]
        if B < 2 or random.random() > p:
            return I, T, R, has_refl
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(B, device=I.device)
        # Mix I and T
        I = lam * I + (1-lam) * I[idx]
        T = lam * T + (1-lam) * T[idx]
        # Mix R only when both samples have valid reflection ground truth
        refl_mask = has_refl & has_refl[idx]  # [B] bool
        if refl_mask.any():
            R_new = R.clone()
            rm = refl_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(R)
            R_new[rm] = (lam * R + (1-lam) * R[idx])[rm]
            R = R_new
        # has_refl stays True if either original was True (conservative)
        has_refl = has_refl | has_refl[idx]
        return I, T, R, has_refl


