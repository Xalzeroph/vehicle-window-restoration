"""Cutting-edge augmentation pipeline for reflection + low-light data."""
import random
import torch



class WindowAugmentation:
    """
    Task-specific augmentations for car window images.
    Preserves paired consistency (same transform on I and T_gt).
    """

    @staticmethod
    def hflip(I, T_gt, R_gt):
        if random.random() < 0.5:
            return I.flip(-1), T_gt.flip(-1), R_gt.flip(-1)
        return I, T_gt, R_gt

    @staticmethod
    def vflip(I, T_gt, R_gt):
        if random.random() < 0.3:
            return I.flip(-2), T_gt.flip(-2), R_gt.flip(-2)
        return I, T_gt, R_gt

    @staticmethod
    def rotate90(I, T_gt, R_gt):
        if random.random() < 0.2:
            k = random.choice([1, 2, 3])
            return torch.rot90(I, k, [-2, -1]), torch.rot90(T_gt, k, [-2, -1]), torch.rot90(R_gt, k, [-2, -1])
        return I, T_gt, R_gt

    @staticmethod
    def brightness_jitter(I, T_gt, R_gt):
        """Simulate varying lighting conditions."""
        if random.random() < 0.4:
            factor = random.uniform(0.7, 1.3)
            I = (I * factor).clamp(0, 1)
        return I, T_gt, R_gt

    @staticmethod
    def contrast_jitter(I, T_gt, R_gt):
        """Simulate different glass/weather conditions."""
        if random.random() < 0.3:
            mean = I.mean(dim=[1, 2], keepdim=True)
            factor = random.uniform(0.8, 1.2)
            I = ((I - mean) * factor + mean).clamp(0, 1)
        return I, T_gt, R_gt

    @staticmethod
    def reflection_intensity(I, T_gt, R_gt):
        """Vary reflection strength to improve robustness."""
        if random.random() < 0.5:
            scale = random.uniform(0.5, 1.5)
            R_scaled = (R_gt * scale).clamp(0, 1)
            I = (T_gt + R_scaled).clamp(0, 1)
        return I, T_gt, R_gt

    @staticmethod
    def gaussian_noise(I, T_gt, R_gt):
        """Sensor noise simulation."""
        if random.random() < 0.3:
            noise = torch.randn_like(I) * random.uniform(0.002, 0.02)
            I = (I + noise).clamp(0, 1)
        return I, T_gt, R_gt

    @staticmethod
    def cutmix(I, T_gt, R_gt):
        """CutMix: paste a random patch from another input sample within batch.
        Performed at batch level, not individual sample.
        """
        return I, T_gt, R_gt  # batch-level, handled in collate

    @classmethod
    def apply_all(cls, I, T_gt, R_gt):
        """Apply all augmentations. Each has independent probability."""
        for fn in [cls.hflip, cls.vflip, cls.rotate90,
                   cls.brightness_jitter, cls.contrast_jitter,
                   cls.reflection_intensity, cls.gaussian_noise]:
            I, T_gt, R_gt = fn(I, T_gt, R_gt)
        return I, T_gt, R_gt