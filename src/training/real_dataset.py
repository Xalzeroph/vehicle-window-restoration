"""Real paired dataset with train/val/test video-level split.
8 train + 2 val + 2 test scenes held out completely for honest evaluation.
R_gt = (I - T).clamp(0,1) but NOT supervised for real data (use supervise_r=False in compute_loss).
"""
import os, random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.training.augmentation import WindowAugmentation


def _split_scenes(root, val_count=2, test_count=2):
    """Split scenes into train/val/test by scene name (stable hash)."""
    scenes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    # Use deterministic hash for reproducibility
    hashed = sorted(scenes, key=lambda s: (sum(ord(c) for c in s), s))
    test = set(hashed[-test_count:]) if test_count > 0 else set()
    val = set(hashed[-(test_count+val_count):-test_count]) if val_count > 0 else set()
    train = set(hashed[:-(test_count+val_count)]) if (test_count+val_count) > 0 else set(hashed)
    print(f"  Split: {len(train)} train + {len(val)} val + {len(test)} test (total {len(scenes)} scenes)")
    return train, val, test


class RealPairedDataset(Dataset):
    """Scene-level paired data with explicit split."""

    def __init__(self, root, size=256, split="train", val_count=2, test_count=2,
                 augment=True, device=None):
        self.size = size
        self.augment = augment and split == "train"
        self.device = torch.device(device) if device is not None else torch.device("cuda", index=torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((size, size))

        train_scenes, val_scenes, test_scenes = _split_scenes(root, val_count, test_count)
        if split == "train":
            allowed = train_scenes
        elif split == "val":
            allowed = val_scenes
        elif split == "test":
            allowed = test_scenes
        else:
            raise ValueError(f"Unknown split: {split}")

        self.pairs = []
        for scene in sorted(allowed):
            scene_dir = os.path.join(root, scene)
            label_path = os.path.join(scene_dir, "label.jpg")
            input_dir = os.path.join(scene_dir, "input")
            if not os.path.exists(label_path) or not os.path.exists(input_dir):
                continue
            for fn in os.listdir(input_dir):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.pairs.append((os.path.join(input_dir, fn), label_path, scene))

        print(f"  Real {split.upper()}: {len(self.pairs)} pairs from {len(allowed)} scenes")

    def __len__(self):
        return max(1, len(self.pairs))

    def _load(self, path):
        return self.to_tensor(self.resize(Image.open(path).convert("RGB")))

    def __getitem__(self, idx):
        idx = idx % len(self.pairs)
        input_path, label_path, scene = self.pairs[idx]
        I = self._load(input_path).to(self.device)
        T_gt = self._load(label_path).to(self.device)
        R_gt = (I - T_gt).clamp(0, 1)
        has_refl = torch.tensor(True, device=self.device)
        if self.augment:
            I, T_gt, R_gt = WindowAugmentation.apply_all(I, T_gt, R_gt)
            R_gt = (I - T_gt).clamp(0, 1)
        return I, T_gt, R_gt, has_refl

    @staticmethod
    def create_datasets(root, size, val_count=2, test_count=2):
        """Create train + val + test datasets."""
        ds_train = RealPairedDataset(root, size, "train", val_count, test_count, augment=True)
        ds_val = RealPairedDataset(root, size, "val", val_count, test_count, augment=False)
        ds_test = RealPairedDataset(root, size, "test", val_count, test_count, augment=False)
        return ds_train, ds_val, ds_test