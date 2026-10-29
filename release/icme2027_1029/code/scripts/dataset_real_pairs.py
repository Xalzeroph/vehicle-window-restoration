#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实配对数据集读取（2026-10-20）。

支持两种 meta 形式
------------------
1. **dict 型**（`pairs_*` / `real_pairs_1020`）：`{case: {domain, scene, ...}}`，
   文件在 `I/ T/ R/ omega/` 下，名为 `{case}.png`
2. **list 型**（opensrc 池）：`[{"I": abs, "T": abs, "R": abs, "omega": abs, ...}]`

接口与 `finetune_rdnet_tint.py::TintDataset` 完全一致：`__getitem__` 返回
`(I, T, R, A)`，形状 `(3,H,W) / (3,H,W) / (3,H,W) / (1,H,W)`，值域 [0,1]，
R 从 +128 存储还原为**有符号**并裁到 [-1,1]（与 training 侧 charbonnier 口径一致）。

设计约束
--------
* `omega_frac` 过小的样本（默认 < 0.005）**跳过**：无反射区样本对去反射无监督价值；
* `--cap-per-scene` 之外还支持**按场景分层采样**：`__getitem__` 先均匀抽场景再抽该场景的样本，
  避免"GT 多帧的场景"主导整个 epoch（一 GT 配多 I 的固有风险）；
* 尺寸不统一的图在读取时统一 resize 到 `base_size`（默认不缩放，靠随机裁剪）。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MIN_OMEGA = 0.005


def _read_rgb(p: str) -> np.ndarray:
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)


class RealPairDataset(Dataset):
    """真实配对（I,T,R,omega）。与 TintDataset 接口一致，可直接换进现有训练循环。"""

    def __init__(self, root: str, size_min: int = 224, size_max: int = 320,
                 min_omega: float = MIN_OMEGA, scene_balanced: bool = True,
                 seed: int | None = None):
        self.root = root
        self.size = size_min
        self.smin, self.smax = size_min, size_max
        self.rng = np.random.default_rng(seed)

        with open(os.path.join(root, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)

        items = []
        if isinstance(meta, dict):
            for case, info in meta.items():
                om_f = info.get("omega_frac")
                if om_f is not None and float(om_f) < min_omega:
                    continue
                items.append({
                    "case": case,
                    "scene": info.get("scene", case),
                    "I": os.path.join(root, "I", case + ".png"),
                    "T": os.path.join(root, "T", case + ".png"),
                    "R": os.path.join(root, "R", case + ".png"),
                    "omega": os.path.join(root, "omega", case + ".png"),
                })
        elif isinstance(meta, list):
            for info in meta:
                if info.get("omega_frac") is not None and float(info["omega_frac"]) < min_omega:
                    continue
                items.append({
                    "case": info.get("name") or info.get("case") or str(len(items)),
                    "scene": info.get("scene", info.get("name", "na")),
                    "I": info["I"], "T": info["T"],
                    "R": info["R"], "omega": info["omega"],
                })
        else:
            raise TypeError(f"meta.json 类型不支持: {type(meta)}")

        assert items, f"空数据集（检查 {root} 与 min_omega={min_omega}）"
        self.items = items

        # 按场景分组（用于分层采样，防单场景主导）
        self.by_scene: dict[str, list[int]] = {}
        for i, it in enumerate(items):
            self.by_scene.setdefault(it["scene"], []).append(i)
        self.scenes = sorted(self.by_scene)
        self.scene_balanced = scene_balanced

    def __len__(self) -> int:
        return len(self.items)

    def _pick(self) -> int:
        if not self.scene_balanced:
            return int(self.rng.integers(len(self.items)))
        s = self.scenes[int(self.rng.integers(len(self.scenes)))]
        idxs = self.by_scene[s]
        return int(idxs[int(self.rng.integers(len(idxs)))])

    def __getitem__(self, i: int):
        if self.scene_balanced:
            # 注意：传入 i 时仍按"训练循环给的序号"取，但内部随机选场景
            # （scene_balanced 打开时 i 只用于满足 DataLoader 的长度需求）
            it = self.items[self._pick()]
        else:
            it = self.items[i]

        I = _read_rgb(it["I"]) / 255.0
        T = _read_rgb(it["T"]) / 255.0
        R = (_read_rgb(it["R"]) - 128.0) / 255.0          # 还原有符号
        A = cv2.imread(it["omega"], cv2.IMREAD_GRAYSCALE)
        if A is None:
            A = np.zeros(I.shape[:2], np.uint8)
        A = (A.astype(np.float32) / 255.0)[None]

        s = self.size
        H, W = I.shape[:2]
        if H < s or W < s:                                  # 小图先放大
            sc = s / min(H, W)
            nh, nw = int(round(H * sc)), int(round(W * sc))
            I = cv2.resize(I, (nw, nh)); T = cv2.resize(T, (nw, nh))
            R = cv2.resize(R, (nw, nh)); A = cv2.resize(A[0], (nw, nh))[None]
            H, W = nh, nw
        y0 = int(self.rng.integers(0, H - s + 1))
        x0 = int(self.rng.integers(0, W - s + 1))
        I = I[y0:y0 + s, x0:x0 + s]; T = T[y0:y0 + s, x0:x0 + s]
        R = R[y0:y0 + s, x0:x0 + s]; A = A[:, y0:y0 + s, x0:x0 + s]
        if self.rng.random() < 0.5:
            I = I[:, ::-1]; T = T[:, ::-1]; R = R[:, ::-1]; A = A[:, :, ::-1]

        # I/T/R 是 HWC → CHW；A 已是 (1,H,W)，**不能**再转置
        # （曾因统一走 transpose 得到 (H,1,W)，与训练侧 alpha 口径不符）
        def _chw(x):
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

        return _chw(I), _chw(T), _chw(R), torch.from_numpy(np.ascontiguousarray(A))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/train_sets/real_pairs_1020")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    ds = RealPairDataset(args.root, size_min=args.size, size_max=args.size)
    print(f"样本 {len(ds)}，场景 {len(ds.scenes)}，分层采样={ds.scene_balanced}")
    I, T, R, A = ds[0]
    print("I", tuple(I.shape), "T", tuple(T.shape), "R", tuple(R.shape), "A", tuple(A.shape))
    print("I range", float(I.min()), float(I.max()),
          "| R range", float(R.min()), float(R.max()),
          "| A mean", float(A.mean()))
