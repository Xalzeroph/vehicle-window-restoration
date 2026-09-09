#!/usr/bin/env python3
"""ExpoMamba 低光增强推理脚本。

Usage:
    python scripts/expo_mamba_inference.py /path/to/images [output_dir]
"""
import argparse
import os
import sys
import warnings

import cv2
import numpy as np
import torch

# 将 ExpoMamba 目录加入 sys.path，保证内部相对导入正确
EXPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ExpoMamba")
sys.path.insert(0, EXPO_DIR)

from models.expomamba_2d import ExpoMamba


def load_model(checkpoint_path, device="cuda"):
    model = ExpoMamba(n_channels=3, bilinear=True, base_channels=48, deep_supervision=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def enhance_image(model, img_bgr, device="cuda"):
    """输入 BGR (H,W,3)，输出 BGR (H,W,3)"""
    h, w = img_bgr.shape[:2]

    # 对齐到 32 的倍数（VSSM patch_embed 需要）
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(tensor)

    output = output[:, :, :h, :w]
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output_bgr


def main():
    parser = argparse.ArgumentParser(description="ExpoMamba Low-Light Enhancement")
    parser.add_argument("input_dir", help="输入图片目录")
    parser.add_argument("output_dir", nargs="?", default="./expo_output", help="输出目录")
    parser.add_argument("--checkpoint", default=None, help="ExpoMamba 权重路径")
    parser.add_argument("--gpu", default="0", help="GPU ID")
    parser.add_argument("--ext", default=".png,.jpg,.jpeg,.bmp,.tif,.tiff", help="图片扩展名")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.checkpoint is None:
        args.checkpoint = os.path.join(project_root, "pretrained", "expomamba", "best_ssim.pth")
    if not os.path.isabs(args.input_dir):
        args.input_dir = os.path.join(project_root, args.input_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, device)
    exts = set(args.ext.lower().split(","))

    files = sorted([
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not files:
        print(f"[ExpoMamba] 未找到图片文件: {args.input_dir}")
        return

    print(f"[ExpoMamba] 处理 {len(files)} 张图片...")
    print(f"  输入: {args.input_dir}")
    print(f"  输出: {args.output_dir}")

    for i, fname in enumerate(files):
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)

        img = cv2.imread(in_path)
        if img is None:
            print(f"  [{i+1}/{len(files)}] 跳过 {fname} — 无法读取")
            continue

        enhanced = enhance_image(model, img, device)
        cv2.imwrite(out_path, enhanced)
        print(f"  [{i+1}/{len(files)}] {fname}")

    print(f"[ExpoMamba] 完成 — {args.output_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
