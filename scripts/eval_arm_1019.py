#!/usr/bin/env python3
"""One canonical tiled evaluator for native paired RDNet/DARL image manifests."""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

from finetune_rdnet_peft import build


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def psnr(predicted, target):
    mse = ((predicted.astype(np.float64) - target.astype(np.float64)) ** 2).mean(axis=(0, 1))
    return float(np.mean(10.0 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12))))


def ssim(predicted, target):
    return float(np.mean([
        structural_similarity(predicted[:, :, c], target[:, :, c], data_range=255)
        for c in range(3)
    ]))


def tiled_transmission(net, classifier, image, device, tile, overlap):
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).float().div_(255)
    height, width = tensor.shape[-2:]
    tensor = F.pad(tensor, (0, max(0, tile - width), 0, max(0, tile - height)),
                   mode="reflect")
    padded_h, padded_w = tensor.shape[-2:]
    stride = tile - overlap

    def starts(length):
        positions = list(range(0, max(1, length - tile + 1), stride))
        last = length - tile
        if positions[-1] != last:
            positions.append(last)
        return positions

    taper = torch.hann_window(tile, periodic=False).clamp_min(1e-2)
    weight = taper[:, None] * taper[None, :]
    output = torch.zeros((1, 3, padded_h, padded_w), dtype=torch.float32)
    weight_sum = torch.zeros((1, 1, padded_h, padded_w), dtype=torch.float32)
    for top in starts(padded_h):
        for left in starts(padded_w):
            patch = tensor[:, :, top:top + tile, left:left + tile].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                alpha = classifier(patch)
                transmission = net(patch, alpha)[1][-1][:, 3:].float().cpu()
            output[:, :, top:top + tile, left:left + tile] += transmission * weight
            weight_sum[:, :, top:top + tile, left:left + tile] += weight
    return (output / weight_sum)[:, :, :height, :width].clamp_(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True, help="'official' or complete DARL teacher checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile", type=int, default=256, help="Inference tile, multiple of 32")
    parser.add_argument("--overlap", type=int, default=64, help="Overlapping pixels for Hann blending")
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as src:
        items = json.load(src)["items"]
    root = os.path.dirname(os.path.abspath(args.manifest))
    os.makedirs(args.out, exist_ok=True)

    net, classifier = build(mona="none" if args.checkpoint == "official" else "both",
                            save_memory=False, grad_ckpt=False)
    if args.checkpoint != "official":
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        net.load_state_dict(weights["icnn"], strict=True)
    device = torch.device(args.device)
    net.to(device).eval()
    classifier.to(device).eval()

    records = []
    with torch.inference_mode():
        for entry in items:
            observed = load_image(os.path.join(root, entry["input"]))
            target = load_image(os.path.join(root, entry["target"]))
            prediction = tiled_transmission(net, classifier, observed, device,
                                            args.tile, args.overlap)
            prediction = prediction.mul_(255).round().to(torch.uint8)[0].permute(1, 2, 0).numpy()
            row = {
                "id": entry["id"],
                "source_t_key": entry["source_t_key"],
                "psnr": psnr(prediction, target),
                "ssim": ssim(prediction, target),
                "identity_psnr": psnr(observed, target),
                "identity_ssim": ssim(observed, target),
            }
            records.append(row)
            print(f"[eval] {len(records)}/{len(items)} {row['id']} PSNR={row['psnr']:.3f}", flush=True)

    keys = ("psnr", "ssim", "identity_psnr", "identity_ssim")
    groups = {}
    for row in records:
        groups.setdefault(row["source_t_key"], []).append(row)
    aggregate = lambda rows: {key: float(np.mean([v[key] for v in rows])) for key in keys}
    summary = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "tile": args.tile,
        "overlap": args.overlap,
        "n_images": len(records),
        "n_sources": len(groups),
        "per_image": aggregate(records),
        "per_scene": aggregate([aggregate(rows) for rows in groups.values()]),
    }
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as dst:
        json.dump(summary, dst, ensure_ascii=False, indent=2)
    print("[eval] final:", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
