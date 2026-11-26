#!/usr/bin/env python3
"""Single authority for paired-manifest RDNet teacher evaluation."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True, help="'official' or complete RDNet teacher checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
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
            folder = os.path.join(root, entry["relative_directory"])
            observed = load_image(os.path.join(folder, "observed.png"))
            target = load_image(os.path.join(folder, "target.png"))
            tensor = torch.from_numpy(observed.transpose(2, 0, 1).copy()).unsqueeze(0).to(
                device=device, dtype=torch.float32
            ).div_(255.0)
            height, width = tensor.shape[-2:]
            pad = (0, (-width) % 32, 0, (-height) % 32)
            if any(pad):
                tensor = F.pad(tensor, pad, mode="reflect")
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                alpha = classifier(tensor)
                transmission = net(tensor, alpha)[1][-1][:, 3:]
            prediction = transmission[:, :, :height, :width].float().clamp_(0, 1)
            prediction = prediction.mul_(255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
            row = {
                "id": entry["id"],
                "source_t_key": entry["source_t_key"],
                "film_level": entry["film_level"],
                "psnr": psnr(prediction, target),
                "ssim": ssim(prediction, target),
                "identity_psnr": psnr(observed, target),
                "identity_ssim": ssim(observed, target),
            }
            records.append(row)
            print(f"[eval] {len(records)}/{len(items)} {row['id']} PSNR={row['psnr']:.3f}", flush=True)

    keys = ("psnr", "ssim", "identity_psnr", "identity_ssim")
    scenes = {}
    for row in records:
        scenes.setdefault(row["source_t_key"], []).append(row)
    aggregate = lambda rows: {key: float(np.mean([v[key] for v in rows])) for key in keys}
    summary = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "n_images": len(records),
        "n_sources": len(scenes),
        "per_image": aggregate(records),
        "per_scene": aggregate([aggregate(rows) for rows in scenes.values()]),
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
