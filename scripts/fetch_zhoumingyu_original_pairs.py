#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image

REPO = "zmy1234567890/tintedglass-rdt-synth"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/train/rdt_0915"

def fetch_one(job):
    path, root = job
    target = root / path
    if target.is_file() and target.stat().st_size > 0:
        return "existing", path
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(14):
        temp = target.with_name(target.name + f".{os.getpid()}.part")
        try:
            with requests.get(f"{BASE}/{path}", stream=True, timeout=(12, 75)) as response:
                if response.status_code == 429:
                    wait = max(int(response.headers.get("Retry-After", "0") or 0), min(120, 8*(attempt+1)))
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                with temp.open("wb") as out:
                    for chunk in response.iter_content(1 << 18):
                        if chunk:
                            out.write(chunk)
            with Image.open(temp) as image:
                if image.size != (448, 448):
                    raise ValueError(f"Unexpected original image size: {image.size}")
                image.verify()
            os.replace(temp, target)
            return "downloaded", path
        except Exception as exc:
            temp.unlink(missing_ok=True)
            if attempt == 13:
                return "failed", f"{path}: {exc}"
            time.sleep(min(90, 3 * (attempt+1)**2))
    return "failed", f"{path}: rate-limited after all retries"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    manifest = json.loads((args.root / "meta.json").read_text())
    response=requests.get(f"https://hf-mirror.com/api/datasets/{REPO}", timeout=45)
    response.raise_for_status()
    published={x["rfilename"] for x in response.json()["siblings"]}
    prefix="train/rdt_0915/"
    expected=[f"T/{scene['scene_id']}.png" for scene in manifest["scenes"]]
    expected += [f"I/{sample['sid']}.png" for sample in manifest["samples"]]
    unpublished=[p for p in expected if prefix+p not in published and not (args.root/p).is_file()]
    paths=[p for p in expected if prefix+p in published or (args.root/p).is_file()]
    remaining = [(path, args.root) for path in paths if not (args.root / path).is_file()]
    print("ZHOU_ORIGINAL_FETCH_PLAN", json.dumps({
        "declared_scenes": len(manifest["scenes"]), "declared_pairs": len(manifest["samples"]),
        "published_or_cached_files": len(paths), "unpublished_files": len(unpublished),
        "files_to_download": len(remaining), "source_repo": REPO
    }), flush=True)
    counts = {"downloaded": 0, "existing": len(paths) - len(remaining), "failed": 0}
    failures = []
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(fetch_one, remaining), 1):
            status, path = result
            counts[status] += 1
            if status == "failed":
                failures.append(path)
            if i % 200 == 0 or i == len(remaining):
                print("ZHOU_ORIGINAL_FETCH_PROGRESS", json.dumps({
                    "processed": i, "total": len(remaining), **counts,
                    "elapsed_seconds": round(time.monotonic()-start, 1),
                    "last_failure": failures[-1] if failures else None
                }), flush=True)
    missing = [str(path) for path in paths if not (args.root / path).is_file()]
    available_input={p.stem for p in (args.root/"I").glob("*.png")}
    available_target={p.stem for p in (args.root/"T").glob("*.png")}
    native_pairs=sum(s["sid"] in available_input and s["scene_id"] in available_target
                     for s in manifest["samples"])
    result = {"repo": REPO, "declared_pairs": len(manifest["samples"]),
              "available_native_pairs": native_pairs,
              "unpublished_files": len(unpublished),
              "missing_published_files": len(missing),
              "failures": failures[:50], **counts}
    print("ZHOU_ORIGINAL_FETCH_DONE", json.dumps(result), flush=True)
    if missing:
        raise SystemExit(f"Published original-data image files still missing: {len(missing)}")

if __name__ == "__main__":
    main()
