#!/usr/bin/env python
"""Download natural scene images for background diversity.
Usage: python src/prepare_scenes.py --count 2000 --out datasets/scenes
"""

import os, io, urllib.request, random, argparse
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Unsplash API: free 50 req/hr without key, 5000/hr with developer key
# Fallback: use built-in publicly available URLs

FALLBACK_URLS = [
    "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=640",  # nature
    "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=640",
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=640",
    "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=640",
    "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=640",
    "https://images.unsplash.com/photo-1475924156734-496f6cac6ec1?w=640",
    "https://images.unsplash.com/photo-1518173946687-a36f968f7b9a?w=640",
    "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=640",
    "https://images.unsplash.com/photo-1511795409834-ef04bbd61622?w=640",
    "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=640",
    "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=640",
    "https://images.unsplash.com/photo-1439853949127-fa647821eba0?w=640",
    "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=640",
    "https://images.unsplash.com/photo-1440581553064-46bc1392954a?w=640",
    "https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?w=640",
    "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=640",
    "https://images.unsplash.com/photo-1518173946687-a36f968f7b9a?w=640",
    "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=640",
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=640",
    "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=640",
]


def download_one(url, out_path, idx, total):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
            img.save(out_path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"  [{idx}/{total}] FAIL: {url[:50]} -> {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=2000, help="Target scene count")
    p.add_argument("--out", type=str, default="datasets/scenes",
                   help="Output directory (e.g. datasets/scenes)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    if existing >= args.count:
        print(f"Already have {existing} scenes >= {args.count}. Skip.")
        return

    needed = args.count - existing
    print(f"Downloading {needed} scenes to {out_dir} ...")

    # Use same URL repeatedly with different seeds for variety
    random.seed(42)
    urls = []
    for i in range(needed):
        base = random.choice(FALLBACK_URLS)
        # Add random query param to bypass CDN cache
        urls.append(base + f"&r={random.randint(0,999999)}")

    success = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for i, url in enumerate(urls):
            fname = f"scene_{existing+i:05d}.jpg"
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                success += 1
                continue
            fut = ex.submit(download_one, url, fpath, i+1, needed)
            futures[fut] = (i+1, url)

        for fut in as_completed(futures):
            if fut.result():
                success += 1
            if success % 100 == 0:
                print(f"  Progress: {success}/{needed}")

    print(f"Downloaded {success}/{needed} scenes -> {out_dir}")
    print(f"HINT: Set --scene-dir datasets/scenes in training command")


if __name__ == "__main__":
    main()