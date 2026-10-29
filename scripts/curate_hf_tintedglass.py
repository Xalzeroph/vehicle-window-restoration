#!/usr/bin/env python3


import argparse,json
from collections import Counter
from pathlib import Path


def accept(src):
    return (src.startswith("nature:") or src.startswith("sir2:wild")
            or (src.startswith("matched:") and not src.startswith("matched:scenes")))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    src=json.loads(a.source.read_text())
    out={**src,"items":[r for r in src["items"] if accept(r["source_t_key"])],
         "curation":"exclude sir2:solid, sir2:post, matched:scenes, artificial vector shape T",
         "note":"Photographic scenes, not proven vehicle exterior viewpoint"}
    a.out.write_text(json.dumps(out,indent=2)+"\n")
    print("HF_PHOTOGRAPHIC_CURATED",len(src["items"]),len(out["items"]),
          "unique_T",len(set(r["source_t_key"] for r in out["items"])),
          "unique_R",len(set(r["source_r_key"] for r in out["items"])),flush=True)


if __name__=="__main__":main()
