#!/usr/bin/env python3


import argparse
import hashlib,json,re,subprocess
from pathlib import Path
from remotezip import RemoteZip
import shutil

URL="https://www.driveandact.com/dataset/kinect_color.zip"

def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--people",type=int,default=9)
    ap.add_argument("--frames-per-clip",type=int,default=20)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    picked={}
    with RemoteZip(URL) as r:
        files=sorted([x for x in r.infolist() if x.filename.endswith(".kinect_color.mp4")],
                     key=lambda i:i.file_size)

        for item in files:
            match=re.search(r"/(vp\d+)/",item.filename)
            if not match:continue
            person=match.group(1)
            if person not in picked:picked[person]=item
        candidates=sorted(picked.items(),key=lambda kv:kv[1].file_size)[:args.people]
        print("SOURCE_PEOPLE",len(picked),"SELECT",[(vp,item.file_size) for vp,item in candidates],flush=True)
        for vp,item in candidates:
            dirname=Path(item.filename).name.removesuffix(".kinect_color.mp4")
            dst=args.out/"videos"/vp/(dirname+".mp4")
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists() or dst.stat().st_size!=item.file_size:
                part=dst.with_suffix(".partial")
                with r.open(item.filename) as inp,part.open("wb") as f:
                    shutil.copyfileobj(inp,f,1<<20)
                if part.stat().st_size!=item.file_size:raise ValueError("Incomplete remote ZIP member")
                part.rename(dst)
            record={"dataset":"Drive&Act (ICCV 2019)","source_archive":URL,
                    "source_member":item.filename,"participant":vp,"capture_group":vp+"/"+dirname,
                    "video_sha256":digest(dst),"video_path":str(dst),"frames":[]}
            dur=float(subprocess.check_output(["ffprobe","-v","error","-show_entries",
                 "format=duration","-of","default=noprint_wrappers=1:nokey=1",str(dst)],
                 text=True).strip())
            folder=args.out/"frames"/vp/dirname
            folder.mkdir(parents=True,exist_ok=True)
            for i in range(args.frames_per_clip):
                sec=(i+.5)*dur/args.frames_per_clip
                frame=folder/f"frame_{i:04}.png"
                if not frame.exists():
                    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-ss",
                                    str(sec),"-i",str(dst),"-frames:v","1","-y",str(frame)],
                                   check=True,timeout=60)
                record["frames"].append({"filename":str(frame),
                    "sha256":digest(frame),"frame_seconds":sec,
                    "capture_group":record["capture_group"],
                    "source":"Drive&Act official author RGB Kinect, research only"})
            index_file=args.out/"records"/(vp+"_"+dirname+".json")
            index_file.parent.mkdir(parents=True,exist_ok=True)
            index_file.write_text(json.dumps(record,indent=2)+"\n")
            print("ACQUIRED_PERSON",vp,"clip",dirname,"frames",len(record["frames"]),flush=True)
    records=[json.loads(p.read_text()) for p in (args.out/"records").glob("*.json")]
    frames=[f for r in records for f in r["frames"]]
    (args.out/"all_cabin_frames.json").write_text(json.dumps({
        "status":"UNREGISTERED CVPR SYNTHESIS RESEARCH INPUT, NOT CALIBRATED",
        "rights":"Research only; no public raw/derived image redistribution",
        "unique_people":len({r["participant"] for r in records}),
        "unique_capture_groups":len(records),"items":frames},indent=2)+"\n")
    print("FINISHED",len(records),"recordings",len(frames),"frames",flush=True)

if __name__=="__main__":
    main()
