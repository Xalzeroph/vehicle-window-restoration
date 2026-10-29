


from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math


@dataclass(frozen=True)
class Source:
    path: str
    sha256: str
    capture_group: str
    dataset: str
    rights: str
    metadata: dict


def _normalized_list(file):
    if not Path(file).is_file():
        raise FileNotFoundError(file)
    obj=json.loads(Path(file).read_text())
    if isinstance(obj,list):return obj
    if "items" in obj:return obj["items"]
    if "source_objects" in obj:return obj["source_objects"]
    raise ValueError("Unrecognized manifest schema "+str(file))


def import_source(file,kind):
    records=_normalized_list(file)
    items=[]
    for row in records:
        if kind=="kitti":
            name=row["filename"]
            digest=row["sha256"]
            group="KITTI/"+row["capture_group"]
            data="KITTI (Geiger et al.)"
            rights="KITTI source terms; research only and verify distribution permission"
        elif kind=="a2d2":
            name=row.get("local_path")
            digest=row.get("sha256")
            group=row["capture_group"]
            data="Audi A2D2"
            rights="CC BY-ND 4.0, do not redistribute adapted images"
        elif kind=="driveact":
            name=row["filename"]
            digest=row["sha256"]
            group="DriveAct/"+row["capture_group"]
            data="Drive&Act ICCV19"
            rights="Fraunhofer IOSB research only, no public redistribution"
        else:raise ValueError("Unknown source kind")
        if not name or not digest or not Path(name).is_file():
            raise ValueError("Missing physical source bytes or checksums: "+repr(name))
        if any(x in str(name).lower() for x in ("real_pair","test_sets","/test/")):
            raise ValueError("Potential held-out-test leakage: "+name)
        items.append(Source(name,digest,group,data,rights,row))
    if len({s.path for s in items})!=len(items):
        raise ValueError("Exact duplicate filenames")
    return items


def fold_of_group(group,seed,train_ratio=.8,val_ratio=.1):
    if not (0<train_ratio<1 and 0<val_ratio<1-train_ratio):
        raise ValueError("Bad split")
    h=hashlib.sha256((str(seed)+"/"+group).encode()).digest()
    q=int.from_bytes(h[:8],"big")/2**64
    return "train" if q<train_ratio else ("val" if q<train_ratio+val_ratio else "test")


def stable_balanced_folds(items,seed):


    groups=sorted({s.capture_group for s in items},
                  key=lambda group:hashlib.sha256(
                      (str(seed)+"/"+group).encode()).digest())
    n=len(groups)
    if n>=3:
        test_count=max(1,round(n*.1))
        val_count=max(1,round(n*.1))
        while test_count+val_count>=n:
            if test_count>val_count:test_count-=1
            else:val_count-=1
        train_count=n-test_count-val_count
    elif n==2:
        train_count,val_count,test_count=1,1,0
    else:
        train_count,val_count,test_count=n,0,0
    folds={
        "train":sorted(groups[:train_count]),
        "val":sorted(groups[train_count:train_count+val_count]),
        "test":sorted(groups[train_count+val_count:]),
    }
    assignments={group:split for split,listed in folds.items() for group in listed}
    return assignments,folds


def dedupe_source_images(sources):

    unique={}
    for source in sources:
        earlier=unique.get(source.sha256)
        if earlier is not None:
            if earlier.capture_group!=source.capture_group:
                raise ValueError("Same photo assigned inconsistent capture groups")
            continue
        unique[source.sha256]=source
    return list(unique.values())


def _coprime_multiplier(n,seed):
    if n==1:return 0
    base=int.from_bytes(hashlib.sha256(("stride/"+str(seed)).encode()).digest()[:8],"big")
    candidate=base%n
    if candidate==0:candidate=1
    while math.gcd(candidate,n)!=1:
        candidate=(candidate+1)%n
    return candidate


class RecipeIndex:


    def __init__(self,transmissions,cabin,presets,seed):
        self.t=tuple(transmissions)
        self.r=tuple(cabin)
        self.p=tuple(presets)
        if not self.t or not self.r or not self.p:
            raise ValueError("Empty input source or parameter collection")
        self.seed=int(seed)
        self.count=len(self.t)*len(self.r)*len(self.p)
        self.stride=_coprime_multiplier(self.count,self.seed)
        self.offset=int.from_bytes(hashlib.sha256(("offset/"+str(seed)).encode()).digest()[:8],"big")%self.count

    def get(self,i):
        if not 0<=i<self.count:raise IndexError(i)
        shuffled=(self.stride*i+self.offset)%self.count
        t_index=shuffled%len(self.t)
        r_index=(shuffled//len(self.t))%len(self.r)
        p_index=shuffled//(len(self.t)*len(self.r))
        t=self.t[t_index];r=self.r[r_index];p=self.p[p_index]
        payload=(t.sha256+"|"+r.sha256+"|"+p["key"]+"|"+str(self.seed)).encode()
        return {"recipe_id":hashlib.sha256(payload).hexdigest()[:24],
                "outdoor_path":t.path,"outdoor_sha256":t.sha256,"outdoor_capture_group":t.capture_group,
                "outdoor_dataset":t.dataset,
                "reflection_path":r.path,"reflection_sha256":r.sha256,
                "reflection_capture_group":r.capture_group,
                "reflection_dataset":r.dataset,
                "parameter_key":p["key"],
                "source_rights":[t.rights,r.rights],
                "classification":"ILLUSTRATIVE RESEARCH RECIPE; NOT CALIBRATED/RELEASED"}

    def first(self,n):
        return [self.get(i) for i in range(min(n,self.count))]


def export_index(out,transmissions,reflection,presets,seed,first_n=1000):
    transmissions=dedupe_source_images(transmissions)
    reflection=dedupe_source_images(reflection)
    unpartitioned=RecipeIndex(transmissions,reflection,presets,seed)
    groupT=set(x.capture_group for x in transmissions)
    groupR=set(x.capture_group for x in reflection)
    assignmentT,foldsT=stable_balanced_folds(transmissions,seed)
    assignmentR,foldsR=stable_balanced_folds(reflection,seed)
    capacities={}
    previews=[]
    for fold in ("train","val","test"):
        t=[s for s in transmissions if assignmentT[s.capture_group]==fold]
        r=[s for s in reflection if assignmentR[s.capture_group]==fold]
        if not t or not r:
            capacities[fold]=0
            continue
        index=RecipeIndex(t,r,presets,seed)
        capacities[fold]=index.count

        for entry in index.first(min(first_n//3,index.count)):
            entry["fold"]=fold
            previews.append(entry)
    if len({x["recipe_id"] for x in previews}) != len(previews):
        raise AssertionError("Duplicate recipe IDs across folds")
    safe_capacity=sum(capacities.values())
    report={
      "state":"VIRTUAL RECIPE CAPACITY; NOT PHYSICALLY RENDERED",
      "science_gate":"NO CALIBRATED CABIN RGBD, HDR EXPOSURE OR INDEPENDENT REAL WINDSHIELD VALIDATION",
      "cardinality":{
        "unique_outdoor_images":len(transmissions),
        "unique_outdoor_capture_groups":len(groupT),
        "unique_cabin_frames":len(reflection),
        "unique_cabin_capture_groups":len(groupR),
        "optical_or_illumination_presets":len(presets),
        "cartesian_potential_before_split":unpartitioned.count,
        "capture_safe_potential_by_fold":capacities,
        "capture_safe_potential_total":safe_capacity,
        "cross_split_recipes_excluded":unpartitioned.count-safe_capacity,
        "rendered_images_in_this_step":0
      },
      "split_groups":{"outside":foldsT,"cabin":foldsR},
      "warning":"Distinct recipes are NOT independent scenes. Split by actual capture groups. No derived A2D2 or DriveAct redistribution.",
      "seed":seed,
      "presets":list(presets),
      "preview_research_recipe_rows":previews,
    }
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(report,indent=2)+chr(10))
    return report


def main():
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--kitti-manifest",type=Path)
    a.add_argument("--a2d2-manifest",type=Path)
    a.add_argument("--cabin-manifest",type=Path,required=True)
    a.add_argument("--presets",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--seed",type=int,default=20261015)
    x=a.parse_args()
    sources=[]
    if x.kitti_manifest:sources+=import_source(x.kitti_manifest,"kitti")
    if x.a2d2_manifest:sources+=import_source(x.a2d2_manifest,"a2d2")
    cabin=import_source(x.cabin_manifest,"driveact")
    presets=json.loads(x.presets.read_text())
    if not isinstance(presets,list) or not all("key" in p for p in presets):
        raise ValueError("Expected named physical configuration records")
    report=export_index(x.out,sources,cabin,presets,x.seed)
    print("VIRTUAL_CAPACITY",report["cardinality"],flush=True)

if __name__=="__main__":main()
