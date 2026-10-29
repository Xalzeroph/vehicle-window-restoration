


import math
import json
from pathlib import Path
from functools import lru_cache
from collections import Counter
import numpy as np
from PIL import Image,ImageFilter
import torch
from torch.utils.data import Dataset

def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"),dtype=np.float32)/255.


@lru_cache(maxsize=256)
def load_img(path):
    return rgb(path)

@lru_cache(maxsize=256)
def load_numpy(path):
    return np.asarray(np.load(path),dtype=np.float32)


class StreamingTintedGlass(Dataset):
    def __init__(self,train_manifest,crop=128,seed=20261029):
        self.manifest=Path(train_manifest)
        self.parent=self.manifest.parent
        self.rows=json.loads(self.manifest.read_text())["items"]
        self.crop=crop
        self.rng=np.random.default_rng(seed)

        scenes={}
        reflections={}
        for row in self.rows:
            meta=json.loads((self.parent/row["relative_directory"]/"metadata.json").read_text())
            sid=meta["original_scene_id"]
            scenes[sid]={"id":sid,"T":meta["source_T"],"m":meta["film_coverage_mask"],
                          "g":meta["film_gain_field"],"category":row["source_category"],
                          "T_key":row["source_t_key"]}
            reflections[row["source_r_key"]]=meta["source_R"]
        self.scenes=list(scenes.values())
        self.reflections=sorted(reflections.items())
        if len(self.scenes)<2 or len(self.reflections)<2:
            raise ValueError("Need two or more independent reflection source identities")
        self.levels=np.array([0.30,0.45,0.60,0.75,1.0],dtype=np.float32)
        self.source_categories=Counter(x["category"] for x in self.scenes)
        self.source_counts=Counter(x["T_key"] for x in self.scenes)
    def __len__(self):
        return len(self.scenes)*len(self.levels)
    def __getitem__(self,index):
        scene=self.scenes[index//len(self.levels)]
        level=float(self.levels[index%len(self.levels)])
        T=load_img(scene["T"])
        m=load_numpy(scene["m"])
        g=load_numpy(scene["g"])
        h,w=T.shape[:2]
        c=self.crop
        y=int(self.rng.integers(h-c+1));x=int(self.rng.integers(w-c+1))
        clean=np.asarray(T[y:y+c,x:x+c],dtype=np.float32)
        alpha=np.clip(level*g[y:y+c,x:x+c]*m[y:y+c,x:x+c]
                      +(1-m[y:y+c,x:x+c]),0.01,1.)
        tinted=np.asarray(clean*alpha[...,None],dtype=np.float32)


        ids=self.rng.choice(len(self.reflections),size=2,replace=False)
        signals=[]
        for rid in ids:
            source=load_img(self.reflections[int(rid)][1])
            rh,rw=source.shape[:2]
            ry=int(self.rng.integers(rh-c+1));rx=int(self.rng.integers(rw-c+1))
            r=np.asarray(source[ry:ry+c,rx:rx+c],dtype=np.float32)
            if self.rng.integers(2):
                r=r[:,::-1].copy()


            blur=float(self.rng.uniform(0,1.2))
            if blur>0:
                r=np.asarray(Image.fromarray(np.uint8((r*255).round()))
                             .filter(ImageFilter.GaussianBlur(blur)),dtype=np.float32)/255.
            rgain=float(np.exp(self.rng.uniform(np.log(.65),np.log(1.45))))
            noise=self.rng.normal(0,1.5/255,size=tinted.shape).astype(np.float32)
            signals.append(tinted+rgain*r+noise)


        G=self.rng.uniform(.7,1.3,size=(1,1,3)).astype(np.float32)
        gamma=float(self.rng.uniform(.85,1.15))
        raw1,raw2=(signals[0]*G,signals[1]*G)
        rawT=tinted*G
        mask=np.all((raw1>0)&(raw1<1)&(raw2>0)&(raw2<1)&(rawT<1),
                    axis=2,keepdims=True).astype(np.float32)
        x1=np.power(np.clip(raw1,0,1),gamma)
        x2=np.power(np.clip(raw2,0,1),gamma)
        target=np.power(np.clip(rawT,0,1),gamma)
        if self.rng.integers(2):
            x1=x1[:,::-1].copy();x2=x2[:,::-1].copy()
            target=target[:,::-1].copy();mask=mask[:,::-1].copy()
        cv=lambda z:torch.from_numpy(np.ascontiguousarray(z.astype(np.float32,copy=False).transpose(2,0,1)))
        return cv(x1),cv(x2),cv(target),cv(mask)


class CounterfactualFilmCamera(Dataset):
    def __init__(self,manifest,crop=128,seed=20261029,
                 clipping_quantile=.95,reflection_field="uniform"):
        self.source=StreamingTintedGlass(manifest,crop,seed)
        self.scenes=self.source.scenes
        self.reflections=self.source.reflections
        self.levels=self.source.levels
        self.rng=np.random.default_rng(seed)
        self.crop=crop
        self.quantile=clipping_quantile
        if reflection_field not in ("uniform","spatial"):
            raise ValueError("Unknown reflected-light support model")
        self.reflection_field=reflection_field
        if not .8<=clipping_quantile<=1:raise ValueError("Invalid anti-clipping quantile")

    def __len__(self):return len(self.scenes)*len(self.levels)

    def __getitem__(self,index):
        scene=self.scenes[index//len(self.levels)]
        level=float(self.levels[index%len(self.levels)])
        T=load_img(scene["T"])
        m=load_numpy(scene["m"])
        g=load_numpy(scene["g"])
        h,w=T.shape[:2];c=self.crop
        yy=int(self.rng.integers(h-c+1));xx=int(self.rng.integers(w-c+1))
        sl=np.s_[yy:yy+c,xx:xx+c]
        underlying=np.asarray(T[sl],dtype=np.float32)
        alpha=np.clip(level*g[sl]*m[sl]+(1-m[sl]),.01,1)
        tinted=underlying*alpha[...,None]
        ids=self.rng.choice(len(self.reflections),2,replace=False)
        mixes=[]
        for rid in ids:
            R=load_img(self.reflections[int(rid)][1])
            rh,rw=R.shape[:2]
            ry=int(self.rng.integers(rh-c+1));rx=int(self.rng.integers(rw-c+1))
            r=np.array(R[ry:ry+c,rx:rx+c],dtype=np.float32,copy=True)
            if self.rng.integers(2):r=r[:,::-1].copy()
            blur=float(self.rng.uniform(0,1.2))
            if blur>0:
                r=np.array(Image.fromarray(np.uint8(np.rint(r*255)))
                    .filter(ImageFilter.GaussianBlur(blur)),dtype=np.float32)/255
            gain=float(np.exp(self.rng.uniform(math.log(.65),math.log(1.45))))
            if self.reflection_field=="spatial":


                yy_grid,xx_grid=np.mgrid[0:c,0:c].astype(np.float32)
                ang=float(self.rng.uniform(0,2*math.pi))
                project=((xx_grid-(c-1)/2)*math.cos(ang)+
                         (yy_grid-(c-1)/2)*math.sin(ang))/c
                offset=float(self.rng.uniform(-.3,.3))
                softness=float(self.rng.uniform(.04,.20))
                z=np.clip((project-offset)/softness,-25,25)
                smooth=1./(1.+np.exp(-z))
                floor=float(self.rng.uniform(0,.25))
                r=r*(floor+(1-floor)*smooth[...,None])
            noise=self.rng.normal(0,1.5/255,r.shape).astype(np.float32)
            mixes.append(tinted+gain*r+noise)
        raw=[];labels=[];camera_gains=[];gammas=[]
        mask=np.ones((c,c,1),dtype=np.float32)
        for j in range(2):
            WB=self.rng.uniform(.7,1.3,size=(1,1,3)).astype(np.float32)


            unconstrained=np.maximum(np.maximum(mixes[0],mixes[1])*WB,0)
            max_px=unconstrained.max(axis=2)
            critical=float(np.quantile(max_px,self.quantile))
            expo=min(1.,(1-1/255)/max(critical,1e-6))
            G=WB*expo
            gamma=float(self.rng.uniform(.85,1.15))
            gammas.append(gamma);camera_gains.append(np.broadcast_to(G,(c,c,3)))
            target=tinted*G
            mask*=np.all((target>=0)&(target<1),axis=2,keepdims=True).astype(np.float32)
            for mixture in mixes:
                raw_float=mixture*G
                mask*=np.all((raw_float>=0)&(raw_float<1),
                             axis=2,keepdims=True).astype(np.float32)
                raw.append(np.clip(raw_float,0,1)**gamma)
            labels.append(np.clip(target,0,1)**gamma)
        if self.rng.integers(2):
            raw=[r[:,::-1].copy() for r in raw]
            labels=[r[:,::-1].copy() for r in labels]
            camera_gains=[r[:,::-1].copy() for r in camera_gains]
            mask=mask[:,::-1].copy()
        cv=lambda z:torch.from_numpy(np.ascontiguousarray(
                       z.astype(np.float32).transpose(2,0,1)))
        return (torch.stack([cv(q) for q in raw]),
                torch.stack([cv(q) for q in labels]),
                torch.stack([cv(q) for q in camera_gains]),
                torch.tensor(gammas,dtype=torch.float32),cv(mask))
