#!/usr/bin/env python3
"""Train a small semantic nuisance head and recoverer on original October pairs.

No generated reflection images. No original Ω pseudo-mask presented as ground truth.
Segmentation uses conservative SegFormer/paired-difference pseudo-labels; pixels
marked 255 are unknown, excluded from supervised semantic loss.
"""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler

class OctoberPairs(Dataset):
    def __init__(self, root, split, crop, seed):
        self.root=Path(root)
        record=json.loads((self.root/"semantic_annotations.json").read_text())
        source=json.loads((self.root/"source_manifest.json").read_text())
        self.index={r["id"]:r for r in source["items"]}
        self.crop=crop
        self.split=split
        self.seed=seed
        self.rows=[]
        self.counts=[]
        for row in record["annotations"]:
            # Strict video-level split: no images of the same video on both sides.
            scene=row["scene"]
            is_valid=(scene in ("VID_20261019_171126","VID_20260426_173827",
                "VID_20260512_164653","VID_20260609_170534"))
            if is_valid!=(split=="validation"):
                continue
            self.rows.append(row)
            self.counts.append(row["counts"])
        if not self.rows:raise ValueError("No samples in "+split)
        print("OCT_PAIRS",split,"samples",len(self.rows),
            "scenes",len({r["scene"] for r in self.rows}),
            "positive_images",sum(any(r["counts"][x]>50 for x in
                ("human_nuisance","vegetation_nuisance","other_weak_nuisance"))
                for r in self.rows),flush=True)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        src=self.index[row["id"]]
        rgb=[]
        for key in ("I","T"):
            with Image.open(self.root/src["files"][key]) as im:
                resized=im.convert("RGB").resize((448,448),Image.Resampling.BILINEAR)
            rgb.append(np.asarray(resized,dtype=np.float32)/255.)
        with Image.open(self.root/row["label_path"]) as im:
            labels=np.asarray(im,dtype=np.uint8)
        if labels.shape!=(448,448):raise ValueError("Label and RGB grid disagree")
        if self.split=="train":
            # Sample foreground-containing patches without inventing pixels.
            rng=np.random.default_rng(random.getrandbits(64))
            foreground=(labels>0)&(labels<255)
            positive=np.argwhere(foreground)
            if len(positive) and rng.random()<.75:
                yy,xx=positive[int(rng.integers(len(positive)))]
                y=max(0,min(448-self.crop,int(yy)-int(rng.integers(self.crop))))
                x=max(0,min(448-self.crop,int(xx)-int(rng.integers(self.crop))))
            else:
                y=int(rng.integers(449-self.crop));x=int(rng.integers(449-self.crop))
            flip=bool(rng.integers(2))
        else:
            y=(448-self.crop)//2;x=(448-self.crop)//2;flip=False
        rgb=[im[y:y+self.crop,x:x+self.crop] for im in rgb]
        labels=labels[y:y+self.crop,x:x+self.crop]
        if flip:
            rgb=[im[:,::-1] for im in rgb];labels=labels[:,::-1]
        img=[torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))) for im in rgb]
        return img[0],img[1],torch.from_numpy(np.ascontiguousarray(labels.astype(np.int64)))

def reconstruct_risk(pred,target):
    e=pred-target
    l=e.square().mean()
    l=l+.25*((e[...,1:]-e[...,:-1]).square().mean()+
               (e[...,1:,:]-e[...,:-1,:]).square().mean())
    return l

def semantic_risk(logits,label):
    valid=(label!=255)
    if not valid.any():
        return logits.sum()*0
    weights=logits.new_tensor([.15,3.,3.,.6])
    ce=F.cross_entropy(logits,label,ignore_index=255,weight=weights)
    probabilities=logits.softmax(1)
    dice=logits.sum()*0
    active=0
    for channel in (1,2,3):
        truth=(label==channel).float()
        if truth.sum()<10:
            continue
        predicted=probabilities[:,channel]*valid
        dice=dice+1.-(2*(predicted*truth).sum()+1)/(predicted.sum()+truth.sum()+1)
        active+=1
    return ce+(.3*dice/active if active else 0.)

@torch.no_grad()
def evaluate(net,classifier,ds,device):
    net.eval()
    stats=[]
    for x,y,mask in DataLoader(ds,batch_size=1,num_workers=0):
        x=x.to(device);y=y.to(device);mask=mask.to(device)
        ctx=classifier(x)
        _,base_stages,logits=net(x,ctx,return_semantic=True,semantic_gate=False)
        _,gated_stages=net(x,ctx,semantic_gate=True)
        base=base_stages[-1][:,3:];pred=gated_stages[-1][:,3:]
        score=lambda z:float(-10*torch.log10((z-y).square().mean().clamp_min(1e-9)))
        valid=(mask!=255)
        acc=(logits.argmax(1)==mask)&valid
        stats.append(dict(base=score(base),gated=score(pred),input=score(x),
            pseudo_accuracy=float(acc.sum()/valid.sum().clamp_min(1))))
    return {k:round(float(np.mean([d[k] for d in stats])),4) for k in stats[0]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--classifier-checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--crop",type=int,default=256)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--seed",type=int,default=20261030)
    p.add_argument("--save-every",type=int,default=300)
    p.add_argument("--log-every",type=int,default=30)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.set_num_threads(4)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from train_vehicle_optical import build_rdnet
    model,classifier=build_rdnet(args.checkpoint,args.classifier_checkpoint,torch.device("cpu"))
    model.enable_semantic_occlusion(width=32)
    model.requires_grad_(False)
    model.semantic_occlusion.requires_grad_(True)
    # Keep mature RDNet reconstruction fixed, optimize only low-capacity
    # semantics+modulation. Avoid unfreezing 266M parameters with <100 frames.
    dev=torch.device(args.device)
    model.to(dev);classifier.to(dev).eval()
    train=OctoberPairs(args.source,"train",args.crop,args.seed)
    valid=OctoberPairs(args.source,"validation",args.crop,args.seed)
    weights=[1.+3*(r["human_nuisance"]>50)+3*(r["vegetation_nuisance"]>50)+
             (r["other_weak_nuisance"]>50) for r in train.counts]
    samples=WeightedRandomSampler(weights,args.steps,replacement=True,
        generator=torch.Generator().manual_seed(args.seed))
    loader=DataLoader(train,batch_size=1,sampler=samples,num_workers=0)
    optimizer=torch.optim.AdamW(model.semantic_occlusion.parameters(),lr=args.lr,weight_decay=1e-4)
    args.out.mkdir(parents=True,exist_ok=False)
    report={"source":"Zhoumingyu October original I/T real pairs",
      "semantic_labels":"SegFormer ADE20K + I/T difference; pseudo/unreviewed",
      "train_n":len(train),"val_n":len(valid),
      "train_scenes":sorted({x["scene"] for x in train.rows}),
      "val_scenes":sorted({x["scene"] for x in valid.rows}),
      "steps":args.steps,"crop":args.crop,"lr":args.lr,
      "trainable_params":sum(p.numel() for p in model.parameters() if p.requires_grad),
      "architecture":"RDNet with neutral-initialized semantic soft gate"}
    t0=time.monotonic()
    report["before"]=evaluate(model,classifier,valid,dev)
    print("OCT_SEM_INITIAL",json.dumps(report,ensure_ascii=False),flush=True)
    trace=[]
    for step,(x,y,mask) in enumerate(loader,1):
        x=x.to(dev);y=y.to(dev);mask=mask.to(dev)
        model.train()
        with torch.no_grad():
            ctx=classifier(x)
        _,stages,logits=model(x,ctx,return_semantic=True)
        recovery=reconstruct_risk(stages[-1][:,3:],y)
        semantic=semantic_risk(logits,mask)
        loss=recovery+.05*semantic
        if not torch.isfinite(loss):raise FloatingPointError("Non-finite semantic risk")
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.semantic_occlusion.parameters(),1.)
        optimizer.step()
        if step==1 or step%args.log_every==0 or step==args.steps:
            item={"step":step,"recovery":round(float(recovery.detach()),6),
                "semantic":round(float(semantic.detach()),6),
                "total":round(float(loss.detach()),6)}
            trace.append(item)
            print("OCT_SEM_STEP",json.dumps(item),flush=True)
        if step%args.save_every==0:
            temp=args.out/"semantic_latest.pth.tmp"
            torch.save({"step":step,"semantic_occlusion":model.semantic_occlusion.state_dict(),
                "optimizer":optimizer.state_dict(),"source_checkpoint":str(args.checkpoint),
                "label_source":report["semantic_labels"]},temp)
            temp.replace(args.out/"semantic_latest.pth")
            print("OCT_SEM_SAVED",step,flush=True)
    report["after"]=evaluate(model,classifier,valid,dev)
    report["seconds"]=time.monotonic()-t0
    report["trace"]=trace
    torch.save({"semantic_occlusion":model.semantic_occlusion.state_dict(),
        "source_checkpoint":str(args.checkpoint),"labels":report["semantic_labels"]},
        args.out/"semantic_head.pth")
    (args.out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("OCT_SEM_DONE",json.dumps({k:v for k,v in report.items() if k!="trace"}),flush=True)

if __name__=="__main__":
    main()
