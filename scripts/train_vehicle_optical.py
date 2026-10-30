#!/usr/bin/env python3


import argparse,hashlib,json,random,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from vehicle_view_contract import validate_sample_metadata


def masked_mean(value,mask):
    if value.ndim != 4 or mask.ndim != 4 or mask.shape[1]!=1:
        raise ValueError("Invalid BCHW loss/valid mask")
    if value.shape[0]!=mask.shape[0] or value.shape[-2:]!=mask.shape[-2:]:
        raise ValueError("Loss/mask mismatch")
    denom=mask.sum()*value.shape[1]
    if float(denom.detach())<=0:
        raise ValueError("No valid observed pixels in batch")
    return (value*mask).sum()/denom


def inspect_manifest(p):
    p=Path(p)
    data=json.loads(p.read_text())
    if "items" not in data or not data["items"]:
        raise ValueError("Missing actual rendered samples")
    candidates={"RESEARCH_CANDIDATES_NOT_AUTHORIZED_FOR_TRAINING_RELEASE"}
    if data.get("status") not in candidates:
        raise ValueError("Unrecognized unvalidated input dataset")
    base=p.parent
    outdoor=set();cabin=set();outdoor_hashes=set();cabin_hashes=set()
    seen_ids=set()
    for row in data["items"]:
        sid=row["id"]
        if sid in seen_ids:raise ValueError("Duplicate sample ID")
        seen_ids.add(sid)
        path=(base/row["relative_directory"]).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError("Untrusted relative path traversal")
        for filename in ("metadata.json","radiance.npz","observed.png","target_tinted.png"):
            if not (path/filename).is_file():
                raise FileNotFoundError(path/filename)
        metadata=json.loads((path/"metadata.json").read_text())
        if not metadata.get("candidate_not_training_release"):
            raise ValueError("Unsupported or mislabeled preview source")
        validate_sample_metadata(metadata,permit_proxy=True)
        t_meta=metadata["source_transmission"]
        r_meta=metadata["source_reflection"]
        if row["capture_group"]!=t_meta["source_group"] or row["cabin_group"]!=r_meta["source_group"]:
            raise ValueError("Source group provenance disagrees with manifest")
        outdoor.add(row["capture_group"])
        cabin.add(row["cabin_group"])
        outdoor_hashes.add(t_meta["sha256"])
        cabin_hashes.add(r_meta["sha256"])
    return {"manifest":data,"root":base,"outdoor_groups":outdoor,"cabin_groups":cabin,
            "outdoor_hashes":outdoor_hashes,"cabin_hashes":cabin_hashes,
            "manifest_sha256":hashlib.sha256(p.read_bytes()).hexdigest()}


def verify_disjoint(a,b):
    for label,field in (("outdoor groups","outdoor_groups"),
                        ("cabin groups","cabin_groups"),
                        ("outdoor images","outdoor_hashes"),
                        ("cabin images","cabin_hashes")):
        overlap=a[field]&b[field]
        if overlap:
            raise ValueError(f"TRAIN/VALIDATION LEAKAGE: {label}: {sorted(overlap)[:4]}")
    return True


def _load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"),dtype=np.float32)/255.


class VehiclePairs(Dataset):
    def __init__(self,manifest,*,crop,augment,seed):
        self.info=inspect_manifest(manifest)
        self.samples=self.info["manifest"]["items"]
        self.base=self.info["root"]
        self.crop=int(crop)
        if self.crop<=0 or self.crop%32:
            raise ValueError("Crop size must be positive multiple of 32")
        self.augment=augment
        self.seed=seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        row=self.samples[index]
        p=self.base/row["relative_directory"]
        with np.load(p/"radiance.npz",allow_pickle=False) as z:
            r=np.asarray(z["reflection"],dtype=np.float32)
            t_scene=np.asarray(z["transmitted"],dtype=np.float32)
            saturated=np.asarray(z["saturated"],dtype=bool)
            covered=np.asarray(z["valid_ghost"],dtype=bool)
        metadata=json.loads((p/"metadata.json").read_text())
        cam=metadata["camera"]
        wb=np.asarray(cam["wb_rgb"],dtype=np.float32).reshape(1,1,3)
        exposure=float(cam["exposure"])
        reflected=np.asarray(np.where(
            r*exposure*wb<=.0031308,12.92*np.clip(r*exposure*wb,0,1),
            1.055*np.maximum(np.clip(r*exposure*wb,0,1),1e-10)**(1/2.4)-.055),
            dtype=np.float32)
        observed=_load_rgb(p/"observed.png")
        target=_load_rgb(p/"target_tinted.png")
        valid=covered&~saturated
        if t_scene.shape!=r.shape:
            raise ValueError("Transmitted/reflected optics shape mismatch")
        if observed.shape!=target.shape or reflected.shape!=observed.shape or valid.shape!=observed.shape[:2]:
            raise ValueError("Inconsistent spatial supervision shapes")
        h,w=valid.shape
        if min(h,w)<self.crop:
            raise ValueError("Input image smaller than train crop")
        rng=np.random.default_rng(
            np.random.SeedSequence([self.seed,index,random.getrandbits(32) if self.augment else 0]))
        if self.augment:


            candidate_patches=[]
            lum=np.array([.2126,.7152,.0722],dtype=np.float32)
            reflected_energy=r@lum
            transmitted_energy=t_scene@lum
            fraction=reflected_energy/(transmitted_energy+reflected_energy+1e-8)
            for _ in range(16):
                y=int(rng.integers(0,h-self.crop+1))
                x=int(rng.integers(0,w-self.crop+1))
                ok=valid[y:y+self.crop,x:x+self.crop]
                availability=float(ok.mean())
                signal=float(np.mean(fraction[y:y+self.crop,x:x+self.crop][ok])) if ok.any() else 0.
                candidate_patches.append((availability,signal,y,x))
            best=max(x[0] for x in candidate_patches)


            candidates=sorted((x for x in candidate_patches
                               if x[0]>=min(.9,best)),key=lambda x:x[1])
            if not candidates:
                raise ValueError("No geometrically valid training patch")
            choice=int(rng.integers(len(candidates)))
            _,_,y,x=candidates[choice]
        else:
            y=(h-self.crop)//2;x=(w-self.crop)//2
        arrs=[observed,target,reflected,valid[...,None].astype(np.float32)]
        arrs=[a[y:y+self.crop,x:x+self.crop] for a in arrs]
        if self.augment and rng.integers(2):
            arrs=[a[:,::-1].copy() for a in arrs]
        out=[torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))) for a in arrs]
        if out[-1].sum()==0:raise ValueError("Patch has zero valid supervision")
        return tuple(out)


def build_rdnet(checkpoint,classifier_checkpoint,device):
    rdnet_dir=Path(__file__).resolve().parents[1]/"RDNet"
    sys.path.insert(0,str(rdnet_dir))
    from models.arch.RDnet_ import FullNet_NLP
    from models.arch.classifier import PretrainedConvNext
    net=FullNet_NLP(channels=[64,128,256,512],layers=[2,2,4,2],
                    num_subnet=4,loss_col=4,num_classes=1000,
                    drop_path=0,save_memory=True,inter_supv=True,
                    head_init_scale=None,kernel_size=3)
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    sd=ck.get("icnn",ck.get("model",ck.get("state_dict",ck)))
    net.load_state_dict(sd,strict=True)
    classifier=PretrainedConvNext("convnext_small_in22k")
    ck2=torch.load(classifier_checkpoint,map_location="cpu",weights_only=False)
    classifier.load_state_dict(ck2.get("icnn",ck2.get("model",ck2)),strict=True)
    classifier.eval().requires_grad_(False)


    for name,param in net.baseball.named_parameters():
        param.requires_grad_(name.startswith(('layers.3.', 'norm3.')))
    net.to(device);classifier.to(device)
    return net,classifier


class _PhysicalViews(Dataset):
    def __init__(self, manifest, crop, augment, seed):
        self.source = VehiclePairs(manifest, crop=crop, augment=augment, seed=seed)
        self.rows = self.source.samples
        self.paired = False
        self.response = 'srgb-sensor'

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        observed, target, _, mask = self.source[index]

        return (observed[None,None],target[None],mask,
                torch.ones_like(observed)[None],torch.ones(1))

    def group(self, index):
        return self.rows[index]['capture_group']


class _HFInterventions(Dataset):


    def __init__(self, manifest, crop, seed, reflection_field):
        from vehicle_film_forward import CounterfactualFilmCamera
        self.source = CounterfactualFilmCamera(manifest, crop=crop, seed=seed,
                                               reflection_field=reflection_field)
        self.scenes = self.source.scenes
        self.levels = self.source.levels
        self.paired = True
        self.response = 'shared-power-camera'

        self.rows = None

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        observations, targets, gains, gammas, mask = self.source[index]


        return observations.view(2,2,*observations.shape[1:]), targets, mask, gains, gammas

    def group(self, index):
        return self.scenes[index // len(self.levels)]['T_key']


class _OriginalZhouPairs(Dataset):
    def __init__(self, manifest, crop, seed, fold):
        self.root = Path(manifest).parent
        self.meta = json.loads(Path(manifest).read_text())
        if not (len(self.meta["samples"]) == 20000 and len(self.meta["scenes"]) == 4000):
            raise ValueError("Unexpected native Zhoumingyu sample catalog")
        self.crop = crop
        self.seed = seed
        self.paired = True
        self.response = "published-I-to-original-T"
        self.fold = fold
        self.rows = []
        for row in self.meta["samples"]:
            identity = row["t_src"]
            value = int.from_bytes(hashlib.sha256(("T:"+identity).encode()).digest()[:8], "big") / 2**64
            is_validation = value >= .9
            if ((fold == "validation") == is_validation
                    and (self.root/"I"/(row["sid"]+".png")).is_file()
                    and (self.root/"T"/(row["scene_id"]+".png")).is_file()):
                self.rows.append(row)
        if not self.rows:
            raise ValueError("No available complete native I/T pairs in this source split")
        print("ZHOU_ORIGINAL_SPLIT", json.dumps({
            "fold": fold, "pairs": len(self.rows),
            "transmission_sources": len(set(row["t_src"] for row in self.rows)),
            "declared_native_pairs":len(self.meta["samples"]),
            "synthetic_augmentation": False}), flush=True)

    def __len__(self):
        return len(self.rows)

    def group(self,index):
        return self.rows[index]["t_src"]

    def __getitem__(self,index):
        row=self.rows[index]
        observed=_load_rgb(self.root/"I"/(row["sid"]+".png"))
        target=_load_rgb(self.root/"T"/(row["scene_id"]+".png"))
        if observed.shape!=target.shape or min(observed.shape[:2])<self.crop:
            raise ValueError(f"Native I/T size mismatch: {row['sid']}")
        h,w=observed.shape[:2]
        if self.fold=="train":
            rng=np.random.default_rng(random.getrandbits(64))
            y=int(rng.integers(h-self.crop+1))
            x=int(rng.integers(w-self.crop+1))
            flip=bool(rng.integers(2))
        else:
            y=(h-self.crop)//2
            x=(w-self.crop)//2
            flip=False
        observed=observed[y:y+self.crop,x:x+self.crop]
        target=target[y:y+self.crop,x:x+self.crop]
        if flip:
            observed=observed[:,::-1]
            target=target[:,::-1]
        cv=lambda x:torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1)))
        image=cv(observed)
        return (image[None,None],cv(target)[None],
                torch.ones(1,self.crop,self.crop),torch.ones_like(image)[None],
                torch.ones(1))


def _masked_sobolev(error,mask):


    energy=masked_mean(error.square(),mask)
    if error.shape[-1]>1:
        horizontal=error[...,1:]-error[...,:-1]
        valid=mask[...,1:]*mask[...,:-1]
        norm=(valid.sum()*error.shape[1]).clamp_min(1.)
        energy=energy+.5*(horizontal.square()*valid).sum()/norm
    if error.shape[-2]>1:
        vertical=error[...,1:,:]-error[...,:-1,:]
        valid=mask[...,1:,:]*mask[...,:-1,:]
        norm=(valid.sum()*error.shape[1]).clamp_min(1.)
        energy=energy+.5*(vertical.square()*valid).sum()/norm
    return energy


def _risk(output, observations, targets, mask):


    B,C,R=observations.shape[:3]
    dims=observations.shape[3:]
    n=B*C*R
    predicted=output[:,3:]
    reference=torch.cat((targets[:,:,None].expand(B,C,R,*targets.shape[2:]).reshape(n,*dims),
                         targets.reshape(B*C,*targets.shape[2:])),dim=0)
    valid_views=mask[:,None,None].expand(B,C,R,*mask.shape[1:]).reshape(n,*mask.shape[1:])
    valid_clean=mask[:,None].expand(B,C,*mask.shape[1:]).reshape(B*C,*mask.shape[1:])
    e=predicted-reference


    view_loss=_masked_sobolev(e[:n],valid_views)
    clean_loss=_masked_sobolev(e[n:],valid_clean)


    risk=(n*view_loss+(B*C)*clean_loss)/(n+B*C)


    with torch.no_grad():
        errors=e[:n].reshape(B,C,R,*dims).detach()
        center=errors.mean(dim=(1,2),keepdim=True)
        common=_masked_sobolev(center[:,0,0],mask)
        residual=_masked_sobolev((errors-center).reshape(n,*dims),valid_views)
    return risk, {'display_view_h1':float(view_loss.detach()),
                  'display_clean_h1':float(clean_loss.detach()),
                  'display_orbit_mean_h1':float(common),
                  'display_orbit_residual_h1':float(residual)}


def _source_weights(dataset):

    counts = Counter(dataset.group(i) for i in range(len(dataset)))
    return torch.tensor([1/counts[dataset.group(i)] for i in range(len(dataset))],
                        dtype=torch.double)


def _evaluation(model, classifier, dataset, device, limit):

    group_indices = {}
    for i in range(len(dataset)):
        group_indices.setdefault(dataset.group(i), []).append(i)
    indices = []
    depth = 0
    while True:
        progress = False
        for group in sorted(group_indices):
            members = group_indices[group]
            if depth < len(members):
                indices.append(members[depth]); progress = True
                if limit and len(indices) >= limit: break
        if not progress or (limit and len(indices) >= limit): break
        depth += 1
    results = {}
    model.eval()
    with torch.inference_mode():
        for index in indices:
            observed, target, valid, _, _ = dataset[index]
            x = observed[0,0][None].to(device)
            t = target[0][None].to(device)


            m = (torch.ones_like(valid) if dataset.paired else valid)[None].to(device)
            with torch.no_grad():
                cls = classifier(x)
            p = model(x, cls, prompt=True)[1][-1][:, 3:]
            error = masked_mean((p-t).square(),m)
            input_error = masked_mean((x-t).square(),m)
            name = dataset.group(index)
            results.setdefault(name, []).append((
                -10*np.log10(max(float(error),1.e-12)),
                -10*np.log10(max(float(input_error),1.e-12))))
    return {'n_source_groups':len(results), 'n_patches':len(indices),
            'pred_psnr_db':float(np.mean([np.mean([v[0] for v in values])
                                           for values in results.values()])),
            'input_psnr_db':float(np.mean([np.mean([v[1] for v in values])
                                            for values in results.values()])),
            'real_car_exterior_tested':False}


def train(args):
    meta = json.loads(args.train_manifest.read_text())
    native = 'samples' in meta and 'scenes' in meta and 'n_scenes' in meta
    hf = meta.get('status') == 'RESEARCH_ONLY_TINTED_GLASS_SOURCE_PHOTOGRAPHIC_NOT_AUTOMOTIVE_EXTERIOR_CERTIFIED'
    if native:
        if args.val_manifest and args.val_manifest.resolve() != args.train_manifest.resolve():
            raise ValueError("Original Zhoumingyu train and validation must use the identical source meta.json")
        dataset = _OriginalZhouPairs(args.train_manifest,args.crop,args.seed,"train")
        valid = (_OriginalZhouPairs(args.val_manifest,args.crop,args.seed,"validation")
                 if args.val_manifest else None)
        if valid:
            train_ids={r["t_src"] for r in dataset.rows}
            valid_ids={r["t_src"] for r in valid.rows}
            if train_ids & valid_ids:
                raise ValueError("Train/validation original T source leakage")
    elif hf:
        if args.val_manifest:
            a = json.loads(args.train_manifest.read_text())
            b = json.loads(args.val_manifest.read_text())
            if b.get('status') != a['status']:
                raise ValueError('Train and validation source schema differs')
            for key in ('source_t_key','source_r_key'):
                if ({row[key] for row in a['items']}
                        & {row[key] for row in b['items']}):
                    raise ValueError(f'Leakage across HF source identities: {key}')
        dataset = _HFInterventions(args.train_manifest, args.crop, args.seed, args.reflection_field)
        valid = (_HFInterventions(args.val_manifest, args.crop, args.seed, args.reflection_field)
                 if args.val_manifest else None)
    else:
        if not args.allow_illustrative:
            raise ValueError('Uncalibrated physical proxy needs explicit research approval')
        dataset = _PhysicalViews(args.train_manifest,args.crop,True,args.seed)
        valid = (_PhysicalViews(args.val_manifest,args.crop,False,args.seed)
                 if args.val_manifest else None)
        if valid:
            verify_disjoint(dataset.source.info,valid.source.info)
    if args.steps < 1 or args.batch < 1 or args.crop % 32:
        raise ValueError('Positive steps/batch and RDNet-compatible crop multiple of 32 required')
    device = torch.device(args.device)
    np.random.seed(args.seed);random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)


    model,classifier=build_rdnet(args.checkpoint,args.classifier_checkpoint,torch.device('cpu'))
    if device.type == 'cuda':
        free,_ = torch.cuda.mem_get_info(device)
        model_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
        classifier_bytes=sum(p.numel()*p.element_size() for p in classifier.parameters())
        optim_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)


        if free <= model_bytes + classifier_bytes + 3*optim_bytes:
            raise RuntimeError('Insufficient shared GPU free memory for RDNet weights, gradients and AdamW states')
        model.to(device)
        classifier.to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
    start_step=0
    trace=[]
    if args.resume_from:
        old=torch.load(args.resume_from,map_location="cpu",weights_only=False)
        if old.get("manifest_sha256")!=hashlib.sha256(args.train_manifest.read_bytes()).hexdigest():
            raise ValueError("Cannot resume with different source dataset")
        if old.get("total_steps")!=args.steps:
            raise ValueError("Cannot resume with different planned step count")
        model.load_state_dict(old["icnn"],strict=True)
        optimizer.load_state_dict(old["optimizer"])
        start_step=int(old["step"])
        torch.set_rng_state(old["torch_rng"])
        np.random.set_state(old["numpy_rng"])
        random.setstate(old["python_rng"])
        if device.type=="cuda" and "cuda_rng" in old:
            torch.cuda.set_rng_state(old["cuda_rng"],device)
        trace=old["trace"]
    indices=torch.multinomial(_source_weights(dataset),args.steps*args.batch,
                              replacement=True,
                              generator=torch.Generator().manual_seed(args.seed)).tolist()
    loader=DataLoader(dataset,batch_size=args.batch,
                      sampler=indices[start_step*args.batch:],num_workers=0)
    args.out.mkdir(parents=True,exist_ok=bool(args.resume_from))
    started=time.monotonic()
    report={'model':'RDNet FullNet_NLP (original reversible backbone)',
            'method':'full-gamut log-odds RGB restoration with all-label empirical H1 risk',
            'train_sources':len(set(dataset.group(i) for i in range(len(dataset)))),
            'dataset':('Zhoumingyu published I/T original pairs (no regeneration)' if native else
                       'HF research-only photographic synthesis' if hf else 'optical proxy research'),
            'generator_camera_response':dataset.response,'crossed_camera_reflection_interventions':hf,
            'original_input_pairs':native,'source_split':'original T identity SHA256 90/10' if native else None,
            'network_output_domain':'display_rgb',
            'architecture_revision':'bounded_logodds_display_v4',
            'trainable_semantic_stage':'baseball.layers.3 + baseball.norm3',
            'reflection_field':args.reflection_field if hf else 'from optical manifest',
            'display_supervision':('published I to published T' if native else 'all synthetic pixels, real paired valid ROI'),
            'steps':args.steps,'batch':args.batch,'crop':args.crop,
            'real_car_exterior_tested':False,
            'train_manifest_sha256':hashlib.sha256(args.train_manifest.read_bytes()).hexdigest()}
    if valid is not None and not args.resume_from:
        report['before']=_evaluation(model,classifier,valid,device,args.eval_limit)
        print('RDNET_INITIAL',json.dumps(report['before']),flush=True)
    for step,(observations,targets,m,_gains,_gammas) in enumerate(loader,start_step+1):


        observations,targets,m=(v.to(device) for v in (observations,targets,m))
        model.train()
        B,C,R=observations.shape[:3]
        measured=observations.reshape(B*C*R,*observations.shape[3:])
        clean=targets.reshape(B*C,*targets.shape[2:])
        samples=torch.cat((measured,clean),dim=0)
        with torch.no_grad():
            context=classifier(samples)
        _,stages=model(samples,context,prompt=True)
        supervised = torch.ones_like(m) if (hf or native) else m
        loss,parts=_risk(stages[-1],observations,targets,supervised)
        if not torch.isfinite(loss):
            raise FloatingPointError('RDNet display-domain restoration risk became nonfinite')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
        optimizer.step()
        if step==1 or step%args.log_every==0 or step==args.steps:
            item={'step':step,'loss':float(loss.detach()),**parts}
            trace.append(item)
            print('RDNET_STEP',json.dumps(item),flush=True)
        if args.save_every and step % args.save_every == 0 and step != args.steps:
            current={name:value.detach().cpu() for name,value in model.state_dict().items()}
            state={'icnn':current,'optimizer':optimizer.state_dict(),'step':step,
                   'total_steps':args.steps,
                   'manifest_sha256':report['train_manifest_sha256'],
                   'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                   'python_rng':random.getstate(),'trace':trace}
            if device.type=="cuda":
                state['cuda_rng']=torch.cuda.get_rng_state(device)
            temp=args.out/'latest.pth.tmp'
            torch.save(state,temp)
            temp.replace(args.out/'latest.pth')
            print('RDNET_CHECKPOINT',json.dumps({'step':step,'path':str(args.out/'latest.pth')}),flush=True)
    report['train_seconds']=time.monotonic()-started
    report['trace']=trace


    saved_state={name: value.detach().cpu() for name,value in model.state_dict().items()}
    torch.save({'icnn':saved_state,
                'architecture':'RDNet FullNet_NLP',
                'architecture_revision':'bounded_logodds_display_v4',
                'trainable_semantic_stage':'baseball.layers.3 + baseball.norm3',
                'training_method':report['method'],
                'training_data_camera_response':dataset.response,
                'network_output_domain':'display_rgb',
                'train_manifest_sha256':report['train_manifest_sha256'],
                'steps':args.steps,
                'constraint':'bounded RGB transmission and signed nuisance residual, not uncalibrated raw-layer sum',
                'decoder':'one 3-channel log-odds correction sigmoid(logit(clamp(I))+4q)'},
               args.out/'rdnet_counterfactual.pth')
    if valid is not None:
        if hf:


            valid.source.rng = np.random.default_rng(args.seed)
        report['after']=_evaluation(model,classifier,valid,device,args.eval_limit)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RDNET_CAUSAL_TRAIN_DONE',json.dumps({k:v for k,v in report.items() if k!='trace'}),flush=True)


def main():
    p=argparse.ArgumentParser(description='Single RDNet training authority for optical/counterfactual data')
    p.add_argument('--train-manifest',type=Path,required=True)
    p.add_argument('--val-manifest',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--classifier-checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,required=True)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--crop',type=int,default=128)
    p.add_argument('--lr',type=float,default=1.e-5)
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--save-every',type=int,default=0)
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--eval-limit',type=int,default=0)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--allow-illustrative',action='store_true')
    p.add_argument('--reflection-field',choices=('uniform','spatial'),default='spatial')
    train(p.parse_args())


if __name__=='__main__': main()
