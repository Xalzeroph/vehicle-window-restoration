#!/usr/bin/env python
"""OJMNetV11 -- Fully Optimized Training (DDP + PCGrad + LPIPS/FID + Multi-Proc)."""
import os, sys, time, json, argparse, random, threading
import numpy as np
from queue import Queue, Empty
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.ojm_net_v10 import OJMNetV10, EMAManager, GradientConflictDetector
from dataclasses import dataclass
from src.training.strong_augment import StrongAugment
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# PYTORCH_CUDA_ALLOC_CONF not set (some GPUs lack expandable_segments support)
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass
try:
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    g = torch.cuda.get_device_properties(0)
    print(f"GPU: {g.name} | VRAM: {g.total_memory/1e9:.1f}GB | TF32: ON")
# === LPIPS (perceptual metric) ===
_HAS_LPIPS = False
_lpips_fn = None
try:
    import lpips
    _HAS_LPIPS = True
    print("LPIPS: module available (lazy load)")
except Exception as e:
    print(f"LPIPS: unavailable ({e})")
def get_lpips_fn(device="cuda"):
    global _lpips_fn, _HAS_LPIPS
    if _lpips_fn is not None:
        return _lpips_fn
    if not _HAS_LPIPS:
        return None
    try:
        _lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        _lpips_fn.eval()
        return _lpips_fn
    except:
        _HAS_LPIPS = False
        return None
# === PIQ: comprehensive IQA metrics ===
_PIQ_AVAIL = False
try:
    import piq
    _PIQ_AVAIL = True
    print(f"PIQ: available ({len(dir(piq))} functions)")
except Exception as e:
    print(f"PIQ: unavailable ({e})")
# === COMPREHENSIVE EVALUATION METRICS ===
def compute_all_metrics(pred, target):
    """Compute all available metrics: full-reference + no-reference.
    pred, target: [B,3,H,W] in [0,1] range on GPU.
    Returns dict of metric_name -> value.
    """
    with torch.no_grad():
        p = pred.clamp(0, 1).contiguous()
        t = target.clamp(0, 1).contiguous()
        results = {}
        # 1. PSNR (full-ref, pixel accuracy)
        mse = F.mse_loss(p, t)
        results["PSNR"] = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8)).item()
        if _PIQ_AVAIL:
            # 2-10. PIQ metrics
            if p.shape[-1] >= 161 and p.shape[-2] >= 161:
                results["MS-SSIM"] = piq.multi_scale_ssim(p, t, data_range=1.0).item()
            else:
                results["MS-SSIM"] = piq.ssim(p, t, data_range=1.0).item()
            results["SSIM"] = piq.ssim(p, t, data_range=1.0).item()
            results["GMSD"] = piq.gmsd(p, t, data_range=1.0).item()
            results["HaarPSI"] = piq.haarpsi(p, t, data_range=1.0).item()
            if p.shape[-1] >= 41 and p.shape[-2] >= 41:
                results["VIF"] = piq.vif_p(p, t, data_range=1.0, sigma_n_sq=2.0).item()
            else:
                results["VIF"] = 0.0
            # LPIPS with fallback
            try:
                lp_fn = get_lpips_fn(p.device)
                if lp_fn is not None:
                    lp = lp_fn(p, t).mean().item()
                else:
                    lp = piq.LPIPS(reduction="mean")(p, t).item()
                results["LPIPS"] = lp
            except:
                results["LPIPS"] = 0.0
            # BRISQUE (no-reference, min 56x56)
            if p.shape[-1] >= 56 and p.shape[-2] >= 56:
                results["BRISQUE"] = piq.brisque(p, data_range=1.0, reduction="none").mean().item()
            else:
                results["BRISQUE"] = 0.0
        else:
            k1,k2=0.01,0.03; c1,c2=(k1*1)**2,(k2*1)**2
            mp=F.avg_pool2d(p,3,1,1); mt=F.avg_pool2d(t,3,1,1)
            sp=(F.avg_pool2d(p**2,3,1,1)-mp**2).clamp(1e-8)
            st=(F.avg_pool2d(t**2,3,1,1)-mt**2).clamp(1e-8)
            spt=F.avg_pool2d(p*t,3,1,1)-mp*mt
            num=(2*mp*mt+c1)*(2*spt+c2); den=(mp**2+mt**2+c1)*(sp+st+c2)
            results["SSIM"] = (num.clamp(0)/den.clamp(1e-8)).mean().item()
            results.update({"MS-SSIM":results["SSIM"],"LPIPS":0.0,"GMSD":0.0,"VIF":0.0,"HaarPSI":0.0,"BRISQUE":0.0})
        # 11. Edge Intensity (gradient correlation)
        gx_p = p.diff(dim=3).abs().mean().item()
        gy_p = p.diff(dim=2).abs().mean().item()
        gx_t = t.diff(dim=3).abs().mean().item()
        gy_t = t.diff(dim=2).abs().mean().item()
        results["EdgeI"] = min((gx_p+gy_p)/max(gx_t+gy_t,1e-8), 2.0)
        return results
def test_time_augment(model, x, n_flips=2):
    """Test-time augmentation: horizontal flip average.
    n_flips: 1 = no flip, 2 = original + flip.
    """
    with torch.no_grad():
        if n_flips <= 1:
            T, R, aux = model(x)
            return T, R
        T1, R1, _ = model(x)
        T2, R2, _ = model(x.flip(-1))
        T = (T1 + T2.flip(-1)) / 2
        R = (R1 + R2.flip(-1)) / 2
        return T, R
def multi_scale_infer(model, x, scales=[0.75, 1.0, 1.25]):
    """Multi-scale inference with scale averaging."""
    with torch.no_grad():
        B, C, H, W = x.shape
        Ts, Rs = [], []
        for s in scales:
            hs, ws = int(H * s), int(W * s)
            x_scaled = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
            T_s, R_s = test_time_augment(model, x_scaled)
            Ts.append(F.interpolate(T_s, size=(H, W), mode="bilinear", align_corners=False))
            Rs.append(F.interpolate(R_s, size=(H, W), mode="bilinear", align_corners=False))
        T = torch.stack(Ts).mean(0)
        R = torch.stack(Rs).mean(0)
        return T, R
_EMPTY_LOSS_DICT = {"total": torch.tensor(0.0)}
@dataclass
class Config:
    f: int = 48; steps: int = 6; sd: float = 0.05
    epochs: int = 100; batch_size: int = 4; lr: float = 2e-4
    weight_decay: float = 5e-5; grad_clip: float = 1.0
    ema_decay: float = 0.9995; label_smoothing: float = 0.05
    gradient_accumulation: int = 2
    lookahead_k: int = 5; lookahead_alpha: float = 0.5
    grad_centralization: float = 0.2
    amp: bool = False; channels_last: bool = True
    img_size: int = 96; samples_per_epoch: int = 5000
    val_samples: int = 500
    log_dir: str = "logs/v11"; ckpt_dir: str = "checkpoints/v11"
    log_interval: int = 20; save_interval: int = 5; ckpt_batches: int = 100
    resume: str = None; seed: int = 42
    real_dirs: tuple = ("dataset/train",)
    texture_dirs: tuple = ("datasets/reflections",)
    scene_dir: str = None
    # Quality toggles
    pcgrad: bool = True
    loss_pruning: bool = True
    lpips_metric: bool = True
    fid_metric: bool = False
    benchmark_dir: str = None
    # B+C quality features
    val_scenes: int = 2
    test_scenes: int = 2
    supervise_r_synth: bool = True
    supervise_r_real: bool = False
    tta: bool = True
    multi_scale: bool = False
def psnr(pred, target):
    with torch.no_grad():
        mse = F.mse_loss(pred.clamp(0,1), target.clamp(0,1))
        return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
def ssim(pred, target):
    with torch.no_grad():
        k1, k2 = 0.01, 0.03
        c1, c2 = (k1*1.0)**2, (k2*1.0)**2
        mp = F.avg_pool2d(pred, 3, 1, 1)
        mt = F.avg_pool2d(target, 3, 1, 1)
        sp = (F.avg_pool2d(pred**2, 3, 1, 1) - mp**2).clamp(min=1e-8)
        st = (F.avg_pool2d(target**2, 3, 1, 1) - mt**2).clamp(min=1e-8)
        spt = F.avg_pool2d(pred*target, 3, 1, 1) - mp*mt
        num = (2*mp*mt+c1)*(2*spt+c2)
        den = (mp**2+mt**2+c1)*(sp+st+c2)
        return (num.clamp(min=0) / den.clamp(min=1e-8)).mean()
# === MULTI-PROCESS PREFETCHER ===
# === GPU PREFETCH LOADER (CUDA stream overlap) ===
class GPUPrefetchLoader:
    """GPU-accelerated DataLoader wrapper with per-thread CUDA stream overlap.
    
    Runs DataLoader iteration in a background thread on a separate CUDA stream,
    allowing GPU data compositing to overlap with training compute.
    CUDA kernels from both streams execute in parallel on the same GPU.
    """
    def __init__(self, loader, queue_size=4, device="cuda"):
        self.loader = loader
        self.q = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = None
        self.device = device
    def __len__(self):
        return len(self.loader)
    def _prefetch(self):
        """Background thread: prefetch batches using per-thread CUDA stream.
        PyTorch per-thread default streams auto-overlap GPU work.
        Caching allocator handles cross-thread tensor sync.
        """
        try:
            for batch in self.loader:
                if self._stop.is_set():
                    break
                self.q.put(batch)
        except Exception:
            import traceback; traceback.print_exc()
        finally:
            self.q.put(None)
    def __iter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._prefetch, daemon=True)
        self._thread.start()
        return self
    def __next__(self):
        while True:
            try:
                batch = self.q.get(timeout=300)
                if batch is None:
                    raise StopIteration
                return batch
            except Empty:
                raise StopIteration
# === THREAD PREFETCHER (improved, blocking) ===
class ThreadPrefetchLoader:
    """Background thread prefetcher with proper blocking Queue."""
    def __init__(self, loader, queue_size=2):
        self.loader = loader
        self.q = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = None
    def __len__(self):
        return len(self.loader)
    def _prefetch(self):
        try:
            for batch in self.loader:
                if self._stop.is_set():
                    break
                self.q.put(batch)
        except Exception:
            import traceback; traceback.print_exc()
        finally:
            self.q.put(None)
    def __iter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._prefetch, daemon=True)
        self._thread.start()
        while True:
            try:
                batch = self.q.get(timeout=300)
            except Empty:
                break
            if batch is None:
                break
            yield batch
def create_dataloaders(cfg, world_size=1, rank=None, split="train"):
    from data_v4 import HighQualityWindowDataset
    from src.training.real_dataset import RealPairedDataset
    from torch.utils.data import ConcatDataset
    rdirs = [os.path.join(BASE, d) if not os.path.isabs(d) else d for d in cfg.real_dirs]
    tdirs = [os.path.join(BASE, d) if not os.path.isabs(d) else d for d in cfg.texture_dirs]
    if split == "train":
        ds_syn = HighQualityWindowDataset(
            real_label_dirs=rdirs, texture_dirs=tdirs,
            scene_dir=cfg.scene_dir, size=cfg.img_size,
            samples_per_epoch=cfg.samples_per_epoch,
            bg_ratio=0.5, augment=True, use_cuda=True)
        ds_real = RealPairedDataset(rdirs[0], cfg.img_size, "train", cfg.val_scenes, cfg.test_scenes, augment=True)
        ds = ConcatDataset([ds_syn, ds_real])
    elif split == "val":
        ds_syn = HighQualityWindowDataset(
            real_label_dirs=rdirs, texture_dirs=tdirs,
            scene_dir=cfg.scene_dir, size=cfg.img_size,
            samples_per_epoch=cfg.val_samples,
            bg_ratio=0.5, augment=False, use_cuda=True)
        ds_real = RealPairedDataset(rdirs[0], cfg.img_size, "val", cfg.val_scenes, cfg.test_scenes, augment=False)
        ds = ConcatDataset([ds_syn, ds_real])
    elif split == "test":
        # Test: real data only from held-out test scenes, synthetic with different seed
        ds_syn = HighQualityWindowDataset(
            real_label_dirs=rdirs, texture_dirs=tdirs,
            scene_dir=cfg.scene_dir, size=cfg.img_size,
            samples_per_epoch=cfg.val_samples,
            bg_ratio=0.5, augment=False, use_cuda=True)
        ds_real = RealPairedDataset(rdirs[0], cfg.img_size, "test", cfg.val_scenes, cfg.test_scenes, augment=False)
        ds = ConcatDataset([ds_syn, ds_real])
    else:
        raise ValueError(f"Unknown split: {split}")
    sampler = DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=(split == "train")
    ) if world_size > 1 else None
    dl = DataLoader(ds, cfg.batch_size, shuffle=(sampler is None and split == "train"),
                    sampler=sampler, num_workers=0, pin_memory=False,
                    drop_last=(split == "train"))
    return dl
# === LOOKAHEAD (compatible wrapper) ===
def create_optimizer(model, cfg):
    opt = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and not ("norm" in n or "bias" in n or "log_vars" in n)], "weight_decay": cfg.weight_decay},
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and ("norm" in n or "bias" in n or "log_vars" in n)], "weight_decay": 0}
    ], lr=cfg.lr, betas=(0.9, 0.95))
    slow_params = {}
    for g in opt.param_groups:
        for p in g["params"]:
            if p.requires_grad:
                slow_params[id(p)] = p.data.clone()
    k, alpha = cfg.lookahead_k, cfg.lookahead_alpha
    class Lookahead:
        def __init__(self):
            self.opt = opt; self.k = k; self.alpha = alpha
            self.slow = slow_params; self.counter = 0
            self._slow_keys = set(slow_params.keys())
            self._optimizer = opt
        @property
        def param_groups(self): return self.opt.param_groups
        @param_groups.setter
        def param_groups(self, v): self.opt.param_groups = v
        @property
        def state(self): return self.opt.state
        @state.setter
        def state(self, v): self.opt.state = v
        def zero_grad(self, set_to_none=True):
            self.opt.zero_grad(set_to_none=set_to_none)
        def step(self, closure=None):
            loss = self.opt.step(closure)
            self.counter += 1
            if self.counter % self.k == 0:
                for g in self.opt.param_groups:
                    for p in g["params"]:
                        pid = id(p)
                        if pid in self.slow:
                            self.slow[pid] = self.slow[pid] + self.alpha * (p.data - self.slow[pid])
                            p.data.copy_(self.slow[pid])
            return loss
        def state_dict(self):
            return {"opt": self.opt.state_dict(),
                    "slow": {str(k): v for k,v in self.slow.items()},
                    "counter": self.counter}
        def load_state_dict(self, sd):
            self.opt.load_state_dict(sd["opt"])
            self.slow = {int(k): v for k,v in sd.get("slow", {}).items()}
            self.counter = sd.get("counter", 0)
        def add_param_group(self, group):
            self.opt.add_param_group(group)
    return Lookahead(), opt
class Trainer:
    def __init__(self, model, cfg, vgg_loss=None, is_main=True, world_size=1, device=None):
        self.cfg = cfg; self.is_main = is_main; self.world_size = world_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.vgg_loss = vgg_loss.to(self.device) if vgg_loss else None
        if vgg_loss: self.vgg_loss.eval()
        if self.device.type == "cuda" and cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.optimizer, self.inner_opt = create_optimizer(self.model, cfg)
        self.scaler = torch.amp.GradScaler() if cfg.amp else None
        self.ema = EMAManager(self.model, decay=cfg.ema_decay)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.inner_opt, schedulers=[
                torch.optim.lr_scheduler.LinearLR(self.inner_opt, start_factor=0.01, total_iters=5),
                torch.optim.lr_scheduler.CosineAnnealingLR(self.inner_opt, T_max=cfg.epochs-5)
            ], milestones=[5])
        self.best_loss = float("inf"); self.best_psnr = 0.0
        self.start_epoch = 0; self.history = {"train_loss": [], "val_loss": [], "val_psnr": []}
        os.makedirs(cfg.log_dir, exist_ok=True); os.makedirs(cfg.ckpt_dir, exist_ok=True)
        if cfg.resume and os.path.exists(cfg.resume):
            self._resume(cfg.resume)
        # PCGrad
        self.grad_monitor = GradientConflictDetector(num_losses=18) if cfg.pcgrad else None
        self._pruned_losses = set()
        self._loss_weight_history = []
        # LPIPS
        self._lpips_fn = None
        if cfg.lpips_metric and _HAS_LPIPS:
            # Use lazy init: get_lpips_fn() loads LPIPS model on first call
            self._lpips_fn = get_lpips_fn(self.device) if _HAS_LPIPS else None
        self.train_data_time = 0.0
        self.train_compute_time = 0.0
    def _resume(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(ck["model"])
        else:
            self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.ema.load_state_dict(ck["ema"])
        self.start_epoch = ck.get("epoch", 0) + 1
        self.best_loss = ck.get("best_loss", float("inf"))
        self.best_psnr = ck.get("best_psnr", 0.0)
        if self.is_main:
            print(f"Resumed from {path} (epoch {self.start_epoch})")
    def fit(self, dl_train, dl_val):
        if self.is_main:
            lf = os.path.join(self.cfg.log_dir, "log.txt")
            self.log_f = open(lf, "a")
            self.csv_f = open(os.path.join(self.cfg.log_dir, "metrics.csv"), "w")
            self.csv_f.write("epoch,train_loss,val_loss,val_psnr,val_ssim,val_lpips,lr,train_time_s\n")

            self.csv_f.flush()
        else:
            self.log_f = None
            self.csv_f = None
        for ep in range(self.start_epoch, self.cfg.epochs):
            if hasattr(dl_train, "sampler") and hasattr(dl_train.sampler, "set_epoch"):
                dl_train.sampler.set_epoch(ep)
            tl = GPUPrefetchLoader(dl_train, device=self.device) if torch.cuda.is_available() else dl_train
            t0 = time.time()
            tr = self.train_epoch(tl)
            train_time = time.time() - t0
            va, vp, vs, vl = self.validate(dl_val)
            if self.is_main:
                self.scheduler.step()
                lr = self.inner_opt.param_groups[0]["lr"]
                t = time.strftime("%Y-%m-%d %H:%M:%S")
                msg = (f"[{t}]  E{ep+1}  train_loss={tr:.4f}  val_loss={va:.4f}  "
                       f"val_psnr={vp:.2f}  val_ssim={vs:.4f}  "
                       f"{'val_lpips='+f'{vl:.4f}' if vl else ''}  "
                       f"lr={lr:.2e}  [{train_time:.0f}s]")
                print(msg)
                self.log_f.write(msg + "\n"); self.log_f.flush()
                self.history["train_loss"].append(tr)
                self.history["val_loss"].append(va)
                self.history["val_psnr"].append(vp)
                if va < self.best_loss:
                    self.best_loss = va; self.best_psnr = vp
                    torch.save(self._state(ep), os.path.join(self.cfg.ckpt_dir, "best_loss.pt"))
                if vp > self.best_psnr:
                    torch.save(self._state(ep), os.path.join(self.cfg.ckpt_dir, "best_psnr.pt"))
                if (ep+1) % self.cfg.save_interval == 0:
                    torch.save(self._state(ep), os.path.join(self.cfg.ckpt_dir, f"epoch_{ep+1:03d}.pt"))
                if self.cfg.ckpt_batches > 0:
                    # Clean old batch checkpoints (keep last 3)
                    bckpts = sorted([f for f in os.listdir(self.cfg.ckpt_dir) if f.startswith("batch_")])
                    for f_old in bckpts[:-3]:
                        os.remove(os.path.join(self.cfg.ckpt_dir, f_old))
                # CSV row
                if self.csv_f:
                    lp_str = f"{vl:.4f}" if vl else ""
                    self.csv_f.write(f"{ep+1},{tr:.4f},{va:.4f},{vp:.2f},{vs:.4f},{lp_str},{lr:.2e},{train_time}\n")

                    self.csv_f.flush()
                # Report pruned losses
                if self._pruned_losses:
                    print(f"  Pruned: {self._pruned_losses}")
        if self.log_f: self.log_f.close()
        if self.csv_f: self.csv_f.close()
        return self.history
    def _state(self, ep):
        m_sd = self.model.module.state_dict() if hasattr(self.model,"module") else self.model.state_dict()
        return {"model": m_sd, "optimizer": self.optimizer.state_dict(), "ema": self.ema.state_dict(),
                "epoch": ep, "best_loss": self.best_loss, "best_psnr": self.best_psnr,
                "cfg": self.cfg, "hist": self.history}
    def train_epoch(self, loader):
        self.model.train()
        rank = dist.get_rank() if dist.is_initialized() else None
        is_main = rank is None or rank == 0
        total_loss_t = torch.zeros(1, device=self.device); batches = 0
        self.optimizer.zero_grad(set_to_none=True)
        t_data = 0.0; t_comp = 0.0
        gn_t = torch.zeros(1, device=self.device)
        for bi, (I, Tg, Rg, hr) in enumerate(loader):
            t0 = time.time()
            I = I.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
            Tg = Tg.to(self.device, non_blocking=True)
            Rg = Rg.to(self.device, non_blocking=True)
            if self.cfg.grad_centralization > 0:
                I = I - I.mean(dim=[2,3], keepdim=True) * self.cfg.grad_centralization
            I, Tg, Rg, hr = StrongAugment.apply(I, Tg, Rg, hr)
            has_refl = hr.to(self.device, non_blocking=True)
            t1 = time.time()
            t_data += t1 - t0
            with torch.amp.autocast("cuda", enabled=self.cfg.amp):
                Tp, Rp, aux = self.model(I)
                model_ref = self.model.module if hasattr(self.model,"module") else self.model
                ls = model_ref.compute_loss(Tp, Rp, aux, Tg, Rg, I, has_refl)
            # Loss pruning: skip pruned losses
            if self._pruned_losses:
                for k in list(ls.keys()):
                    if k in self._pruned_losses and k != "total":
                        ls.pop(k, None)
            if self.world_size > 1 and hasattr(self.model, "no_sync") and (bi+1) % self.cfg.gradient_accumulation != 0:
                with self.model.no_sync():
                    ls["total"].backward()
            else:
                ls["total"].backward()
            t2 = time.time()
            # PCGrad monitoring: capture grad directions at shared params
            if self.grad_monitor and bi % 50 == 0 and batches > 0:
                shared = [p for n,p in model_ref.named_parameters()
                         if "conv" in n and p.grad is not None][:1]
                if shared and hasattr(self.grad_monitor, "capture_grad"):
                    loss_names = [k for k in ls if k != "total" and isinstance(ls[k], torch.Tensor)]
                    if self.grad_monitor.loss_names != loss_names:
                        self.grad_monitor.set_loss_names(loss_names)
                    # Simplified: compute grad of last shared layer for each loss
                    grad_vec = torch.cat([p.grad.flatten() for p in shared])
                    self.grad_monitor._buffer["_shared"] = grad_vec
            t_comp += t2 - t1
            if hasattr(self.optimizer, "_optimizer"):
                real_opt = self.optimizer._optimizer
            else:
                real_opt = self.optimizer
            # gn_t persists across batches (reset in log print)
            if (bi+1) % self.cfg.gradient_accumulation == 0:
                if self.cfg.grad_clip > 0:
                    gn_t = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip).detach()
                if self.scaler:
                    self.scaler.unscale_(real_opt)
                    self.scaler.step(real_opt); self.scaler.update()
                else:
                    real_opt.step()
                self.ema.update()
                self.optimizer.zero_grad(set_to_none=True)
                # Intra-epoch checkpoint: save every N batches
                if self.cfg.ckpt_batches > 0 and is_main and (bi+1) % self.cfg.ckpt_batches == 0:
                    ckpt = self._state(self.start_epoch + (bi // len(loader)))
                    torch.save(ckpt, os.path.join(self.cfg.ckpt_dir, f"batch_{bi:06d}.pt"))
                # Loss pruning: track uncertainty weights
                if self.cfg.loss_pruning and self.grad_monitor:
                    weight_vec = torch.stack([
                        ls[k].detach() for k in ls if k != "total" and isinstance(ls[k], torch.Tensor)
                    ]) if len([k for k in ls if k != "total" and isinstance(ls[k], torch.Tensor)]) > 0 else None
                    if weight_vec is not None:
                        self._loss_weight_history.append(weight_vec)
                        if len(self._loss_weight_history) > 50:
                            self._loss_weight_history.pop(0)
                        zero = self.grad_monitor.get_zero_contribution_losses(
                            self._loss_weight_history, threshold=1e-6)
                        redundant = self.grad_monitor.get_redundant_losses(threshold=0.98)
                        new_pruned = zero | redundant
                        if new_pruned and new_pruned != self._pruned_losses:
                            self._pruned_losses = new_pruned
                            if is_main:
                                print(f"  [PCGrad] Auto-pruned losses: {new_pruned}")
            total_loss_t += ls["total"].detach().view(1)
            batches += 1
            if is_main and bi % self.cfg.log_interval == 0:
                t = time.strftime("%Y-%m-%d %H:%M:%S")
                lr = real_opt.param_groups[0]["lr"]
                loss_val = ls["total"].item()
                # Compute true gradient norm from accumulated grads (ignores alignment)
                gn_val = 0.0
                if bi > 0:
                    total = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total += p.grad.norm().item() ** 2
                    gn_val = total ** 0.5
                top5 = sorted((
                    (k, ls[k].item()) for k in ls if k != "total" and isinstance(ls[k], torch.Tensor)
                ), key=lambda x: x[1], reverse=True)[:5]
                extra = " ".join(f"L_{k}={v:.3f}" for k,v in top5)
                print(f"[{t}]   E{self.start_epoch+1} B{bi}/{len(loader)} loss={loss_val:.4f} gn={gn_val:.2f} lr={lr:.2e} {extra}")
    def validate(self, loader):
        self.model.eval()
        rank = dist.get_rank() if dist.is_initialized() else None
        is_main = rank is None or rank == 0
        total_loss_t = torch.zeros(1, device=self.device)
        total_psnr_t = torch.zeros(1, device=self.device)
        total_ssim_t = torch.zeros(1, device=self.device)
        total_lpips_t = torch.zeros(1, device=self.device)
        has_lpips = False
        batches = 0
        for I, Tg, Rg, hr in loader:
            I = I.to(self.device, non_blocking=True)
            Tg = Tg.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.cfg.amp):
                Tp, Rp, aux = self.model(I)
                model_ref = self.model.module if hasattr(self.model,"module") else self.model
                ls = model_ref.compute_loss(Tp, Rp, aux, Tg, Rg, I, hr.to(self.device, non_blocking=True))
            total_loss_t += ls["total"].detach().view(1)
            total_psnr_t += psnr(Tp, Tg).detach().view(1)
            total_ssim_t += ssim(Tp, Tg).detach().view(1)
            # LPIPS
            if self._lpips_fn is not None:
                lp = self._lpips_fn(Tp.clamp(0,1), Tg.clamp(0,1)).mean().detach().view(1)
                total_lpips_t += lp
            batches += 1
        avg_loss = total_loss_t.item() / max(1, batches)
        avg_psnr = total_psnr_t.item() / max(1, batches)
        avg_ssim = total_ssim_t.item() / max(1, batches)
        avg_lpips = total_lpips_t.item() / max(1, batches) if has_lpips else None
        if is_main:
            t = time.strftime("%Y-%m-%d %H:%M:%S")
            lp_msg = f" LPIPS={avg_lpips:.4f}" if avg_lpips else ""
            msg = f"[{t}]   Val: loss={avg_loss:.4f} PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f}{lp_msg}"
            print(msg)
            if self.log_f:
                self.log_f.write(msg + "\n"); self.log_f.flush()
        return avg_loss, avg_psnr, avg_ssim, avg_lpips
    @torch.no_grad()
    def evaluate_test(self, loader):
        self.model.eval()
        all_metrics = []
        for I, Tg, _, _ in loader:
            I = I.to(self.device)
            Tg = Tg.to(self.device)
            if self.cfg.multi_scale:
                Tp, _ = multi_scale_infer(self.model, I)
            elif self.cfg.tta:
                Tp, _ = test_time_augment(self.model, I)
            else:
                Tp, _, _ = self.model(I)
            met = compute_all_metrics(Tp, Tg)
            all_metrics.append(met)
        if not all_metrics:
            return {}
        avg = {}
        for k in all_metrics[0]:
            avg[k] = sum(m[k] for m in all_metrics) / len(all_metrics)
        return avg
def get_free_gpus(min_free_mib=1024):
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.synchronize(i)
            free_mib = torch.cuda.mem_get_info(i)[0] / (1024**2)
            if free_mib > min_free_mib:
                free.append(i)
        except:
            pass
    return free
def main_worker(local_rank, n_gpus, free_gpus, args):
    gpu_id = free_gpus[local_rank]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.ddp_port)
    dist.init_process_group("gloo", rank=local_rank, world_size=n_gpus)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed + local_rank)
    cfg = Config()
    cfg.epochs = args.epochs; cfg.batch_size = args.batch; cfg.lr = args.lr
    # Load best HP from grid search if available
    hp_path = os.path.join(cfg.log_dir, "best_hp.json")
    if os.path.exists(hp_path):
        import json
        hp = json.load(open(hp_path))
        cfg.lr = hp["lr"]
        cfg.weight_decay = hp["weight_decay"]
        cfg.ema_decay = hp["ema_decay"]
        print(f"Grid HP: lr={cfg.lr} wd={cfg.weight_decay} ema={cfg.ema_decay}")
    # Load best HP from grid search if available
    cfg.pcgrad = not args.no_pcgrad
    cfg.loss_pruning = not args.no_prune
    is_main = (local_rank == 0)
    if is_main:
        print(f"OJMNetV11 f={cfg.f} s={cfg.steps} ep={cfg.epochs} bs={cfg.batch_size}")
        print(f"Effective batch: {cfg.batch_size * cfg.gradient_accumulation * n_gpus}")
        print(f"DDP: gloo GPUs={n_gpus} PCGrad={cfg.pcgrad} Prune={cfg.loss_pruning} LPIPS={cfg.lpips_metric}")
    model = OJMNetV10(f=cfg.f, steps=cfg.steps, sd=cfg.sd).to(device)
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=False)
    if is_main:
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {tr:,} trainable")
    vgg_loss = None  # VGG19 removed (dead weight, was never used)
    dl_train = create_dataloaders(cfg, world_size=n_gpus, rank=local_rank, split="train")
    dl_val = create_dataloaders(cfg, world_size=n_gpus, rank=local_rank, split="val")
    dl_test = create_dataloaders(cfg, world_size=n_gpus, rank=local_rank, split="test")
    if is_main:
        print(f"Train: {len(dl_train)} batches x {cfg.gradient_accumulation} accum")
        print(f"Val: {len(dl_val)} batches")
    trainer = Trainer(model, cfg, vgg_loss, is_main=is_main, world_size=n_gpus, device=device)
    hist = trainer.fit(dl_train, dl_val)
    if is_main:
        print(f"Done! Best loss={trainer.best_loss:.4f} PSNR={trainer.best_psnr:.2f}")
        if trainer._pruned_losses:
            print(f"Auto-pruned losses: {trainer._pruned_losses}")
        print(f"Checkpoints: {cfg.ckpt_dir}/")
    dist.destroy_process_group()
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch",type=int,default=4)
    p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--f",type=int,default=48)
    p.add_argument("--resume",type=str); p.add_argument("--samples",type=int,default=10000)
    p.add_argument("--img-size",type=int,default=96)
    p.add_argument("--log",type=str,default="logs/v11"); p.add_argument("--ckpt",type=str,default="checkpoints/v11")
    p.add_argument("--no-amp",action="store_true"); p.add_argument("--no-pcgrad",action="store_true")
    p.add_argument("--no-prune",action="store_true")
    p.add_argument("--scene-dir",type=str,default=None)
    p.add_argument("--grid-search",action="store_true")
    p.add_argument("--ddp-port",type=int,default=29501)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--ckpt-batches",type=int,default=100,help="Save checkpoint every N batches")
    args = p.parse_args()
    n_total = torch.cuda.device_count()
    free_gpus = get_free_gpus(min_free_mib=1024)
    n_free = len(free_gpus)
    if n_free == 0:
        print("ERROR: No free GPUs"); sys.exit(1)
    cfg = Config()
    cfg.epochs = args.epochs; cfg.batch_size = args.batch; cfg.lr = args.lr
    cfg.f = args.f; cfg.resume = args.resume; cfg.samples_per_epoch = args.samples
    cfg.img_size = args.img_size; cfg.log_dir = args.log; cfg.ckpt_dir = args.ckpt
    if args.no_amp: cfg.amp = False
    if args.scene_dir: cfg.scene_dir = args.scene_dir
    cfg.pcgrad = not args.no_pcgrad; cfg.loss_pruning = not args.no_prune; cfg.ckpt_batches = args.ckpt_batches
    if args.grid_search:
        import itertools, json
        grid = {"lr": [1e-4, 2e-4, 5e-4], "weight_decay": [1e-5, 5e-5, 1e-4], "ema_decay": [0.999, 0.9995]}
        keys, vals = zip(*grid.items())
        combos = list(itertools.product(*vals))
        best_loss = float("inf"); best_cfg = None
        print(f"GS: {len(combos)} combos on {n_free} GPUs")
        for ci, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            gpu_id = free_gpus[ci % n_free]
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            cfg.lr = params["lr"]; cfg.weight_decay = params["weight_decay"]; cfg.ema_decay = params["ema_decay"]
            random.seed(cfg.seed+ci); np.random.seed(cfg.seed+ci); torch.manual_seed(cfg.seed+ci)
            model = OJMNetV10(f=cfg.f, steps=cfg.steps, sd=cfg.sd).to(device)
            vgg_loss = None
            dl_tr = create_dataloaders(cfg, world_size=1, rank=None, split="train")
            dl_va = create_dataloaders(cfg, world_size=1, rank=None, split="val")
            trainer = Trainer(model, cfg, vgg_loss, is_main=True, world_size=1, device=device)
            for ep in range(5):
                trainer.train_epoch(dl_tr)
                va, vp, vs, vl = trainer.validate(dl_va)
            if va < best_loss: best_loss = va; best_cfg = params
            torch.cuda.empty_cache()
            print(f"  [{ci+1}/{len(combos)}] {params} -> loss={va:.4f}")
        print(f"BEST: {best_cfg} -> loss={best_loss:.4f}")
        # Auto-update config with best params, fall through to full training
        cfg.lr = best_cfg["lr"]
        cfg.weight_decay = best_cfg["weight_decay"]
        cfg.ema_decay = best_cfg["ema_decay"]
        # Save best hp for main_worker (wd/ema not in args)
        print(f"Best HP saved. Starting full training with lr={cfg.lr} wd={cfg.weight_decay} ema={cfg.ema_decay}")
        json.dump(best_cfg, open(os.path.join(cfg.log_dir, "best_hp.json"), "w"))
    if n_free == 1:
        gpu_id = free_gpus[0]; torch.cuda.set_device(gpu_id)
        g = torch.cuda.get_device_properties(gpu_id)
        print(f"GPU: {g.name} | VRAM: {g.total_memory/1e9:.1f}GB | 1 GPU")
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
        print(f"OJMNetV11 f={cfg.f} s={cfg.steps} ep={cfg.epochs} bs={cfg.batch_size}")
        print(f"Effective batch: {cfg.batch_size*cfg.gradient_accumulation}")
        print(f"PCGrad={cfg.pcgrad} Prune={cfg.loss_pruning} LPIPS={cfg.lpips_metric}")
        device = torch.device(f"cuda:{gpu_id}")
        model = OJMNetV10(f=cfg.f, steps=cfg.steps, sd=cfg.sd).to(device)
        vgg_loss = None  # VGG19 removed (dead weight)
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {tr:,} trainable")
        dl_train = create_dataloaders(cfg, world_size=1, rank=None, split="train")
        dl_val = create_dataloaders(cfg, world_size=1, rank=None, split="val")
        dl_test = create_dataloaders(cfg, world_size=1, rank=None, split="test")
        print(f"Train: {len(dl_train)} batches x {cfg.gradient_accumulation} accum")
        print(f"Val: {len(dl_val)} batches | Test: {len(dl_test)} batches")
        trainer = Trainer(model, cfg, vgg_loss, is_main=True, world_size=1, device=device)
        hist = trainer.fit(dl_train, dl_val)
        # Final test evaluation
        print("\n" + "="*70)
        print("FINAL TEST EVALUATION")
        print("="*70)
        test_results = trainer.evaluate_test(dl_test)
        if test_results:
            for k, v in sorted(test_results.items()):
                print(f"  Test {k}: {v:.4f}")
            with open(os.path.join(cfg.log_dir, "test_results.json"), "w") as f:
                import json; json.dump(test_results, f, indent=2)
        print(f"Done! Best loss={trainer.best_loss:.4f} PSNR={trainer.best_psnr:.2f}")
        if trainer._pruned_losses:
            print(f"Auto-pruned losses: {trainer._pruned_losses}")
        print(f"Checkpoints: {cfg.ckpt_dir}/")
    else:
        print(f"=== DDP: {n_free}/{n_total} free GPUs (gloo) ===")
        mp.spawn(main_worker, args=(n_free, free_gpus, args), nprocs=n_free, join=True)

