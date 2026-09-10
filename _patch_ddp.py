# Save as patch_ddp.py and run
import sys
lines = open(r'E:\vehicle\train_v11.py', encoding='utf-8').readlines()
out = []
out.append('#!/usr/bin/env python\n')
out.append('"""OJMNetV11 - Fully Optimized Training (DDP 8-GPU)."""\n')
out.append('import os, sys, time, json, argparse, warnings, random, math, gc\n')
out.append('import numpy as np\n')
out.append('os.environ["NCCL_BLOCKING_WAIT"] = "0"\n')
out.append('os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"\n')
out.append('import torch.distributed as dist\n')
out.append('import torch.multiprocessing as mp\n')
out.append('from torch.nn.parallel import DistributedDataParallel as DDP\n')
out.append('from torch.utils.data.distributed import DistributedSampler\n')
out.append('try:\n')
out.append('    import torch._dynamo\n')
out.append('    torch._dynamo.config.suppress_errors = True\n')
out.append('except ImportError:\n')
out.append('    pass\n')
out.append('import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n')
out.append('from torch.utils.data import DataLoader\n')
out.append('from dataclasses import dataclass\n')
out.append('from models.ojm_net_v10 import OJMNetV10, EMAManager, VGG19HypercolumnLoss\n')
out.append('from src.training.strong_augment import StrongAugment\n')
out.append('BASE = os.path.dirname(os.path.abspath(__file__))\n')
out.append('sys.path.insert(0, BASE)\n')
out.append('torch.set_float32_matmul_precision("high")\n')
out.append('torch.backends.cudnn.benchmark = True\n')
out.append('torch.backends.cudnn.allow_tf32 = True\n')
out.append('torch.backends.cuda.matmul.allow_tf32 = True\n')

skip = 0
for i, l in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue
    # Skip old header lines
    if i < 30:
        if l.startswith('#') or l.startswith('import') or l.startswith('os.environ') or l.startswith('try:') or l.startswith('    import torch._dynamo') or l.startswith('except') or l.startswith('    pass') or l.startswith('BASE') or l.startswith('sys.path') or l.startswith('torch.set') or l.startswith('torch.backends') or l.startswith('from torch.utils.data import DataLoader') or l.startswith('from dataclasses') or l.strip().startswith('from models') or l.strip().startswith('from src'):
            continue
    # Fix create_dataloaders - add DistributedSampler
    if i == 123:  # original dl_train line
        out.append('    train_sampler = DistributedSampler(ds_train, shuffle=True)\n')
        out.append('    dl_train = DataLoader(ds_train, cfg.batch_size, sampler=train_sampler, num_workers=0, pin_memory=False, drop_last=True)\n')
        skip = 1  # skip original dl_train and dl_val lines
        continue
    if i == 124:  # original dl_val line
        out.append('    val_sampler = DistributedSampler(ds_val, shuffle=False)\n')
        out.append('    dl_val = DataLoader(ds_val, cfg.batch_size, sampler=val_sampler, num_workers=0, pin_memory=False)\n')
        continue
    
    # Fix Trainer.fit - add sampler set_epoch
    if 'for ep in range(self.epoch+1, self.cfg.epochs+1):' in l:
        out.append(l)
        out.append('            if hasattr(tl, "loader") and hasattr(tl.loader, "sampler") and hasattr(tl.loader.sampler, "set_epoch"):\n')
        out.append('                tl.loader.sampler.set_epoch(ep)\n')
        continue
    
    # Fix main - DDP launch
    if l.strip() == 'if __name__=="__main__": main()':
        out.append('if __name__ == "__main__":\n')
        out.append('    n_gpus = torch.cuda.device_count()\n')
        out.append('    print(f"=== DDP: {n_gpus} GPUs === ")\n')
        out.append('    if n_gpus > 1:\n')
        out.append('        os.environ["MASTER_ADDR"] = "127.0.0.1"\n')
        out.append('        os.environ["MASTER_PORT"] = "29501"\n')
        out.append('        mp.spawn(main, nprocs=n_gpus, join=True)\n')
        out.append('    else:\n')
        out.append('        main()\n')
        continue
    
    out.append(l)

with open(r'E:\vehicle\train_v11.py', 'w', encoding='utf-8') as f:
    f.writelines(out)
print(f"Done: {len(out)} lines, {len(lines)} original")