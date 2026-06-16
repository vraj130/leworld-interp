"""Decide the released-scale d12 data path: benchmark warm h5 DataLoader throughput.

The Phase 6 note said the h5 pipeline was ~1.6 s/step "regardless of worker count". That
may have been cold NFS streaming. The 44 GB h5 fits in this box's 58 GB available RAM, so if
random reads are fast once warm, we can train released-scale d12 directly from the h5 with no
giant uint8 cache (which would not fit in RAM at 352 GB). This measures windows/s warm.

Target to stay GPU-bound at batch 128: about 128 / 0.28 = ~460 windows/s.

    uv run python -m experiments.bench_io --workers 24 --batches 80
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from leworld_interp import data as D
from leworld_interp import paths

HISTORY_SIZE, NUM_PREDS, FRAMESKIP = 3, 1, 5


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--workers", type=int, default=24)
    pa.add_argument("--batch-size", type=int, default=128)
    pa.add_argument("--batches", type=int, default=80)
    pa.add_argument("--warm", action="store_true", help="sequentially read the h5 first to warm page cache")
    args = pa.parse_args()

    if args.warm:
        t0 = time.time()
        n = 0
        with open(str(paths.PUSHT_H5), "rb") as f:
            while True:
                b = f.read(1 << 26)  # 64 MB
                if not b:
                    break
                n += len(b)
        dt = time.time() - t0
        print(f"[warm] read {n/1e9:.1f} GB sequential in {dt:.0f}s = {n/1e9/dt:.2f} GB/s", flush=True)

    am, asd = D.compute_action_stats(paths.PUSHT_H5)
    ds = D.build_dataset(paths.PUSHT_H5, num_steps=HISTORY_SIZE + NUM_PREDS, frameskip=FRAMESKIP,
                         action_mean=am, action_std=asd, keys_to_load=["pixels", "action"])
    print(f"[bench] full-dataset windows: {len(ds)} | workers={args.workers} bs={args.batch_size}", flush=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None,
        pin_memory=True, collate_fn=D.collate)

    it = iter(loader)
    # skip a few to spin up workers
    for _ in range(5):
        next(it)
    t0 = time.time()
    seen = 0
    for _ in range(args.batches):
        b = next(it)
        seen += b["pixels"].size(0)
    dt = time.time() - t0
    wps = seen / dt
    print(f"[bench] {seen} windows in {dt:.1f}s = {wps:.0f} windows/s "
          f"=> {args.batch_size / wps * 1000:.0f} ms/step at bs{args.batch_size}", flush=True)
    print(f"[verdict] GPU-bound target ~460 wps: {'FEASIBLE direct-from-h5' if wps > 420 else 'too slow, need RAM cache'}",
          flush=True)


if __name__ == "__main__":
    main()
