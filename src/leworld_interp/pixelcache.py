"""Fast uint8 pixel cache for from-scratch training (Phase 6 depth-scaling).

Random reads of 224x224x3 frames from the 46 GB PushT HDF5 over NFS bottleneck training
at ~1.6 s/step regardless of worker count. This caches a fixed set of EPISODES into a
memmap'd uint8 array on local-visible storage (OS page-cached, shared across the parallel
depth runs), so the dataloader does pure memmap slicing and training becomes GPU-bound.
Normalization (ImageNet for pixels, z-score for actions) is done on the GPU in the loop.

Build once with :func:`build_cache`; train from :class:`CachedWindows`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def build_cache(h5_path, ep_indices, out_dir):
    """Read the given episodes' pixels/actions into a memmap cache under out_dir."""
    import h5py
    import hdf5plugin  # noqa: F401

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_indices = list(map(int, ep_indices))

    with h5py.File(str(h5_path), "r", swmr=True) as f:
        ep_len = f["ep_len"][:]
        ep_off = f["ep_offset"][:]
        action_dim = int(np.prod(f["action"].shape[1:]))
        H, W, C = f["pixels"].shape[1:]
        lens = [int(ep_len[e]) for e in ep_indices]
        total = int(sum(lens))
        px = np.memmap(out_dir / "pixels.u8", dtype=np.uint8, mode="w+", shape=(total, H, W, C))
        act = np.zeros((total, action_dim), dtype=np.float32)
        pos = 0
        local_off = []
        for e, L in zip(ep_indices, lens):
            g = int(ep_off[e])
            px[pos:pos + L] = f["pixels"][g:g + L]
            act[pos:pos + L] = np.asarray(f["action"][g:g + L], dtype=np.float32)
            local_off.append(pos)
            pos += L
            if len(local_off) % 200 == 0:
                px.flush()
                print(f"  cached {len(local_off)}/{len(ep_indices)} episodes ({pos} frames)", flush=True)
    px.flush()
    np.save(out_dir / "actions.npy", act)
    meta = {"n_frames": total, "H": H, "W": W, "C": C, "action_dim": action_dim,
            "ep_len": lens, "ep_offset": local_off, "n_episodes": len(ep_indices)}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    print(f"[cache] wrote {total} frames ({total * H * W * C / 1e9:.1f} GB) for "
          f"{len(ep_indices)} episodes to {out_dir}")
    return meta


class CachedWindows(torch.utils.data.Dataset):
    """num_steps-frame windows at stride frameskip from a pixel cache.

    __getitem__ returns (pixels uint8 (T,3,H,W), action raw float32 (T, frameskip*action_dim)).
    Normalize on the GPU; this stays pure memmap slicing for throughput.
    """

    def __init__(self, cache_dir, num_steps, frameskip):
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.span = num_steps * frameskip
        self.adim = self.meta["action_dim"]
        self._px = None
        self._act = None
        lens, offs = self.meta["ep_len"], self.meta["ep_offset"]
        self.clip_indices = [
            (off, start)
            for off, L in zip(offs, lens)
            for start in range(L - self.span + 1)
            if L >= self.span
        ]

    def _open(self):
        if self._px is None:
            self._px = np.memmap(self.dir / "pixels.u8", dtype=np.uint8, mode="r",
                                 shape=(self.meta["n_frames"], self.meta["H"], self.meta["W"], self.meta["C"]))
            self._act = np.load(self.dir / "actions.npy", mmap_mode="r")

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, idx):
        self._open()
        off, start = self.clip_indices[idx]
        g = off + start
        frames = self._px[g:g + self.span:self.frameskip]          # (T,H,W,C) uint8
        px = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)  # (T,3,H,W)
        a = self._act[g:g + self.span]                              # (span, adim)
        action = torch.from_numpy(np.ascontiguousarray(a)).reshape(self.num_steps, -1).float()
        return {"pixels": px, "action": action}


def collate(items):
    return {
        "pixels": torch.stack([it["pixels"] for it in items], 0),
        "action": torch.stack([it["action"] for it in items], 0),
    }


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def gpu_normalize_pixels(px_uint8, device):
    """(B,T,3,H,W) uint8 -> ImageNet-normalized float on device."""
    x = px_uint8.to(device, non_blocking=True).float().div_(255.0)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (x - mean) / std


def action_znorm(mean2, std2, frameskip):
    """Return (mean, std) tiled to frameskip*action_dim for GPU z-scoring."""
    mean = torch.as_tensor(np.tile(np.asarray(mean2), frameskip), dtype=torch.float32)
    std = torch.as_tensor(np.tile(np.asarray(std2), frameskip), dtype=torch.float32)
    return mean, std
