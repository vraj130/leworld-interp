"""PushT data access with model-faithful preprocessing.

Reproduces the exact preprocessing the released LeWM-PushT checkpoint was
trained/evaluated with:

  * pixels: uint8 (T, 3, 224, 224) -> float/255 -> ImageNet normalize -> resize 224
    (matches ``eval.py``'s ``img_transform``).
  * action: raw per-environment-step (span, action_dim) z-scored by the dataset's
    own action statistics, *before* the (num_steps, frameskip*action_dim) reshape
    (matches training's ``get_column_normalizer`` placement).

Windows come from ``stable_worldmodel``'s ``HDF5Dataset``: ``num_steps`` frames at
stride ``frameskip``, with ``action`` reshaped to ``(num_steps, frameskip*action_dim)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def compute_action_stats(h5_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean/std of the raw action column (NaN rows dropped),
    matching the training-time z-score normalizer."""
    import h5py

    with h5py.File(str(h5_path), "r", swmr=True) as f:
        a = f["action"][:]
    a = a[~np.isnan(a).any(axis=1)]
    return a.mean(0).astype(np.float32), a.std(0).astype(np.float32)


class ActionZScore:
    """Picklable per-dim z-score for the raw action column."""

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        self.std = torch.as_tensor(np.asarray(std), dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / self.std


class LeWMTransform:
    """Dict-in / dict-out transform applied per window inside the HDF5 reader.

    Receives ``pixels`` as (num_steps, 3, 224, 224) uint8 and ``action`` as the
    raw (span, action_dim) tensor (reshaped to (num_steps, frameskip*action_dim)
    by the reader *after* this transform runs).
    """

    def __init__(self, action_mean, action_std, img_size: int = 224):
        self.act = ActionZScore(action_mean, action_std)
        self.normalize = T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
        self.img_size = img_size

    def __call__(self, steps: dict) -> dict:
        if "pixels" in steps:
            px = steps["pixels"].to(torch.float32).div_(255.0)
            px = self.normalize(px)
            if px.shape[-1] != self.img_size or px.shape[-2] != self.img_size:
                px = TF.resize(px, [self.img_size, self.img_size], antialias=True)
            steps["pixels"] = px
        if "action" in steps:
            steps["action"] = self.act(steps["action"])
        return steps


def build_dataset(
    h5_path: str | Path,
    *,
    num_steps: int,
    frameskip: int,
    action_mean=None,
    action_std=None,
    normalize: bool = True,
    keys_to_load=None,
):
    """An ``HDF5Dataset`` of ``num_steps``-frame windows with LeWM preprocessing.

    If ``normalize`` and no stats are given, action stats are computed from the
    file. ``keys_to_load`` defaults to the columns the audit uses.
    """
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

    transform = None
    if normalize:
        if action_mean is None or action_std is None:
            action_mean, action_std = compute_action_stats(h5_path)
        transform = LeWMTransform(action_mean, action_std)

    if keys_to_load is None:
        keys_to_load = ["pixels", "action", "proprio", "state", "episode_idx", "step_idx"]

    return HDF5Dataset(
        path=str(h5_path),
        frameskip=frameskip,
        num_steps=num_steps,
        transform=transform,
        keys_to_load=keys_to_load,
    )


def split_indices(
    n_clips: int, *, seed: int = 3072, val_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic clip-level train/val split mirroring training (seed 3072,
    90/10). Returns ``(train_idx, val_idx)`` as numpy int arrays."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_clips, generator=g).numpy()
    n_val = int(round(n_clips * val_frac))
    return perm[n_val:], perm[:n_val]


_COLLATE_TENSOR_KEYS = ("pixels", "action", "proprio", "state", "episode_idx", "step_idx")


def collate(items: list[dict]) -> dict:
    """Stack a list of window dicts into batched tensors ``(B, T, ...)``."""
    out: dict[str, torch.Tensor] = {}
    keys = [k for k in _COLLATE_TENSOR_KEYS if k in items[0]]
    for k in keys:
        out[k] = torch.stack([it[k] for it in items], dim=0)
    return out


def load_batch(dataset, indices) -> dict:
    """Fetch ``indices`` from ``dataset`` and collate into a batched dict."""
    items = [dataset[int(i)] for i in indices]
    return collate(items)
