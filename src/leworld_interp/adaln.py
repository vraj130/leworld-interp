"""AdaLN-zero chunk utilities for the LeWM predictor's ConditionalBlocks.

Each ``ConditionalBlock`` computes ``adaLN_modulation(c).chunk(6, dim=-1)`` which,
in this exact order, gives::

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

with ``modulate(x, shift, scale) = x * (1 + scale) + shift`` and the block forward::

    x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))   # attn site
    x = x + gate_mlp * mlp (modulate(norm2(x), shift_mlp, scale_mlp))   # mlp  site

So ``gate_msa`` / ``gate_mlp`` directly scale the two residual updates. There are 6
blocks but **12 gated injection sites** (an attn branch and an mlp branch per block);
report everything at that 12-site resolution.

The final ``adaLN_modulation[-1]`` Linear is ``(6*dim, dim)`` and is zero-initialised
at the start of training, so any learned norm is a direct fossil record of how much
each site grew. ``split_rows`` slices that Linear's output rows into the 6 chunks.
"""

from __future__ import annotations

import numpy as np
import torch

CHUNK_NAMES: tuple[str, ...] = (
    "shift_msa",
    "scale_msa",
    "gate_msa",
    "shift_mlp",
    "scale_mlp",
    "gate_mlp",
)
GATE_CHUNKS: tuple[str, ...] = ("gate_msa", "gate_mlp")
N_CHUNKS = 6

# the two gated injection sites per block, with the chunks that govern each
SITES = (
    {"branch": "attn", "gate": "gate_msa", "shift": "shift_msa", "scale": "scale_msa"},
    {"branch": "mlp", "gate": "gate_mlp", "shift": "shift_mlp", "scale": "scale_mlp"},
)


def conditional_blocks(model) -> list:
    """The predictor's ConditionalBlocks in depth order (block 0 = first)."""
    return list(model.predictor.transformer.layers)


def adaln_linear(block):
    """The final (zero-init at train start) Linear of a block's adaLN MLP."""
    return block.adaLN_modulation[-1]


def split_chunks(t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a ``(..., 6*dim)`` adaLN *output* into the 6 named ``(..., dim)`` chunks."""
    return dict(zip(CHUNK_NAMES, t.chunk(N_CHUNKS, dim=-1)))


def split_rows(t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a ``(6*dim, ...)`` adaLN *parameter* along dim 0 into 6 named chunks."""
    return dict(zip(CHUNK_NAMES, t.chunk(N_CHUNKS, dim=0)))


def static_adaln_norms(model) -> dict:
    """Per-block, per-chunk static norms of the adaLN final Linear.

    Returns numpy arrays shaped ``(n_blocks, 6)`` in CHUNK_NAMES order:
      * ``w_fro``  -- Frobenius norm of each ``(dim, dim)`` weight chunk
      * ``b_l2``   -- L2 norm of each ``(dim,)`` bias chunk
      * ``b_mean`` -- signed mean of each bias chunk (gate DC offset)
    plus ``dim``, ``chunk_names``, ``n_blocks``.
    """
    blocks = conditional_blocks(model)
    nb = len(blocks)
    w_fro = np.zeros((nb, N_CHUNKS), dtype=np.float64)
    b_l2 = np.zeros((nb, N_CHUNKS), dtype=np.float64)
    b_mean = np.zeros((nb, N_CHUNKS), dtype=np.float64)
    dim = None
    for bi, blk in enumerate(blocks):
        lin = adaln_linear(blk)
        w = lin.weight.detach().float().cpu()  # (6*dim, dim)
        b = lin.bias.detach().float().cpu()  # (6*dim,)
        dim = w.shape[0] // N_CHUNKS
        wrows = split_rows(w)
        brows = split_rows(b)
        for ci, name in enumerate(CHUNK_NAMES):
            w_fro[bi, ci] = torch.linalg.norm(wrows[name]).item()
            b_l2[bi, ci] = torch.linalg.norm(brows[name]).item()
            b_mean[bi, ci] = brows[name].mean().item()
    return {
        "w_fro": w_fro,
        "b_l2": b_l2,
        "b_mean": b_mean,
        "dim": int(dim),
        "chunk_names": list(CHUNK_NAMES),
        "n_blocks": nb,
    }
