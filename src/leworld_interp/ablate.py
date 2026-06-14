"""Causal mean-ablation of the predictor's per-block adaLN conditioning.

All interventions act at a single ConditionalBlock's ``adaLN_modulation`` (never at
``cond_proj``, which would ablate every block at once); the vendored block forward is
untouched. Three intervention types:

  * mean ablation (Parts 1/2)  -- pre-hook replaces the conditioning input ``c`` with its
    batch mean (the mean action embedding). Removes per-sample action identity.
  * random control (Part 2 add) -- pre-hook replaces ``c`` with ``c + u`` where ``u`` is a
    random direction scaled so ``||c+u - c|| == ||c_mean - c||`` per token; matched-norm
    control isolating action-specific damage from generic-perturbation / AR compounding.
  * per-branch (Part 3)        -- post-hook on the adaLN output overrides only the MLP
    chunks (shift_mlp, scale_mlp, gate_mlp) or only the attn chunks with their batch mean.

Mean is taken over the batch dimension per (token, channel), so run the predictor on the
full eval batch in one call for a stable mean. Use as a context manager; ``clear()``
removes hooks between configurations.
"""

from __future__ import annotations

import torch

from .adaln import N_CHUNKS, conditional_blocks

ATTN_CHUNK_IDX = (0, 1, 2)  # shift_msa, scale_msa, gate_msa
MLP_CHUNK_IDX = (3, 4, 5)   # shift_mlp, scale_mlp, gate_mlp


class AdaLNAblator:
    def __init__(self, model):
        self.blocks = conditional_blocks(model)
        self._h: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.clear()

    def clear(self):
        for h in self._h:
            h.remove()
        self._h = []

    # -- input-space interventions (pre-hooks on adaLN_modulation) --
    @staticmethod
    def _mean_pre():
        def hook(mod, args):
            c = args[0]
            return (c.mean(dim=0, keepdim=True).expand_as(c).contiguous(),)
        return hook

    @staticmethod
    def _rand_pre(unit):
        def hook(mod, args):
            c = args[0]
            cbar = c.mean(dim=0, keepdim=True)
            disp = (cbar - c).norm(dim=-1, keepdim=True)        # (B,T,1) mean-ablation norm
            u = unit[: c.size(0), : c.size(1)].to(c.device, c.dtype)
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return (c + u * disp,)
        return hook

    # -- output-space intervention (post-hook on adaLN_modulation) --
    @staticmethod
    def _branch_post(chunk_idx):
        def hook(mod, args, out):
            o = out.clone()
            dim = o.shape[-1] // N_CHUNKS
            for ci in chunk_idx:
                sl = slice(ci * dim, (ci + 1) * dim)
                o[..., sl] = o[..., sl].mean(dim=0, keepdim=True)
            return o
        return hook

    # -- arming --
    def mean_ablate(self, block_ids):
        for i in block_ids:
            self._h.append(self.blocks[i].adaLN_modulation.register_forward_pre_hook(self._mean_pre()))
        return self

    def random_ablate(self, block_ids, unit):
        for i in block_ids:
            self._h.append(self.blocks[i].adaLN_modulation.register_forward_pre_hook(self._rand_pre(unit)))
        return self

    def branch_ablate(self, block_id, branch):
        idx = MLP_CHUNK_IDX if branch == "mlp" else ATTN_CHUNK_IDX if branch == "attn" else tuple(range(N_CHUNKS))
        self._h.append(self.blocks[block_id].adaLN_modulation.register_forward_hook(self._branch_post(idx)))
        return self
