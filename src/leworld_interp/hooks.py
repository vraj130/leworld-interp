"""Forward-hook capture of ConditionalBlock internals for Measurements B and C.

The vendored ``ConditionalBlock.forward`` is left untouched; we attach forward
pre/post hooks to each block and to its ``adaLN_modulation`` / ``attn`` / ``mlp``
submodules. Per block per forward pass we capture:

  * ``x_in``    -- residual stream entering the block      (forward_pre_hook on block)
  * ``adaln``   -- adaLN_modulation output, (.., 6*dim)    (post-hook on adaLN_modulation)
  * ``attn_out``-- attn branch output, BEFORE gate_msa     (post-hook on attn)
  * ``mlp_out`` -- mlp  branch output, BEFORE gate_mlp      (post-hook on mlp)
  * ``x_out``   -- residual stream after the block          (post-hook on block)

From these, ``x_mid = x_in + gate_msa * attn_out`` and the two gated updates are
reconstructed, so every quantity B/C needs is available without editing the model.
Use as a context manager; call ``snapshot()`` after each forward to grab and clear.
"""

from __future__ import annotations

import torch

from .adaln import conditional_blocks


class BlockCapture:
    def __init__(self, model, to_cpu: bool = False):
        self.blocks = conditional_blocks(model)
        self.to_cpu = to_cpu
        self._handles: list = []
        self._buf: list[dict] = [dict() for _ in self.blocks]

    # -- hook factories --
    def _store(self, i, key, t):
        self._buf[i][key] = t.detach().cpu() if self.to_cpu else t.detach()

    def _pre(self, i):
        def hook(mod, args):
            self._store(i, "x_in", args[0])
        return hook

    def _post(self, i):
        def hook(mod, args, out):
            self._store(i, "x_out", out)
        return hook

    def _sub(self, i, key):
        def hook(mod, args, out):
            self._store(i, key, out)
        return hook

    def __enter__(self):
        for i, blk in enumerate(self.blocks):
            self._handles.append(blk.register_forward_pre_hook(self._pre(i)))
            self._handles.append(blk.register_forward_hook(self._post(i)))
            self._handles.append(blk.adaLN_modulation.register_forward_hook(self._sub(i, "adaln")))
            self._handles.append(blk.attn.register_forward_hook(self._sub(i, "attn_out")))
            self._handles.append(blk.mlp.register_forward_hook(self._sub(i, "mlp_out")))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def snapshot(self) -> list[dict]:
        """Return the per-block captured dicts and reset the buffer for the next pass."""
        out = self._buf
        self._buf = [dict() for _ in self.blocks]
        return out
