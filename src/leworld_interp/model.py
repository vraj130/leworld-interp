"""Build the vendored LeWM (JEPA) model and load the released weights strictly.

The released checkpoint (HuggingFace ``quentinll/lewm-pusht``) ships a pure
``state_dict`` in ``weights.pt`` plus a ``config.json`` describing the model.
We reconstruct the model from the *vendored* ``jepa.JEPA`` / ``module.*`` classes
(byte-identical to the upstream ``stable_worldmodel.wm.lewm`` definitions, so the
state_dict keys match) and load with ``strict=True`` so any drift fails loudly.

Everything here keeps the model in ``eval()`` and (by default) fp32, per the
audit invariants: BatchNorm1d running stats must stay frozen across the paired
counterfactual passes, and early-layer divergences must not be read at bf16
precision.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from .lewm.jepa import JEPA
from .lewm.module import MLP, ARPredictor, Embedder


def set_seed(seed: int) -> int:
    """Seed python, numpy and torch (incl. CUDA). Returns the seed for logging."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def _model_kwargs(block: dict) -> dict:
    """Drop Hydra-special keys (``_target_``, ``_partial_``) so a config block
    can be splatted straight into a constructor."""
    return {k: v for k, v in block.items() if not k.startswith("_")}


# The released checkpoint was saved with a classic-HF-ViT encoder
# (``encoder.encoder.layer.N.attention.attention.query`` ...). transformers >= 5.x
# refactored ViT to ``encoder.layers.N.attention.q_proj`` etc. This is a pure key
# rename of a numerically identical architecture; the per-layer sub-name map below
# turns the old keys into the names the installed model expects. Validated by a
# bijective, zero-mismatch ``load_state_dict(strict=True)`` plus the Phase 0
# fidelity gate (a wrong remap would wreck the next-embedding MSE).
_VIT_LAYER_PREFIX = re.compile(r"^encoder\.encoder\.layer\.(\d+)\.(.*)$")
_VIT_SUBNAME_MAP = (
    ("attention.attention.query", "attention.q_proj"),
    ("attention.attention.key", "attention.k_proj"),
    ("attention.attention.value", "attention.v_proj"),
    ("attention.output.dense", "attention.o_proj"),
    ("intermediate.dense", "mlp.fc1"),
    ("output.dense", "mlp.fc2"),
)


def _remap_vit_key(key: str) -> str:
    m = _VIT_LAYER_PREFIX.match(key)
    if not m:
        return key
    tail = m.group(2)
    for old, new in _VIT_SUBNAME_MAP:
        tail = tail.replace(old, new)
    return f"encoder.layers.{m.group(1)}.{tail}"


def remap_legacy_vit_encoder(state_dict: dict) -> dict:
    """Rename a classic-HF-ViT encoder state_dict to the refactored naming.

    No-op if the classic ``encoder.encoder.layer.*`` keys are absent.
    """
    if not any(k.startswith("encoder.encoder.layer.") for k in state_dict):
        return state_dict
    return {_remap_vit_key(k): v for k, v in state_dict.items()}


def build_lewm(
    config_path: str | Path,
    weights_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    strict: bool = True,
) -> tuple[JEPA, dict]:
    """Construct the LeWM model from ``config.json`` and load ``weights.pt``.

    Returns ``(model, cfg)`` with the model in ``eval()`` mode on ``device``.
    Raises (via ``load_state_dict(strict=True)``) on any missing/unexpected key.
    """
    import stable_pretraining as spt  # heavy; import lazily

    cfg = json.loads(Path(config_path).read_text())

    enc = cfg["encoder"]
    encoder = spt.backbone.utils.vit_hf(
        enc["size"],
        patch_size=enc["patch_size"],
        image_size=enc["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    def mlp(key: str) -> MLP:
        c = cfg[key]
        return MLP(
            input_dim=c["input_dim"],
            output_dim=c["output_dim"],
            hidden_dim=c["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**_model_kwargs(cfg["predictor"])),
        action_encoder=Embedder(**_model_kwargs(cfg["action_encoder"])),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )

    try:
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - fall back for non-tensor-only payloads
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and not any(
        k.startswith("encoder.") for k in sd
    ):
        sd = sd["state_dict"]

    sd = remap_legacy_vit_encoder(sd)
    model.load_state_dict(sd, strict=strict)  # loud on any mismatch
    model = model.to(device=device, dtype=dtype).eval()
    return model, cfg


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Total and per-top-level-submodule parameter counts (for the setup report)."""
    out = {"total": sum(p.numel() for p in model.parameters())}
    for name, child in model.named_children():
        out[name] = sum(p.numel() for p in child.parameters())
    return out
