"""Canonical filesystem locations for the leworld-interp AEZ audit.

Code and small version-controlled artifacts live under the repo; large
artifacts (checkpoints, datasets, cached activations, probe weights) live
under ``DATA_ROOT`` (read from ``.env``). See CLAUDE.md for the rationale.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def data_root() -> Path:
    dr = os.environ.get("DATA_ROOT")
    if not dr:
        raise RuntimeError(
            "DATA_ROOT is not set. Expected it in the repo .env file "
            "(see CLAUDE.md) or in the environment."
        )
    return Path(dr)


DATA_ROOT = data_root()

# --- large artifacts (under DATA_ROOT) ---
CHECKPOINTS = DATA_ROOT / "checkpoints"
DATASETS = DATA_ROOT / "datasets"
ACTIVATIONS = DATA_ROOT / "activations"
PROBES = DATA_ROOT / "probes"

# --- small version-controlled artifacts (under the repo) ---
RESULTS = REPO_ROOT / "results"

# --- canonical released artifacts for the PushT model ---
LEWM_PUSHT_DIR = CHECKPOINTS / "lewm-pusht"
LEWM_PUSHT_CONFIG = LEWM_PUSHT_DIR / "config.json"
LEWM_PUSHT_WEIGHTS = LEWM_PUSHT_DIR / "weights.pt"
PUSHT_H5 = DATASETS / "lewm-pusht" / "pusht_expert_train.h5"


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
