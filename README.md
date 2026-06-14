# leworld-interp

Mechanistic interpretability of action conditioning in **LeWM (LeWorldModel)**, a
roughly 15M-parameter, end-to-end JEPA-based video world model.

## Research focus

This project investigates the **Action Emergence Zone (AEZ) hypothesis**: the claim
that there exists a specific depth in LeWM's predictor where action conditioning
transitions from loosely mixed input signal into causally committed latent
consequence. The earliest planned verification is an audit of whether LeWM's
**AdaLN** conditioning preserves the action signal across predictor depth or washes
it out.

The work is action-conditioned-predictor interpretability, which is architecturally
distinct from prior perceptual-physics interpretability on unconditioned encoders.
Findings are intended to map to actionable insights for practitioners building
JEPA-style world models.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for all package and environment
management. Do not use pip, conda, or poetry.

Environments are per-machine and named after the host (for example
`.venv-ai4ce-shannon`), because compiled environments are not portable across the
shared AI4CE workstations even though storage is shared over NAS.

```bash
# Per-machine, first time only: make uv name the env after the host
echo 'export UV_PROJECT_ENVIRONMENT=".venv-$(hostname -s)"' >> ~/.bashrc
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
source ~/.bashrc

# Build the environment for this machine (reads uv.lock)
uv sync

# Local config
cp .env.example .env
```

Run anything through uv so the correct per-host environment is used automatically:

```bash
uv run python -m <module>
```

## Layout

Code lives under the repo root. Data, checkpoints, and cached activations live
outside the repo under the data root, referenced via `DATA_ROOT` in `.env`.

- Repository root: `/mnt/NAS/home/vg2097/leworld-interp`
- Data root: `/mnt/NAS/data/vg2097/leworld-interp-data`

## Hardware

Primary development is on `shannon` (dual RTX 3090, 24 GB each). The PyTorch build is
the `cu124` wheel set. Always check `nvidia-smi` before launching, since the machines
are shared.

## Contributing notes

See `CLAUDE.md` for the full operating rules (machine selection, uv workflow, path
conventions). Dependencies change only through `uv add` / `uv remove`, and `uv.lock`
is committed and never hand-edited.

## Reference

LeWM paper: arXiv 2603.19312 (Maes, Le Lidec, Scieur, LeCun, Balestriero).