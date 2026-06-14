# Phase 0 setup report — LeWM AEZ audit

**Host** `ai4ce-shannon` (2x RTX 3090, both verified free at launch); env
`.venv-ai4ce-shannon`; `torch==2.6.0+cu124`, `cuda.is_available()=True`,
`device_count()==2`.

## What was installed / vendored
- Vendored **verbatim** into `src/leworld_interp/lewm/`: `jepa.py`, `module.py`
  (+ `LICENSE`) from `github.com/lucas-maes/le-wm` @ commit
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (MIT), SHA-256 verified identical to
  source. The AEZ hooks will attach to these classes.
- Added via `uv add` (lock updated, torch pin preserved):
  `stable-worldmodel[env,train]==0.1.1`, `stable-pretraining==0.1.7`,
  `gdown`, `hdf5plugin==6.0.0` (needed to read the dataset's compression filters).
  Pulls `transformers==5.12.0`, `lightning`, `mujoco==3.9.0`, etc.

## Checkpoint (loaded `strict=True`)
- Source: HuggingFace **`quentinll/lewm-pusht`** (the official LeWM mirror — the
  Drive folder named in the prompt holds *baselines*, not LeWM). `weights.pt` is a
  pure `state_dict`; `config.json` gives exact dims.
  - `weights.pt`: 72,290,721 bytes, SHA-256
    `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`.
- Reconstructed with the **vendored** `jepa.JEPA` / `module.{ARPredictor,Embedder,MLP}`
  (state_dict keys identical to swm's `wm.lewm`) and loaded with `strict=True`.
- **One shim, validated:** `transformers 5.x` renamed the HF ViT encoder
  (`encoder.encoder.layer.N.attention.attention.query` → `encoder.layers.N.attention.q_proj`,
  `intermediate.dense`→`mlp.fc1`, `output.dense`→`mlp.fc2`, ...). A pure key rename of
  an identical architecture (`model.remap_legacy_vit_encoder`); 303→303 keys,
  **0 missing / 0 unexpected / 0 shape mismatch**, then `strict=True`. Correctness
  re-confirmed by the fidelity numbers below (a wrong remap would wreck the MSE).
- Model: **18.03M params** (encoder 5.50M, predictor 10.79M, action_encoder 0.16M,
  projector 0.79M, pred_proj 0.79M); 6 ConditionalBlocks, 16 heads, mlp_dim 2048.
  adaLN final-Linear ‖W‖_F per block = [5.81, 6.08, 6.50, 7.16, 7.45, 7.25] — well
  away from zero-init (preview of Measurement A).

## Dataset
- HuggingFace dataset repo `quentinll/lewm-pusht`:
  `pusht_expert_train.h5.zst` (13,136,247,974 bytes) → decompressed
  `pusht_expert_train.h5` (**46.3 GB**) under `DATA_ROOT/datasets/lewm-pusht/`.
- **18,685 episodes**, 2,336,736 steps; `pixels` uint8 (224,224,3); `action` dim **2**;
  also `proprio` (4), `state` (7). `frameskip=5`, so `action_encoder.input_dim =
  frameskip × action_dim = 5 × 2 = 10` (read from data + config, never hardcoded).

## Fidelity gate — PASS (eval(), fp32, seed 0)
Held-out clips via a seeded 90/10 clip split (seed 3072, mirrors training).
- **Teacher-forced next-embedding MSE** (== training `pred_loss`, 512 clips / 508 eps):
  **0.00813**; target emb energy/dim ≈ 1.01 ⇒ **relative MSE 0.80%**;
  per-position ≈ flat [0.00812, 0.00802, 0.00824]; **beats a persistence baseline
  (0.188) by 23.1×**.
- **Open-loop rollout** (model's own `rollout`, 256 clips / 254 eps, horizon 8):
  true-action final plan cost **30.0** vs within-batch **shuffled 329.5 (11.0×
  worse)**; normalized drift with true actions rises gently 0.0095 → 0.171 over 8
  steps while shuffled saturates near 1.74. The model genuinely uses action
  conditioning and the rollout/cost machinery is sane.

Artifacts: `results/phase0/fidelity_summary.json`, `results/phase0/fidelity.png`;
raw arrays `DATA_ROOT/activations/phase0/fidelity_arrays.npz`. Reproduce metrics/plot
without the model via `uv run python -m experiments.phase0_fidelity --from-cache`.

**Verdict:** model loaded faithfully; downstream AEZ measurements are trustworthy.
Stop here for review before Phase 1 (Measurement A, static gate audit).
