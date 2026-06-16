# Phase 7: released-scale depth-12 training and fidelity gate (report before measurements)

The Phase 6 depth law (commitment depth scales with total depth at a roughly constant fraction
about 0.37) was established in a reduced-data regime (rel-MSE about 0.025) and verified at
released quality (rel-MSE about 0.007) only at depth 6. This phase closes that gap by training
ONE depth-12 model at released-data scale, so its commitment depth can be measured at released
quality. d12 is the choice because it is where the absolute and relative hypotheses diverge most
(absolute predicted block about 2, relative predicted block about 4; reduced-regime d12 gave
block 5 at fraction 0.42). Verifying d12 at released scale gives two released-scale points (d6 and
d12) that span the divergence.

## Infrastructure (what made released-scale training feasible here)

shannon GPU 1 had recovered (both RTX 3090s healthy); the per-host venv was broken and was rebuilt
from the committed lock with `uv sync --frozen`. The box has only 62 GiB RAM and no local fast
disk, so the Phase 6 raw-uint8 cache does not scale: the full 18,685-episode dataset is 352 GB of
raw frames, far beyond RAM, and a direct HDF5 dataloader runs at 92 windows/s (1.39 s/step) even
warm, because the pixels are stored in 100-frame compressed chunks so each random read amplifies to
a 15 MB chunk decompress. The fix is a lossless per-frame zstd cache: the full 18,485-episode train
set (the same 200 episodes as Phase 6 held out for val) compresses to 24.7 GB, which is RAM-resident
and page-cached, and decodes at about 9800 frames/s/core across 24 workers. It was validated
bit-exact against the raw frames (max pixel difference 0), so it is faithful to the released uint8
data, not lossy. With it, training is GPU-bound at about 240 ms/step.
Code: src/leworld_interp/pixelcache.py (build_cache_zstd, ZstdCachedWindows);
experiments/train_depth_released.py (resumable, eval and checkpoint every 5k steps, early-stop when
val rel-MSE reaches the released band and plateaus).

## Training (faithful to the released objective)

Predictor depth 12, everything else at released settings: next-embedding prediction MSE plus 0.09
SIGReg, AdamW lr 5e-5 wd 1e-3, bf16 autocast, gradient clip 1.0, history 3, num_preds 1, batch 128,
warmup-then-cosine. Trained from scratch on all 18,485 non-val episodes (released scale), seed 3072.
Early-stopped at step 80,000 (6.3 hours, GPU-bound) after three consecutive evals plateaued in the
released band. Val descent: rel-MSE 0.119 (5k), 0.045 (10k), 0.0186 (20k), 0.0138 (30k), 0.0100
(50k), 0.0080 (60k), 0.0079 (70k), 0.0078 (75k), 0.0078 (80k). The best checkpoint (rel 0.0078) is
the canonical released d12. Checkpoint: DATA_ROOT/checkpoints/depth_scaling_released/depth_12.

## Fidelity gate: PASS at released quality (N = 1500 held-out val clips)

| model | data scale | episodes | params | TF rel-MSE | skill vs persist | rollout shuf/true | gate |
|---|---|--:|--:|--:|--:|--:|:--|
| released d6 (audit reference) | released | 18,685 | 18.03M | about 0.007 | high | high | reference |
| **released d12 (this phase)** | **released** | **18,485** | **28.82M** | **0.0085** | **20.2x** | **15.3x** | **PASS** |
| reduced d12 (Phase 6) | reduced | 1,500 | 28.82M | 0.0258 | 7.9x | 6.1x | (for contrast) |

The released d12 reaches rel-MSE 0.0085, which is 3.0x better than the reduced-regime d12 (0.0258)
and within about 20% of the released d6 reference (0.007). The open-loop rollout cost under shuffled
actions is 15.3x the true-action cost, up from 6.1x in the reduced regime, so the model uses the
action conditioning far more strongly, as expected at released quality. This decisively clears the
acceptance bar (materially better than the reduced regime, approaching released quality, not the
0.025 plateau).

## Status

Training and fidelity are complete and the released d12 passes at released-approaching quality.
STOPPED here for review before the three measurements (cumulative-ablation commitment depth,
per-branch MLP share, D_l shape), as instructed. On go, those run at N = 1000, readout token,
eval and fp32, mean ablation, and the headline comparison is the released-scale d12 commitment
fraction against the reduced-regime d12 (0.42) and the released d6 (0.33).

Artifacts: results/measurement_phase7_released_d12/{training_and_fidelity.md, fidelity_table.json};
checkpoint and zstd cache under DATA_ROOT. Reproduce the gate:
`uv run python -m experiments.depth_fidelity --ckpt-root DATA_ROOT/checkpoints/depth_scaling_released --depths 12`.
