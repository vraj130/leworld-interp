# DATA_ROOT catalogue (frozen experimental record)

All large artifacts live under `DATA_ROOT = /mnt/NAS/data/vg2097/leworld-interp-data` (set in the
repo `.env`), never in the repo. Code and small artifacts (memos, preregistrations, summaries, plots)
live in the repo under `results/`. This file is the index from the repo into DATA_ROOT so the record
is citable. Totals: checkpoints 3.6 GB, datasets/caches 247 GB, activations 79 MB.

## Raw datasets (`DATA_ROOT/datasets/`)

| path | size | contents |
|---|--:|---|
| lewm-pusht/pusht_expert_train.h5 | 44 GB | released PushT expert dataset, 18,685 episodes, 2,336,736 frames (224x224x3 uint8, 100-frame zstd chunks), action_dim 2 |
| lewm-reacher/reacher.h5 | 93 GB | released reacher dataset, 10,000 episodes, 2,010,000 frames (201/episode), action_dim 2 |

## Training caches (`DATA_ROOT/datasets/`)

Built from the raw h5 to make training feasible on this 62 GB-RAM, no-local-disk box (random h5 reads
are I/O-bound at ~1.4 s/step; a RAM-resident cache is GPU-bound at ~0.25 s/step).

| path | size | episodes | frames | format | used by |
|---|--:|--:|--:|---|---|
| pusht_train_cache/{train,val} | 30 GB | 1500 / 200 | 186,541 / 25,737 | raw uint8 memmap | Phase 6 PushT reduced sweep |
| pusht_released_cache/train | 24 GB | 18,485 | 2,310,999 | lossless zstd per-frame | Phase 7 PushT released d12 (val reuses pusht_train_cache/val) |
| reacher_sweep_cache/{train,val} | 30 GB | 9800 / 200 | 1,969,800 / 40,200 | lossless zstd per-frame | Phase 8 reacher released sweep |

All caches use the episode-disjoint val split (200 episodes, seed 3072). zstd caches are bit-exact vs
the h5 frames (validated). Cache builders: `src/leworld_interp/pixelcache.py` (`build_cache`,
`build_cache_zstd`); window datasets `CachedWindows`, `ZstdCachedWindows`.

## Checkpoints (`DATA_ROOT/checkpoints/`)

Released references (official LeWM checkpoints, ViT old-naming, loaded via the `remap_legacy_vit_encoder`
shim in `src/leworld_interp/model.py`):

| path | model | role |
|---|---|---|
| lewm-pusht/ | released PushT predictor depth 6 | the original audit subject (Phases 0 to 4) |
| lewm-reacher/ | released reacher predictor depth 6 | Phase 5 replication subject; Phase 8 d6 reference |
| reacher_official_ref/depth_6/ | copy of lewm-reacher in depth_N layout | convenience copy so measure_depth can load the official reacher d6 |

Retrained checkpoints (our pipeline, new ViT naming, loadable via `build_lewm`). best/ holds the
lowest-val checkpoint, promoted to the canonical weights.pt; latest.pt is the resumable full state.
Reported rel-MSE below is the ALL-POSITION val metric logged during training; the READOUT fidelity
(the metric used for the gate and measurements) is much lower for reacher (see Phase 8 note).

| path | env | scale | depths (best all-pos rel-MSE) | phase |
|---|---|---|---|---|
| depth_scaling/depth_{3,6,12,18} | PushT | reduced (1500 eps, 30k steps) | d3 0.020, d6 0.024, d12 0.023, d18 0.022 | Phase 6 |
| depth_scaling_released/depth_12 | PushT | released (18,485 eps, 80k steps) | d12 0.0078 | Phase 7 |
| depth_scaling_reacher/depth_{3,6,12,18} | reacher | released (9800 eps, ~185-195k steps) | d3 0.022, d6 0.022, d12 0.021, d18 0.023 | Phase 8 (canonical sweep) |
| depth_scaling_reacher_fourier/depth_{3,6} | reacher | released | d3 0.022, d6 0.022 | Phase 8 original fourier output, superseded by (copied into) depth_scaling_reacher/depth_{3,6} |

Per-block predictor capacity is constant at ~1.80M parameters across all depths (capacity confound
controlled): total params d3 12.64M, d6 18.03M, d12 28.82M, d18 39.62M.

## Activations / cached arrays (`DATA_ROOT/activations/`)

Per-phase raw arrays backing each plot and summary, so every figure regenerates via `--from-cache`
without a GPU. 79 MB total.

| path | phase | contents |
|---|---|---|
| phase0/ | Phase 0 | fidelity-gate arrays |
| measurement_a/ | Phase 1 (A) | static adaLN gate norms |
| measurement_bc/ | Phase 2 (B,C) | injection ratios, D_l propagation |
| measurement_d/ | Phase 3 (D) | layerwise ridge-probe cache |
| measurement_e/ | Phase 4 (E) | per-block/cumulative/per-branch ablation arrays |
| measurement_phase5_reacher/ | Phase 5 | reacher replication arrays |
| measurement_phase6_depthscaling/ | Phase 6 | per-depth PushT reduced-sweep arrays |
| measurement_phase7_released_d12/ | Phase 7 | released PushT d12 arrays + robustness |
| measurement_phase8_reacher_sweep/ | Phase 8 | per-depth reacher arrays + robustness + official_d6 |

## Reproduce

Every phase regenerates its plots and summaries from these arrays with no GPU and no retraining; see
`results/REPRODUCIBILITY.md` for the exact per-phase `--from-cache` command and its verified status.
Retraining (not needed for reproduction) uses `experiments/{train_depth,train_depth_released,train_sweep}.py`
against the caches above.
