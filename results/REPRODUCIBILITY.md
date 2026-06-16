# Reproducibility matrix (frozen experimental record)

Every phase regenerates its plots and printed summaries from saved arrays under
`DATA_ROOT/activations/` (catalogued in `results/DATA_CATALOG.md`) with **no GPU and no retraining**,
via a `--from-cache` flag. All 13 reproductions were verified green on 2026-06-16 (each command run
with `CUDA_VISIBLE_DEVICES=""` to confirm no GPU dependency). Benign warnings (non-writable-numpy,
JAX/INFO logging, matplotlib marker notes) are not failures.

| phase | command (`uv run python -m ...`) | status | reproduced headline |
|---|---|:--|---|
| 0 fidelity | `experiments.phase0_fidelity --from-cache` | PASS | TF rel-MSE 0.0080, 23.1x persistence, rollout shuf/true 10.98x |
| 1 (A) static gate | `experiments.measurement_a --from-cache` | PASS | gate late/early x1.41; "gates grow into mid/late depth" |
| 2 (B,C) injection / D_l | `experiments.measurement_bc --from-cache` | PASS | D_l full-swap 0.115..0.434; capacity-vs-use D_l x3.90 |
| 3 (D) ridge probes | `experiments.measurement_d --from-cache` | PASS | action R2 ~0.68 flat, consequence 0.32->0.43, no crossover |
| 4 (E) ablation | `experiments.measurement_e --from-cache` | PASS | commitment depth 2, MLP fraction 0.54, prereg holds |
| 4 (E) probe check | `experiments.measurement_e_probecheck --from-cache` | PASS | late-block recovery 0.59 demb / up to 0.86 dstate -> SOFTEN |
| 5 reacher replication | `experiments.phase5_replication --from-cache --env reacher` | PASS | rel 0.007 (80x), commit 2, MLP 0.75, REPLICATES |
| 6 PushT depth sweep | `experiments.measurement_phase6 --from-cache --depths 3 6 12 18` | PASS | retrained-d6 gate PASS (commit 2, MLP 0.86) |
| 6 fidelity table | `experiments.depth_fidelity --from-cache` | PASS | 4 depths all PASS (N 1500) |
| 7 PushT released d12 | `experiments.measurement_phase7 --from-cache` | PASS | commit block 4, frac 0.333, VERIFIED |
| 8 reacher released sweep | `experiments.measurement_phase8 --from-cache --depths 3 6 12 18` | PASS | fractions [0.33,0.33,0.33,0.39], VERDICT GENERALIZES |
| 8 fidelity table | `experiments.sweep_fidelity --from-cache --env reacher` | PASS | all 5 rows passed (official d6 0.0067, trained 0.0008..0.0012 readout) |
| all depth-scaling plots | `experiments.plot_phase6 --gate --sweep --phase7 --phase8` | PASS | regenerated 4 PNGs from cache |

## Notes

- The `--from-cache` paths read only `DATA_ROOT/activations/<phase>/` arrays and `results/` JSON; they
  never touch a checkpoint, the raw h5, or a GPU. Retraining and re-measuring (which do need a GPU and
  the caches) are the non-cached paths and are not required to reproduce any figure or number.
- Plot PNGs differ byte-for-byte between runs (matplotlib embeds a timestamp) but are visually and
  numerically identical; the JSON summaries are deterministic from the cached arrays.
- Training entry points (not needed for reproduction): `experiments/train_depth.py` (Phase 6 reduced),
  `experiments/train_depth_released.py` (Phase 7 PushT released d12), `experiments/train_sweep.py`
  (Phase 8 reacher sweep, env-parameterized). Fidelity gates: `experiments/depth_fidelity.py`,
  `experiments/sweep_fidelity.py`. One-off IO benchmark: `experiments/bench_io.py`.
- Cross-machine provenance: the Phase 8 reacher d3 and d6 were trained on fourier
  (`experiments/{train_sweep_fourier,sweep_fidelity_fourier}.py`, logs under `results/fourier/`) and
  d12, d18 on shannon, both from the same shared `reacher_sweep_cache`; all five reacher models were
  re-gated and re-measured uniformly on shannon, so the reported numbers are single-pipeline.
