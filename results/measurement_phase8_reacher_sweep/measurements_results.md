# Phase 8 results: the depth law GENERALIZES across environments (reacher matches PushT)

Question: the depth law (action commitment at a roughly constant fraction of predictor depth,
about 0.33, MLP-routed) was swept only on PushT; Phase 5 confirmed the d6 point on reacher but not
the scaling. Does the full depth sweep on reacher reproduce the PushT fractions?

Setup: LeWM retrained from scratch on reacher at predictor depth {3, 6, 12, 18}, only
predictor.depth varied, identical released config (next-emb MSE + 0.09 SIGReg, AdamW lr 5e-5 wd
1e-3, bf16, frameskip 5), from a lossless RAM-resident zstd cache (full 9,800 train episodes,
episode-disjoint 200-episode val holdout, seed 3072, bit-exact validated). d3 and d6 trained on
fourier, d12 and d18 on shannon, on the SAME shared NAS cache (verified: 9,800 train / 200 val,
seed 3072, params/block 1.799M constant). All five models (the four trained depths plus the official
released reacher d6) were re-gated and re-measured uniformly on shannon's val cache. Measurements at
the readout token, eval() and fp32, mean ablation, N = 1000.

## Fidelity gate at the readout (N = 1500): all released-quality

| model | depth | params total | params/block | readout rel-MSE | skill | rollout shuf/true | gate |
|---|--:|--:|--:|--:|--:|--:|:--|
| official released d6 (ref) | 6 | 18.03M | 1.80M | 0.0068 | 79x | 90x | PASS |
| trained d3 | 3 | 12.64M | 1.80M | 0.0012 | 425x | 554x | PASS |
| trained d6 | 6 | 18.03M | 1.80M | 0.0010 | 530x | 669x | PASS |
| trained d12 | 12 | 28.82M | 1.80M | 0.0010 | 580x | 770x | PASS |
| trained d18 | 18 | 39.62M | 1.80M | 0.0008 | 709x | 884x | PASS |

All pass at the readout. The from-scratch models are in fact better than the official d6 at the
readout (they optimize next-emb MSE directly for many steps), and all use action conditioning
strongly (rollout shuffled-over-true 554x to 884x). A metric note: reacher position-0 (predict the
next frame from a single frame, no velocity) is intrinsically near-unsolvable and sits at rel about
0.065 for every model including the official one, so the all-position average (about 0.022 to 0.028)
is not a useful fidelity number; the readout is the right metric and the position commitment is
measured at. Per-block capacity is constant at 1.80M across depths (capacity confound controlled).

## Headline: commitment fraction is identical to PushT at every depth

| total depth | reacher commit block | reacher fraction | PushT fraction | match |
|--:|--:|--:|--:|:--|
| 3  | 1 | 0.33 | 0.33 | yes |
| 6  | 2 | 0.33 | 0.33 | yes |
| 12 | 4 | 0.33 | 0.33 (released), 0.42 (reduced) | yes |
| 18 | 7 | 0.39 | 0.39 (reduced) | yes |

The reacher commitment blocks {1, 2, 4, 7} are identical to the PushT blocks, and the fractions
{0.33, 0.33, 0.33, 0.39} fall exactly on the PushT values, all inside the 0.28 to 0.40 band. The
commitment grows with depth (relative, not a fixed absolute block), at the same fraction as PushT.

Robustness: the commitment block is seed-stable. Across three seeds at the key depths, d6 gives
commit 2 (fraction 0.333) every time, d12 gives commit 4 (0.333) every time, and d18 gives commit 7
(0.389) every time. Not a single-draw artifact.

Internal consistency: the trained reacher d6 (commit 2, fraction 0.33, MLP share 0.74) reproduces the
official released reacher d6, which on our val cache also gives commit 2, fraction 0.333, MLP share
0.744. So our reacher training pipeline matches the official released reacher at d6, exactly as the
Phase 6 retrained-d6 PushT gate did for PushT.

## Supporting signals

MLP routing holds across depth. Mean MLP share of the per-branch ablation effect: 0.74 (d3), 0.74
(d6), 0.67 (d12), 0.67 (d18). At d3 and d6 the share is cleanly above 0.7. At d12 and d18 the
attention-only damage is at the machine noise floor (about 1e-4), so the share of 0.67 should be read
as attention negligible, conditioning essentially all MLP-routed, with the exact ratio fragile (the
same noise-floor situation seen on PushT d12 at released scale). The preregistration committed this
exact allowance (share above 0.7 or attention at the noise floor), so MLP routing holds at every
depth.

D_l propagation. Near-linear at every depth (linear-fit R2 0.98 to 1.00). The deep-third slope is
below 1 at every depth (late-over-early 0.67, 0.53, 0.73, 0.81), so reacher shows the deep-third
propagation plateau across the whole sweep rather than only emerging at higher depth as on PushT
(where late-over-early fell from about 1.1 at d3/d6 to 0.56 at d18). This is a minor
environment-specific difference in the D_l shape detail; the core picture (near-linear propagation
with a deep-third sub-linearity) holds in both environments.

## Preregistered outcome: GENERALIZES

The committed GENERALIZES prediction held on all three of its parts:
- The reacher commitment fractions cluster near 0.33 (within 0.28 to 0.40) and scale with depth, the
  absolute block growing 1, 2, 4, 7, matching PushT.
- MLP routing holds (share above 0.7 at d3/d6, attention at the noise floor at d12/d18, both allowed
  by the committed prereg).
- The deep-third propagation plateau is present (in fact at every depth on reacher).

No falsifiers fired. The PARTIAL outcome (relative law with a constant clearly off 0.33, for example
0.25 or 0.45) did not occur: the constant is 0.33, identical to PushT. The DIVERGES outcome
(commitment pinned to a fixed absolute block, or routing broken with attention above the floor) did
not occur: commitment grows 1 to 7 with depth and routing holds.

Conclusion: the commitment fraction holds at about 0.33 across both the depth axis and two
environments. The relative-depth, MLP-routed commitment law is environment-general, not
PushT-specific.

## Artifacts

- measurements_summary.json: per-depth summaries, robustness, verdict (GENERALIZES).
- fidelity_table.{md,json}: the readout fidelity and parameter table.
- phase8_reacher_sweep.png: reacher cumulative-ablation curves and the fraction-vs-depth comparison
  to PushT.
- preregistration.md: the committed generalizes / partial / diverges prediction.
- Raw arrays: DATA_ROOT/activations/measurement_phase8_reacher_sweep/{depth_{3,6,12,18}.json,
  official_d6.json, robustness.json}. Checkpoints: DATA_ROOT/checkpoints/depth_scaling_reacher
  (d3/d6 from fourier, d12/d18 from shannon) and the official lewm-reacher.
- Reproduce: uv run python -m experiments.measurement_phase8 --from-cache --depths 3 6 12 18;
  uv run python -m experiments.plot_phase6 --phase8.
