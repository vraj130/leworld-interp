# Phase 7 measurements: the depth law holds at released scale (VERIFIED)

Question: the Phase 6 depth law (action commitment sits at a roughly constant fraction of
predictor depth, about 0.37, not a fixed early block) was established in the reduced-data regime
and verified at released quality only at depth 6. Does the released-scale depth-12 model (rel-MSE
0.0085, fidelity PASS) put its commitment at the same fraction, or does data scale move it toward
the absolute prediction (block about 2, fraction about 0.17)?

All measurements at the readout token, eval() and fp32, mean ablation, N = 1000 held-out val clips.

## Headline: VERIFIED, and tighter than the reduced regime

| model | data scale | commitment block | fraction of depth | MLP share | D_l linear R2 | D_l late/early |
|---|---|--:|--:|--:|--:|--:|
| released d6 (audit reference) | released | 2 | 0.33 | 0.91 | ~1.0 | ~1.0 |
| **released d12 (this phase)** | **released** | **4** | **0.33** | **0.95** | 0.99 | 0.59 |
| reduced d12 (Phase 6) | reduced | 5 | 0.42 | 0.77 | 0.99 | 0.74 |

The released-scale d12 commits at block 4, fraction 0.33, identical to the released d6 fraction and
firmly in the relative band, decisively above the absolute prediction (block about 2, fraction about
0.17). So the depth law is now confirmed at TWO released-scale points (d6 and d12) that span the
absolute-versus-relative divergence. If anything, released-scale training SHARPENS the law: the
fraction is a tighter constant at released scale (0.33 at both d6 and d12) than the reduced regime
suggested (0.42 at d12). The reduced regime was already relative; released scale makes the constant
cleaner, not different in kind.

Cumulative damage / full for released d12 (l = 0..11): 1.00, 0.93, 0.82, 0.71, 0.59, 0.47, 0.36,
0.26, 0.17, 0.09, 0.04, 0.01. It crosses the 50% bar between block 4 (0.59) and block 5 (0.47), so
the commitment block (deepest l with at least half the full damage) is 4 with comfortable margin,
not a knife-edge. The deep tail is idle (last block contributes 0.011 of full).

## Robustness: the commitment block is not a threshold artifact

The commitment block was re-measured across seeds and sample sizes. All four runs agree exactly:

| seed | N | commitment block | fraction | MLP share |
|--:|--:|--:|--:|--:|
| 0 | 1000 | 4 | 0.333 | 0.95 |
| 1 | 1000 | 4 | 0.333 | 0.86 |
| 2 | 1000 | 4 | 0.333 | 0.92 |
| 0 | 1500 | 4 | 0.333 | 0.86 |

Block 4 / fraction 0.333 is stable; the MLP share varies in 0.86 to 0.95 (see the noise-floor note).

## MLP routing: dominant, with the d12 noise-floor caveat

Per-block attn-only mean-ablation damage for released d12 is at the numerical floor (values about
plus or minus 1e-4, minimum -0.00008, several negative, meaning ablating the attention conditioning
chunks has no measurable effect beyond noise). So the MLP share of the per-branch sum (0.95) should
be read as "the MLP branch carries essentially all measurable per-block action damage and the
attention branch is negligible," not as a precise 0.95 ratio. The conclusion (MLP-routed
conditioning, attention negligible) is robust; the exact share is fragile because its denominator is
at machine epsilon. This matches and strengthens the released-d6 (0.91) and reduced-d12 (0.77)
MLP-dominance.

## D_l propagation: the emergent plateau persists at released scale

The action-perturbation propagation D_l stays near-linear (linear-fit R2 = 0.99) but its deep-third
slope is 0.59 of the early-third slope, the same deceleration seen at reduced-regime d12 (0.74) and
d18 (0.56). So the saturation that emerges only at higher depth is a real property of the model at
released scale, not a reduced-regime artifact.

## Transfer logic (why this comparison is sound)

The released d6 is the official released checkpoint; the released d12 is trained by our pipeline at
released-data scale. The Phase 6 retrained-d6 consistency gate already established that our training
recipe reproduces the official released-d6 audit (commitment block 2, MLP routing), so comparing the
official d6 against our released-scale d12 is sound. We did not separately train a released-scale d6,
since the gate already pins our recipe to the official model at d6.

## What this changes for the memo

The single largest reviewer target, that the depth law lived in the reduced-data regime and was
released-verified only at d6, is closed. The law is now directly verified at d6 and d12 at released
scale, and the commitment fraction is constant (0.33) at both. The remaining honest caveat is only
that intermediate depths (d3, d18) are verified in the reduced regime, not at released scale.

## Artifacts

- measurements_summary.json: released-d12 summary, reduced-d12, released-d6 reference, robustness,
  attn noise-floor, verdict.
- phase7_released_d12.png: commitment-depth curves and the fraction-vs-depth comparison.
- training_and_fidelity.md, fidelity_table.json: the released-scale training and fidelity gate.
- Raw arrays: DATA_ROOT/activations/measurement_phase7_released_d12/{depth_12_released.json,
  robustness.json}. Reproduce: `uv run python -m experiments.measurement_phase7 --from-cache`.
