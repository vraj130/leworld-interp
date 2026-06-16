# Phase 8 reacher depth sweep: training and fidelity gate (report before measurements)

Goal: show the depth law (commitment at a roughly constant fraction of predictor depth) is not
PushT-specific. Following the focused plan, we train reacher at predictor depth 12 (the absolute
vs relative divergence depth) and depth 18 to released quality, and pair them with the official
released reacher depth 6 (already released-quality, Phase 5 fraction 0.33). That gives three
released-scale reacher points (d6, d12, d18) spanning a 3x depth range, mirroring the Phase 7
PushT treatment.

## Training (faithful to released config, only predictor.depth varies)

From scratch on the full 9,800 non-val reacher episodes (episode-disjoint 200-episode val holdout,
seed 3072), lossless RAM-resident zstd cache (31 GB, bit-exact validated) so training is GPU-bound
at about 0.28 s/step. Released objective and hyperparameters: next-emb MSE + 0.09 SIGReg, AdamW lr
5e-5 wd 1e-3, bf16, grad-clip 1.0, history 3, num_preds 1, frameskip 5, batch 128. d12 and d18 ran
in parallel (one per GPU) and converged by about 185k steps (about 15 h each).

Reacher trains slower than PushT and needs roughly 3x more steps. A correctness check confirmed the
pipeline: the official released reacher scores rel-MSE 0.0066 (about 79x persistence) on our val
cache, exactly matching the Phase 5 reference, so the slow convergence is genuine, not a bug.

## A metric note (important): readout vs all-position

The teacher-forced loss is evaluated at three context positions. Position 0 predicts the next frame
from a single frame, with no velocity information, which for the reacher arm is intrinsically close
to unsolvable: every model, including the official released reacher, sits at about rel 0.065 to 0.071
there. The all-position average is therefore dominated by position 0 (official d6 all-position rel
0.028; our d12 all-position 0.022), and is not a good fidelity measure for reacher. The READOUT
position (predict the next frame from three frames of history plus the action), the last position,
is the standard fidelity metric used in Phase 0/5/6/7 and the exact position where commitment is
measured. All numbers below are at the readout.

## Fidelity gate at the readout (N = 1500 held-out val clips): all PASS

| model | scale | depth | params total | params/block | readout rel-MSE | skill vs persist | rollout shuf/true | gate |
|---|---|--:|--:|--:|--:|--:|--:|:--|
| official released d6 (reference) | released | 6 | 18.03M | 1.80M | 0.0068 | 79x | 90x | PASS |
| reacher d12 (this phase) | released | 12 | 28.82M | 1.80M | 0.0010 | 580x | 770x | PASS |
| reacher d18 (this phase) | released | 18 | 39.62M | 1.80M | 0.0008 | 709x | 884x | PASS |

All three pass decisively at the readout. The trained d12 and d18 are in fact better than the
official d6 at the readout (rel 0.0010 and 0.0008 vs 0.0068), because the deeper predictors model the
near-deterministic reacher next-frame transition more precisely. Per-block capacity is constant at
1.80M parameters across depths, so any commitment-fraction difference across depths is a depth
effect, not a capacity effect (the parameter-count confound is controlled).

Earlier in training the all-position val metric read about rel 0.022 (about 20x persistence), which
looked like a plateau above released quality; the per-position breakdown shows this is the
irreducible position-0 floor shared by the official released model, and that at the readout the
trained models are released-quality. So the focused plan's models are not merely skill-matched to
the PushT released points, they are released-quality at the readout.

## Status

Training and fidelity are complete; all three reacher points pass at the readout at released quality.
STOPPED here for review before the measurements (cumulative-ablation commitment depth and fraction,
per-branch MLP share, D_l shape) on d6 (official), d12, and d18. The preregistration
(generalizes / partial / diverges) is committed in
results/measurement_phase8_reacher_sweep/preregistration.md. On go, the measurements run at N = 1000,
readout token, eval and fp32, mean ablation, and the headline is whether the reacher fractions
cluster near the PushT constant (about 0.33).

Artifacts: results/measurement_phase8_reacher_sweep/{fidelity_table.md, fidelity_table.json,
preregistration.md}; checkpoints DATA_ROOT/checkpoints/depth_scaling_reacher/depth_{12,18} and the
official DATA_ROOT/checkpoints/lewm-reacher; zstd cache DATA_ROOT/datasets/reacher_sweep_cache.
Reproduce the gate: uv run python -m experiments.sweep_fidelity --env reacher --depths 12 18.
