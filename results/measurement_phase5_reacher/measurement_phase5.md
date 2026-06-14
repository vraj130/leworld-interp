# Phase 5 replication on reacher (3D control)

Env choice: **reacher** (DMC reacher, `swm/ReacherDMControl-v0`), chosen over cube because
its low-dimensional continuous action (action_dim 2) and contact-free dynamics give the
cleanest replication, as instructed. Same released-checkpoint route as PushT: HF
`quentinll/lewm-reacher` weights.pt (strict load via the vendored JEPA + ViT remap) and the
`reacher.tar.zst` dataset (extracted to a 98.9 GB `reacher.h5`, 10,000 episodes, 2.01M steps,
`action_dim=2`, frameskip 5). Scope: only the three least-ambiguous signals, fidelity-gated.
N = **1000 val clips**, `eval()` + fp32, seed 0. Raw arrays:
`DATA_ROOT/activations/measurement_phase5_reacher/p5_arrays.npz`.

## Fidelity gate (must pass before trusting any number) — PASS

Teacher-forced next-emb MSE **0.00663** (relative **0.7%**, **80x** a persistence baseline);
open-loop rollout cost true **3.62** vs within-batch shuffled **329.5** (**91x**). The model
is loaded faithfully on reacher; replication numbers are trustworthy.

## (1) C: D_l action-perturbation propagation (readout)

| after block | 0 | 1 | 2 | 3 | 4 | 5 |
|--|--|--|--|--|--|--|
| D_l full-swap | 0.250 | 0.358 | 0.444 | 0.529 | 0.602 | 0.638 |
| D_l final-swap | 0.159 | 0.273 | 0.375 | 0.469 | 0.546 | 0.585 |

**Monotonic, roughly linear growth (~2.5x over depth), no decay** — same qualitative shape as
PushT (which grew ~3.9x). Both swap variants monotonic. Confirmed.

## (2) E: cumulative mean-ablation (ablate blocks >= l) — commitment depth

`0.538, 0.431, 0.289, 0.166, 0.061, 0.007` (fraction of full: 1.00, 0.80, 0.54, 0.31, 0.11,
0.01). Monotonic, front-loaded. **Commitment depth = 2** at the 50%-of-full bar, identical to
PushT (k=2). Per-block marginal contributions are roughly equal for blocks 0 to 3 then taper,
and **block 5 is near-idle** (0.007). Same early-graded pattern as PushT. Confirmed.

## (3) E: per-branch MLP-routing

MLP-chunk ablation dominates attn-chunk ablation at every block; **MLP share of the branch
sum = 0.75 (> 0.7)**. The conditioning is MLP-routed on reacher too. Confirmed.

## Acceptance: REPLICATES

Same qualitative verdict row (early graded commitment, MLP-routed), commitment depth 2 (within
0 of PushT's k=2), MLP share 0.75 > 0.7, D_l monotonic. The early-graded-MLP-routed-commitment
finding is **not PushT-specific**; it holds across a 2D pushing task and a 3D continuous-control
reacher. Reproduce with `uv run python -m experiments.phase5_replication --env reacher --from-cache`.
