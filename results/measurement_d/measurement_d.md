# Measurement D — layerwise ridge probes (Phase 3)

4000 train / 1000 val held-out PushT clips, `eval()` + fp32, seed 0. At each predictor
block output (and the pre-block-0 input as an anchor), at the **readout token**, linear
RidgeCV probes (standardized features, val R²) decode two things from the residual:
raw action (the 10-dim conditioning `action[2]` = frameskip×action_dim, read from data)
and consequence (next-state embedding delta `emb[3]-emb[2]`, 192-dim; plus physical
state delta, 7-dim). Probe inputs cached at `DATA_ROOT/activations/measurement_d/probe_cache.npz`;
weights at `probe_weights.npz`. `--from-cache` refits without the model.

## Results (validation R²)

| residual | input(−1) | b0 | b1 | b2 | b3 | b4 | b5 |
|--|--|--|--|--|--|--|--|
| raw action | **0.201** | 0.679 | 0.689 | 0.690 | 0.675 | 0.658 | 0.649 |
| consequence (Δemb) | 0.258 | 0.316 | 0.339 | 0.360 | 0.382 | 0.400 | 0.427 |
| phys state Δ | 0.272 | 0.501 | 0.514 | 0.521 | 0.517 | 0.511 | 0.509 |

- **Input anchor (control):** at the predictor input (frame-2 embedding, before any
  conditioning) action is already decodable at R²=0.20 — PushT is expert data, so the
  policy makes state predict action. That is the leakage floor; decodability above it is
  the conditioning's contribution.
- **Raw action jumps to ~0.68 at block 0 and stays flat-high** (peak b2=0.69, slight
  decline to 0.65; late/early **0.96**). The conditioning injects nearly all of its
  decodable action content by the first block; later blocks do not increase it.
- **Consequence (Δemb) rises monotonically** 0.316 → 0.427 (late/early **1.35**).
- **No crossover.** Action R² exceeds consequence R² at every block; the gap narrows
  smoothly (−0.36 → −0.22) but never closes. No sharp depth where consequence takes over.

## Read (stated plainly)

There is **no crossover**, and **raw-action decodability stays high throughout** — on
the content axis this matches the case the brief says "supports the distributed reading."
But D, like C, cannot by itself separate early-commitment-with-carry from distributed
maintenance: the residual carries the block-0-injected action forward, so action stays
linearly decodable at every depth whether or not later blocks re-use it. What D adds over
C is the *shape*: action content is **front-loaded** (already ~maximal at block 0), and
consequence is **progressively built** (smooth rise, no zone). That shape — combined with
B's front-loaded realized injection (injection ratio falls with depth; raw gate⊙mlp update
×0.83) — points at **early commitment with downstream consequence-building**, not a
mid-stack zone and not active per-block re-injection.

## Preregistered prediction for E (committed)

D matches the third case (action ~maximal at block 0, only consequence rising). **Committed
prediction: FRONT-LOADED per-block ablation** — mean-ablating an *early* block's conditioning
hurts teacher-forced MSE and planning most; ablating *late* blocks' conditioning hurts little.
Correspondingly, **cumulative ablation from an early block destroys performance, and the
commitment depth is early (expected l ≤ 2)**. The MLP-routing hypothesis (B): **per-branch
ablation of the MLP chunks reproduces most of the full-block effect.**

**Null / falsifier:** a FLAT per-block ablation profile (equal effect at every depth) would
mean genuinely distributed conditioning (decision row 4) and would falsify the front-loaded
prediction. A late-peaked profile would mean late commitment (also a falsifier). E adjudicates.

Artifacts: `measurement_d.png`, `measurement_d_summary.json`. Reproduce with
`uv run python -m experiments.measurement_d --from-cache`.
