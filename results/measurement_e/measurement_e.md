# Measurement E — causal mean-ablation (Phase 4)

1000 held-out PushT clips (968 episodes), `eval()` + fp32, seed 0, 8-frame windows.
Mean ablation only (conditioning into a block's `adaLN_modulation` replaced by the
batch-mean action embedding; never zeroed, never at `cond_proj`). Encoder run once;
every ablation is a predictor re-run on cached embeddings. Baseline TF readout MSE
0.00749, rollout cost 14.72. Raw arrays: `DATA_ROOT/activations/measurement_e/e_arrays.npz`.

## Part 1 — per-block (single block ablated), l = 0..5

| metric | b0 | b1 | b2 | b3 | b4 | b5 | shape |
|--|--|--|--|--|--|--|--|
| action TF-MSE increase | 0.0028 | 0.0025 | 0.0037 | 0.0073 | 0.0102 | 0.0066 | **back-loaded (peak b4)** |
| planning cost increase | 13.9 | 13.6 | 18.0 | 26.6 | 32.1 | 25.8 | back-loaded (peak b4) |
| consequence-R² drop (Δemb) | 0.106 | 0.061 | 0.043 | 0.040 | 0.033 | 0.026 | **front-loaded (peak b0)** |
| random-dir control MSE | 0.0106 | 0.0026 | 0.0019 | 0.0013 | 0.0021 | 0.0029 | **front-loaded (peak b0)** |

Read: the single-block action-MSE is **confound-dominated** and not a clean commitment
probe. The matched-norm random control is comparable to or larger than the true ablation
(0.0106 vs 0.0028 at b0) and front-loaded, i.e. any matched-norm conditioning perturbation
at an early block propagates and corrupts the readout (generic sensitivity / AR compounding).
On top of that, the same `c` is re-injected at every block, so removing one block's fresh
injection is compensated downstream. Hence the back-loaded single-block action-MSE reflects
**where the readout integrates conditioning (late)**, not where action commits. The
**consequence-drop is action-specific and front-loaded**: ablating an early block's
conditioning destroys the consequence representation at that block most, while late blocks'
conditioning barely changes their own consequence R² (it is carried from upstream).

## Part 2 — cumulative (ablate all blocks ≥ l) → commitment depth

| ablate ≥ l | 0 | 1 | 2 | 3 | 4 | 5 |
|--|--|--|--|--|--|--|
| TF-MSE increase | 0.141 | 0.115 | 0.086 | 0.055 | 0.028 | 0.007 |
| fraction of full | 1.00 | 0.82 | 0.61 | 0.39 | 0.20 | 0.05 |

Monotonic, **front-loaded**; mean ablation (action removal), so this is the clean
commitment lever (no random-control confound). **Commitment depth = 2** at the 50%-of-full
bar (threshold-sensitive: 2 at 50%, 3 at ~1/3). The per-block marginal contributions
(decrements ≈ 0.026, 0.029, 0.031, 0.027, 0.021, 0.007) are **roughly equal for blocks 0-3
and taper through 4-5**, so commitment is **graded/early, not a sharp cliff and not uniform**:
blocks 0-3 each carry a similar slab of the action contribution, block 5 carries almost none.

## Part 3 — per-branch (MLP chunks vs attn chunks)

| metric | b0 | b1 | b2 | b3 | b4 | b5 |
|--|--|--|--|--|--|--|
| MLP-only MSE incr | 0.0007 | 0.0010 | 0.0013 | 0.0021 | 0.0028 | 0.0029 |
| attn-only MSE incr | 0.0002 | 0.0001 | 0.0002 | 0.0001 | 0.0002 | 0.0004 |
| full (both) | 0.0015 | 0.0020 | 0.0026 | 0.0033 | 0.0045 | 0.0056 |

**MLP-routing confirmed causally:** MLP-chunk ablation dominates attn-chunk ablation at
every depth (MLP share of the branch sum > 0.7; mlp-only ≈ 0.54 of full, the remainder
being the nonlinear MLP×attn interaction). B's activation-level MLP dominance is causal.

## Preregistered outcomes (committed after D)

- **Pred 1, FRONT-LOADED per-block single ablation: FALSIFIED (literally).** Single-block
  action-MSE is back-loaded (peak b4). The falsifier fired, but the metric is
  confound-dominated (random control comparable, re-injection redundancy); the
  commitment-clean metrics (cumulative, consequence-drop) are early-emphasized, so the
  underlying early-commitment claim survives via the cleaner tools.
- **Pred 2, commitment depth l ≤ 2: HELD** (depth 2 at 50% bar; graded, 2-3).
- **Pred 3, MLP routing dominant: HELD.**
- **Double dissociation: REVISED by the retrained-probe robustness check (below).** The
  frozen-probe consequence-drop looked front-loaded (single dissociation), but ~60% of it was
  distribution shift, not consequence loss. Corrected, the consequence axis is graded/distributed
  (late blocks retain and refine consequence), which leans the result toward a partial *double*
  dissociation: early action commitment plus distributed consequence refinement.

## Robustness: retrained-probe check (does the consequence-drop survive?)

The consequence-drop applied the FROZEN Measurement-D probe (fit on unablated activations) to
ablated activations, so a late-block drop could be the linear consequence subspace shifting
under ablation rather than real consequence loss. Re-fitting a fresh ridge probe on the
ablated distribution (4000 train / 1000 val, matched to D) at each block:

| Δemb R² | b0 | b1 | b2 | b3 | b4 | b5 |
|--|--|--|--|--|--|--|
| baseline (unablated) | 0.316 | 0.339 | 0.360 | 0.382 | 0.400 | 0.427 |
| frozen on ablated | 0.198 | 0.273 | 0.311 | 0.341 | 0.365 | 0.398 |
| retrained on ablated | 0.261 | 0.314 | 0.342 | 0.364 | 0.388 | 0.414 |
| recovery fraction | 0.53 | 0.62 | 0.64 | 0.57 | 0.65 | 0.54 |

(phys-state Δ recovery is even higher, 0.43 → 0.86, peaking at the late blocks.) Late-block
(b3..b5) mean recovery = **0.59**, so a retrained probe recovers most of the frozen drop:
**the consequence is still largely present in the ablated activations; the frozen probe just
could not read it (distribution shift).** The true per-block consequence loss (baseline minus
retrained) is small and only mildly front-loaded (Δemb: 0.055 at b0, ~0.013 late). **The
carry-not-construct claim is softened: late blocks do not merely carry; they retain and
refine the consequence representation.** Note: a retrained *ridge* probe is still linear, so
this is recoverable *linear* consequence under the ablated distribution; the qualitative call
(late blocks participate) is robust.

## Verdict (decision table)

Row 2, **early commitment / commitment-at-injection** (soft/graded, MLP-routed). NOT a
localized mid-stack AEZ (no interior causal peak). NOT flat/distributed-uniform (clear
early emphasis in cumulative, consequence-drop, and B's injection ratio; block 5 ≈ idle).
NOT wash-out (action is causally used; ablation hurts). The substrate-rethink trigger is
**not** met (no confirmed flat result), so no V-JEPA 2-AC scoping check.

Artifacts: `measurement_e.png`, `measurement_e_summary.json`. Reproduce with
`uv run python -m experiments.measurement_e --from-cache`.
