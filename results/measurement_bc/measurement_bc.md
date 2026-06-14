# Measurements B & C — activation-level action conditioning (Phase 2, G2)

512 held-out PushT clips (504 episodes), `eval()` + fp32, seed 0. All quantities at
the **readout token** (last of the 3-frame context, index 2) and **12-site resolution**.
Paired passes share states; only the action changes — full-history swap (`a'_full`,
within-batch permutation) and final-action-only swap (`a'_final`, same permutation but
only the last action token). Raw per-sample arrays: `DATA_ROOT/activations/measurement_bc/bc_arrays.npz`.

## Perturbation magnitudes (so D_l is per-unit-perturbation interpretable)
`‖a−a'‖` full **7.47**, final **3.96** (z-action). `‖c−c'‖` at the readout token is
**identical (24.99)** for both variants — by construction the readout's own action token
gets the same swap; the variants differ only in whether tokens 0–1 are also perturbed.

## (C) Action-perturbation propagation D_l — the headline curve

| after block | 0 | 1 | 2 | 3 | 4 | 5 |
|--|--|--|--|--|--|--|
| D_l full-swap | 0.115 | 0.183 | 0.251 | 0.314 | 0.375 | 0.434 |
| D_l final-swap | 0.109 | 0.175 | 0.244 | 0.305 | 0.365 | 0.425 |

- **Monotonic, ≈linear growth (×3.9 over depth); no decay, no plateau.** Per-block
  increments are nearly constant (~0.066, 0.069, 0.061, 0.060, 0.060).
- **The two variants agree almost exactly** (full barely above final). Since the
  readout-token perturbation is identical, this means the readout's action divergence
  comes overwhelmingly from its **own** action token, not the swapped history — i.e.
  the cleanest causal signal (final-action swap, trusted on disagreement) gives the
  same curve. There is no variant disagreement about where D_l plateaus (neither plateaus).

## (B) Realized injection ratio ‖gate⊙branch‖ / ‖x‖ (12 sites)

| block | 0 | 1 | 2 | 3 | 4 | 5 |
|--|--|--|--|--|--|--|
| attn site (gate_msa) | 0.080 | 0.061 | 0.050 | 0.044 | 0.041 | 0.039 |
| mlp site (gate_mlp) | 0.111 | 0.089 | 0.085 | 0.080 | 0.077 | 0.084 |
| mlp/attn ratio | 1.39 | 1.46 | 1.69 | 1.82 | 1.90 | 2.14 |

- **Per-block injection is FRONT-LOADED** (largest at block 0), the opposite of the
  parameter-norm trend in Measurement A.
- **MLP branch dominates and increasingly so** (mlp/attn injection ratio 1.39 → 2.14).
  The Measurement-A parameter finding (gate_mlp ≈ 2× gate_msa) holds at the activation
  level and grows with depth. **Hypothesis to confirm at E** (does ablating mlp-branch
  conditioning hurt more than attn-branch?).

## Requirement #1 — capacity vs realized use (the contradiction, disentangled)

Late/early ratios (MLP branch): gate ‖W‖ param capacity (A) **×1.66** | mean|gate|
realized **×2.16** | raw update ‖gate⊙mlp‖ **×0.83** | residual ‖x‖ **×1.09** |
injection ratio **×0.75** | **D_l ×3.90**.

The contradiction the brief warned about **is present**: the gate parameter norm (and
the realized gate magnitude, ×2.16) rise with depth, yet the **per-block injection
ratio falls (×0.75)**. But it is **not** a parameter-space artifact / wash-out: the
flat-to-falling ratio is explained by the **branch output shrinking** (raw update ×0.83)
against a **mildly growing residual** (×1.09) — not an unused gate — and the **causal
effect D_l grows ×3.9**. So late blocks open their gates wider onto smaller branch
outputs; the action signal injected early is **amplified/preserved, not re-injected
with growing strength and not overwritten.**

## Read against the four-row decision table

- NOT **true wash-out** — D_l grows ×3.9, never decays; gates are active throughout.
- NOT **early commitment with late gates near zero** — late gates are not near zero
  (mean|gate_mlp| rises to 0.67; injection ratio still ~0.08 at block 5).
- NOT a clean **AEZ candidate (grow-then-plateau)** — D_l grows but does **not** plateau;
  it is essentially linear (no mid-predictor zone where it flattens).
- **Closest row: "roughly uniform across blocks → distributed conditioning, no zone."**
  Per-block D_l increments are near-constant; commitment accrues gradually rather than at
  a single depth. Refinement: injection is front-loaded and then amplified, so the story
  is "mixed early, progressively committed," not a localized emergence zone.

**This is the row that pressures the strong localized-AEZ framing.** It does not kill the
weaker "gradual commitment" reading, and D_l is a propagation measure that can grow by
passive carry-forward as easily as by active per-block use. The decisive test is the
**causal ablation (Measurement E)** — per-block and cumulative mean-ablation locate (or
rule out) a commitment depth. G2 verdict is therefore *provisional: leaning distributed/
gradual, AEZ-as-a-sharp-zone not supported at the activation level; E adjudicates.*

Artifacts: `measurement_bc.png`, `measurement_bc_summary.json`. Reproduce with
`uv run python -m experiments.measurement_bc --from-cache`.
