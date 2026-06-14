# Measurement A — static adaLN gate audit (Phase 1, G1)

No data, no forward pass. For each of the 6 ConditionalBlocks the final adaLN Linear
(zero-initialised at train start) is split into the 6 output chunks
`[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`; we report the
Frobenius norm of each weight chunk (and bias L2) at **12-site resolution** (the
attn-branch gate `gate_msa` and mlp-branch gate `gate_mlp`, per block, directly scale
the two residual updates). Raw norms: `DATA_ROOT/activations/measurement_a/adaln_chunk_norms.npz`.

## Gate magnitude by depth (action-conditional part, ‖W‖_F)

| block | gate_msa (attn) | gate_mlp (mlp) | gate share of block budget |
|------:|----------------:|---------------:|---------------------------:|
| 0 | 1.242 | 2.457 | 0.266 |
| 1 | 1.236 | 2.671 | 0.271 |
| 2 | 1.292 | 3.025 | 0.282 |
| 3 | 1.288 | 3.449 | 0.284 |
| 4 | 1.408 | 3.823 | 0.302 |
| 5 | 1.435 | 4.090 | 0.326 |

Combined gate (msa+mlp) early-mean 3.80 → late-mean 5.38 (**late/early = 1.41**).

## Read (static, necessary-not-sufficient)

- **No wash-out anywhere.** Every gate chunk is well away from the zero-init; the
  smallest gate (b1.attn, 1.236) is ~30% of the largest (b5.mlp, 4.09), never near zero.
  This rules out the *static* signatures of both "early commitment, late gates near
  zero" and "true wash-out."
- **Gates grow into depth, they do not fade.** gate_mlp rises monotonically +66%
  (2.46→4.09); gate_msa rises gently +16% (1.24→1.44). Later blocks also spend a
  *larger share* of their adaLN budget on gating (0.27→0.33). This is the static
  pattern consistent with an **AEZ candidate / late-emphasis distributed** profile,
  and inconsistent with early commitment.
- **Action conditioning enters mainly through the MLP branch.** gate_mlp is ~2× gate_msa
  at every depth — the residual updates that are most action-gated are the MLP ones.
- **Gates are action-conditional, not constant.** Gate bias L2 is small (≤0.28) and
  its signed mean is ≈0, so the gate is driven by the conditioning input `c`, not a
  learned DC offset.

This is a static fossil record: ‖W_gate‖_F shows the gate *can* vary strongly with
the action embedding and does so increasingly with depth. Whether that translates into
rising *causal* use (vs. a large-but-unused parameter) is exactly what Measurements C
(perturbation propagation) and E (mean-ablation) decide. The static evidence points
away from wash-out/early-commitment and toward an emergence-with-depth picture.

Artifacts: `measurement_a.png`, `measurement_a_summary.json`. Reproduce with
`uv run python -m experiments.measurement_a --from-cache`.
