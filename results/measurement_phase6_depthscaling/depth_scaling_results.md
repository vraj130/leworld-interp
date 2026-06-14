# Phase 6 depth-scaling results: commitment depth is RELATIVE, not absolute

Question: the released depth-6 audit located action commitment at predictor block 2. Is that
location absolute (architecture-driven, fixed near block 2 regardless of total depth) or
relative (a fixed fraction of the stack that scales with depth)? Preregistered in
preregistration.md before any measurement.

Setup: LeWM retrained from scratch on PushT at predictor depth {3, 6, 12, 18}, only
predictor.depth varied, identical config / pixel cache (1500 train, 200 val episodes,
episode-disjoint) / 30k steps / seed 3072. Per-block capacity constant at 1.80M parameters,
so depth is the only thing that changes. All four pass the fidelity gate at comparable quality
(rel-MSE 0.021 to 0.026, 7.5 to 8.8x persistence). Measurements at the readout token, eval()
and fp32, mean ablation, N = 1000 held-out val clips per depth. Ran on CPU because shannon GPU 1
fell off the PCI bus and poisoned CUDA init node-wide; the measurements are inference-only and
fp32-equivalent on CPU.

## Transfer justification: the retrained-d6 gate PASSED

The four models live in a reduced-data regime (rel-MSE about 0.025) weaker than the released
checkpoint (about 0.007), so the depth law is bridged to the released-scale finding through one
hard check: the retrained depth-6 model reproduces the released depth-6 audit. It does, on all
three signals (N = 1000): commitment depth 2 (identical to released), MLP share mlp/(mlp+attn)
0.857 (released 0.905, both well above the 0.7 bar), and a linear D_l with no plateau (linear-fit
R2 = 1.00). See gate_d6.md. Loss-slope diagnostic: all four models had plateaued by 30k (the
final 6k steps account for under 2% of total descent in every case), so the reduced-regime
solutions are settled, not undertrained, and the residual gap to released quality is a data gap
rather than a step gap.

## Headline: commitment depth scales with total depth (RELATIVE)

Commitment depth is the deepest block l where ablating the conditioning from block l onward still
does at least 50% of the full-ablation damage.

| total depth | commitment block (absolute) | fraction of depth | absolute prediction | relative prediction |
|--:|--:|--:|:--|:--|
| 3  | 1 | 0.33 | 2 (0.67) | 1 (0.33) |
| 6  | 2 | 0.33 | 2 (0.33) | 2 (0.33) |
| 12 | 5 | 0.42 | 2 (0.17) | 4 (0.33) |
| 18 | 7 | 0.39 | 2 (0.11) | 6 (0.33) |

The absolute block index grows 1, 2, 5, 7 while the fraction of the stack stays in a tight band,
mean 0.37. This is the relative hypothesis. The released audit's "commitment at block 2 of 6" is
the depth-6 instance of a fixed fraction, roughly the first 40% of predictor depth, not a fixed
absolute block.

Preregistered outcome:

- ABSOLUTE is FALSIFIED. Both committed absolute falsifiers fired: the depth-12 commitment (5)
  is clearly above 3, and the depth-18 commitment (7) is above 4.
- RELATIVE is CONFIRMED. Commitment tracks a near-constant fraction of depth. The observed
  fraction is about 0.37, a little deeper than the preregistered one third (observed absolute
  blocks 1, 2, 5, 7 versus the registered 1, 2, 4, 6), so the law is relative with a
  proportionality constant slightly above one third rather than exactly one third.
- No null / messy trigger fired: MLP routing held above 0.7 at every depth, and every depth
  passed the fidelity gate.

## Supporting signal 1: the deep tail goes idle as depth grows

Cumulative damage from ablating only the final block, as a fraction of full:

| depth | last block / full |
|--:|--:|
| 3  | 0.23 |
| 6  | 0.08 |
| 12 | 0.01 |
| 18 | 0.00 |

The deepest blocks contribute essentially nothing to the action pathway at higher depth. A fixed
front fraction of the stack does the action work and the remaining tail is increasingly redundant.
This is the same picture as the relative-commitment result seen from the other end.

## Supporting signal 2: MLP routing is depth-invariant

Mean MLP share of the per-branch ablation effect, mlp/(mlp+attn), per depth: 0.82 (d3), 0.86 (d6),
0.77 (d12), 0.86 (d18). All above the 0.7 bar, so MLP-routed conditioning is a stable property of
this AdaLN-zero predictor, not a depth-6 artifact. One nuance worth stating: at the larger depths
the earliest one to three blocks are more attention-mixed (for example d12 block 2 share 0.21,
d18 blocks 0 and 1 around 0.5 to 0.6) before MLP dominance sets in from roughly block 3 onward, so
the dominance is a property of the bulk and the commitment band, not literally every block.

## Supporting signal 3: a propagation plateau emerges only at higher depth

The action-perturbation propagation curve D_l stays close to linear at every depth (linear-fit R2
0.999, 0.999, 0.995, 0.988 for d3, d6, d12, d18). What changes with depth is the deep-third slope.
The ratio of late-third to early-third slope is 1.10 and 1.09 at d3 and d6 (slightly accelerating)
but falls to 0.74 at d12 and 0.56 at d18 (the deep third grows progressively less). So a saturation
or plateau in propagation becomes visible only once there are enough layers, consistent with the
commitment finishing around the 0.4 mark and the deep tail refining or saturating rather than
continuing to inject. A zone-like saturation that is invisible at depth 6 and appears at depth 12
to 18 is itself a finding.

## What this changes

The single named limitation of the original audit, that the predictor is shallow and depth-scaling
was untested so the early commitment might sharpen or move in a deeper predictor, is now resolved.
The commitment does move with depth, in a specific and orderly way: it stays at a roughly constant
fraction (about 0.37) of predictor depth. The framing changes from "early commitment in the first
three blocks" to "commitment at a fixed fraction, roughly the first 40%, of predictor depth," with
the routing (MLP-dominant) and the graded-not-cliff character preserved across all depths tested.

Honest caveat carried forward: this depth law is established in the reduced-data regime and bridged
to released-scale through the retrained-d6 consistency gate. It is not directly verified at
released-scale at every depth, because only the depth-6 model exists at released quality.

## Artifacts

- gate_d6.md, phase6_gate_d6.png: the retrained-d6 internal-consistency gate.
- phase6_sweep.png: the cross-depth comparison (commitment depth, MLP routing, D_l shape).
- measurements_summary.json: all per-depth summaries plus the loss-slope table.
- fidelity_table.md, fidelity_table.json: the training and fidelity gate.
- preregistration.md: the committed absolute-versus-relative prediction.
- Raw per-depth arrays under DATA_ROOT/activations/measurement_phase6_depthscaling/depth_{d}.json.
- Reproduce: uv run python -m experiments.measurement_phase6 --depths 3 6 12 18 --device cpu;
  uv run python -m experiments.plot_phase6 --sweep --depths 3 6 12 18.
