# Phase 6 preregistration (committed before measuring commitment depth)

Written after training and fidelity, before running any commitment-depth or MLP-routing
measurement on the retrained models. The audit found commitment depth k about 2 on the
released depth-6 predictor (50%-of-full cumulative-ablation bar). Question: is that
location absolute (architecture-driven, stays near block 2 regardless of total depth) or
relative (scales with depth, stays near one third of the stack)?

## Hypotheses and their quantitative predictions

Predicted commitment depth (50% bar) per total depth, absolute index and fraction:

| total depth | absolute hypothesis | relative hypothesis |
|--:|:--|:--|
| 3  | ~2  (frac 0.67) | ~1  (frac 0.33) |
| 6  | ~2  (frac 0.33) | ~2  (frac 0.33) |
| 12 | ~2  (frac 0.17) | ~4  (frac 0.33) |
| 18 | ~2  (frac 0.11) | ~6  (frac 0.33) |

The two hypotheses agree at depth 6 (both ~2) and diverge most at depth 12 and 18, which are
the discriminating points.

## Falsifiers (committed)

- Absolute is FALSIFIED if commitment depth grows with total depth, in particular if the
  depth-12 commitment is clearly above ~3 or the depth-18 commitment above ~4.
- Relative is FALSIFIED if commitment depth stays near ~2 regardless of total depth, in
  particular if depth-12 and depth-18 commitments are both ~2 to 3.
- Null / messy outcome: commitment depth follows neither cleanly, or MLP-routing breaks at
  higher depth (MLP share drops below ~0.7), or a depth fails to train to comparable
  fidelity. Any of these is reported plainly as a limitation-section upgrade rather than
  forced into one of the two clean laws.

## Internal consistency check

The retrained depth-6 model (reduced episode set) is expected to reproduce commitment depth
about 2, matching the released depth-6 audit. If it does not, the reduced-data retraining is
not comparable to the released checkpoint and the depth comparison is read with that caveat.

I will state after measuring which hypothesis the data matched and which falsifiers fired.

## Outcome (recorded after measuring, N = 1000 per depth)

Observed commitment depth (50% bar): d3 = 1 (frac 0.33), d6 = 2 (0.33), d12 = 5 (0.42),
d18 = 7 (0.39). Mean fraction 0.37.

RELATIVE matched. The commitment fraction is near-constant while the absolute block grows
1, 2, 5, 7. ABSOLUTE falsified: both committed absolute falsifiers fired (depth-12 commitment 5
is clearly above 3, depth-18 commitment 7 is above 4). The relative proportionality constant came
out about 0.37, slightly deeper than the registered one third (observed 1, 2, 5, 7 versus
registered 1, 2, 4, 6), so the law is relative with a constant a little above one third.

No null / messy trigger fired: MLP share stayed above 0.7 at every depth (0.82, 0.86, 0.77, 0.86)
and all four depths passed the fidelity gate.

Internal-consistency check passed: retrained-d6 reproduced commitment depth 2, MLP share 0.857,
and a linear D_l, matching the released depth-6 audit (see gate_d6.md). Full write-up in
depth_scaling_results.md.
