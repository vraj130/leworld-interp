# Phase 8 preregistration (committed before any reacher commitment-depth measurement)

Written after deciding to train the reacher depth sweep, before running any commitment-depth or
MLP-routing measurement on the reacher models. The fidelity gate (teacher-forced rel-MSE and
rollout shuffled-vs-true) does not reveal commitment numbers, so committing here keeps the
commitment analysis preregistered.

## Background (what PushT established)

On PushT the action commitment sits at a roughly constant fraction of predictor depth (relative,
not a fixed absolute block). Reduced-regime sweep {3,6,12,18}: commitment blocks {1,2,5,7},
fractions {0.33, 0.33, 0.42, 0.39}, mean about 0.37. Released-scale verification: d6 fraction 0.33
and d12 fraction 0.33. MLP routing dominates at every depth, and the D_l propagation curve develops
a deep-third plateau at higher depth (late-over-early slope falling toward about 0.6 at d12/d18).
Phase 5 already showed the released reacher d6 reproduces the PushT d6 picture: commitment block 2
(fraction 0.33), MLP share 0.75, monotonic D_l.

## Question

Does the depth law (commitment at a roughly constant fraction of predictor depth, MLP-routed,
plateau-at-depth) hold on reacher across {3,6,12,18}, or is it PushT-specific?

## Hypotheses and quantitative predictions

- GENERALIZES (environment-general law). Reacher commitment fractions cluster near the PushT
  constant, about 0.33 (within roughly 0.28 to 0.40), scaling with depth so the absolute block
  grows about as 1, 2, 4, 6 for depths 3, 6, 12, 18. MLP routing holds (share above 0.7 or
  attention at the noise floor). The D_l deep-third plateau emerges at higher depth. This is the
  target outcome: the commitment fraction holds at about 0.33 across both the depth axis and two
  environments.

- PARTIAL (relative law, environment-dependent constant). Reacher commitment is still relative
  (the fraction is roughly constant across depth, the absolute block grows with depth) but the
  constant differs materially from PushT's 0.33, for example clustering near 0.25 or near 0.45.
  Still a relative-depth law, but the constant is environment-dependent. A more nuanced finding.

- DIVERGES. Reacher shows no clean relative law: the fraction does not track depth (for example
  commitment pinned to a fixed absolute block regardless of total depth, the Phase 6 absolute
  hypothesis), or MLP routing breaks (share drops below 0.7 with attention not at the floor), or a
  depth fails to reach released-approaching fidelity and is excluded. Reported plainly;
  cross-environment instability is itself a limitation and a finding.

## Falsifiers (committed)

- GENERALIZES is falsified if the reacher fractions fall outside about 0.28 to 0.40 at the
  discriminating depths (d12, d18), or if commitment does not grow with depth (stays near a fixed
  absolute block), or if MLP routing breaks with attention clearly above the noise floor.
- PARTIAL is the outcome if the fractions cluster tightly (relative law holds) but the cluster
  center is clearly off 0.33 (below about 0.28 or above about 0.40).
- The relative law itself is falsified (DIVERGES) if commitment stays near a fixed absolute block
  across depths (for example near 2 at every depth, fraction shrinking from 0.67 to 0.11), which
  is the absolute hypothesis Phase 6 already rejected on PushT.

## Internal consistency check

The retrained reacher d6 (our pipeline, released-data scale) is expected to reproduce the Phase 5
released reacher d6 audit: commitment block about 2 (fraction 0.33), MLP routing dominant. If it
does not, our reacher training is not comparable to the released reacher checkpoint and the sweep
is read with that caveat. This is the reacher analog of the Phase 6 retrained-d6 PushT gate.

## Acceptance

Only fidelity-passing depths (released-approaching quality, comparable to the Phase 5 reacher
reference rel-MSE about 0.0066, well off the reduced-regime plateau) are compared into the fraction
analysis. I will state after measuring which hypothesis matched and which falsifiers fired.
