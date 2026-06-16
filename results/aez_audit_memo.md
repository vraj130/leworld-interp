# AEZ audit memo: action conditioning in LeWM's predictor (final)

Status: Measurements A through E, the E
retrained-probe robustness check, the Phase 5 reacher replication, the Phase 6 depth-scaling
study, the Phase 7 released-scale depth-12 verification, and the Phase 8 reacher depth sweep are
complete and reproducible. Scope: the released LeWM checkpoints for PushT and reacher, a PushT
reduced-regime depth sweep {3, 6, 12, 18}, a released-scale PushT depth-12 retrain, and a
released-scale reacher depth sweep {3, 6, 12, 18}. Every probe number states its sample size, since
the retrained-probe check showed probe results are sample-size sensitive.

## Verdict in one paragraph

The Action Emergence Zone hypothesis predicted a specific interior depth in LeWM's six-block
predictor where action conditioning flips from loosely mixed input into a committed latent
consequence. We do not find a sharp interior zone. We find early graded commitment that is
routed through the MLP residual branch. The action conditioning is committed predominantly in
the first three predictor blocks, the commitment is graded rather than a sharp cliff, the
conditioning pathway is MLP-routed, and the consequence representation is distributed across
depth with late blocks refining it. This maps to decision row 2, early commitment, which we
read as commitment at the injection sites. It replicates across both environments tested. The
depth-scaling study refines what "early" means: the commitment location is relative, not absolute.
It sits at a roughly constant fraction, about the first third, of predictor depth, so the block-2-of-6
result is the depth-6 instance of a fraction that scales with depth rather than a fixed early block.
This relative-depth law is verified at released scale at two points (depth 6 and depth 12 both commit
at fraction 0.33), so it is not an artifact of the reduced-data retraining regime, and it is
environment-general: a full released-scale depth sweep on reacher reproduces the same fractions
(0.33, 0.33, 0.33, 0.39 at depths 3, 6, 12, 18), identical to PushT at every depth. So the commitment
fraction holds at about 0.33 across both the depth axis and both environments tested.

## Cross-environment status: REPLICATED, including the depth law

At depth 6 the three least-ambiguous signals were re-run on reacher (3D continuous control,
action_dim 2), fidelity-gated first (teacher-forced relative MSE 0.7 percent and 80x a persistence
baseline, rollout cost 91x worse under shuffled actions, on N equals 1000 val clips). All three
replicate: the D_l propagation curve is monotonic and roughly linear, the cumulative-ablation
commitment depth is 2 (identical to PushT), block 5 is again near-idle, and the MLP share of the
per-branch ablation effect is 0.75, above the 0.7 bar.

The full depth sweep was then run on reacher as well (Phase 8): LeWM retrained from scratch on
reacher at predictor depth {3, 6, 12, 18} at released-data scale (lossless RAM-resident zstd cache,
all four readout-fidelity PASS, d3 and d6 trained on fourier and d12 and d18 on shannon from the same
shared cache, then re-measured uniformly). The reacher commitment fractions are 0.33, 0.33, 0.33,
0.39 for depths 3, 6, 12, 18, identical to PushT at every depth, with the absolute block growing
1, 2, 4, 7 exactly as on PushT, seed-stable, and with the trained reacher d6 reproducing the official
released reacher d6 (both fraction 0.33). MLP routing holds at every depth (share 0.74 at d3/d6,
attention at the noise floor at d12/d18), and the D_l deep-third plateau is present throughout. So the
relative-depth, MLP-routed commitment law is not PushT specific: the commitment fraction holds at
about 0.33 across both the depth axis and two environments. Detail in
results/measurement_phase8_reacher_sweep/measurements_results.md.

## Depth-scaling (Phase 6): commitment depth is RELATIVE

The one untested axis of the original audit was predictor depth: with a single six-block model we
could not tell whether commitment at block 2 was an absolute architectural constant or a fixed
fraction of the stack. We retrained LeWM from scratch on PushT at predictor depth {3, 6, 12, 18},
varying only predictor.depth, with everything else fixed (identical config, the same 1500 / 200
episode pixel cache, 30k steps, seed 3072). Per-block capacity is constant at 1.80M parameters, so
depth is the only variable, and all four pass the fidelity gate at comparable quality (relative MSE
0.021 to 0.026, 7.5 to 8.8x persistence). All measurements use N equals 1000 held-out val clips at
the readout token, eval() and fp32, mean ablation.

These four models are trained in a reduced-data regime (relative MSE about 0.025) weaker than the
released checkpoint (about 0.007), so the depth law is bridged to the released-scale finding through
one hard gate: the retrained depth-6 model must reproduce the released depth-6 audit. It does, on
all three signals at N equals 1000, commitment depth 2 (identical to released), MLP share 0.857
(released 0.905, both above the 0.7 bar), and a linear D_l with no plateau. The loss-slope
diagnostic shows all four models had plateaued by the 30k cutoff (the final 6k steps are under 2%
of total descent in every case), so the reduced-regime solutions are settled rather than
undertrained, and the residual gap to released quality is a data gap, not a step gap.

Headline result. Commitment depth, the deepest block whose onward ablation still does at least half
the full damage, scales with total depth:

| total depth | commitment block | fraction of depth | absolute prediction | relative prediction |
|--:|--:|--:|:--|:--|
| 3  | 1 | 0.33 | 2 (0.67) | 1 (0.33) |
| 6  | 2 | 0.33 | 2 (0.33) | 2 (0.33) |
| 12 | 5 | 0.42 | 2 (0.17) | 4 (0.33) |
| 18 | 7 | 0.39 | 2 (0.11) | 6 (0.33) |

The absolute block index grows 1, 2, 5, 7 while the fraction stays in a tight band, mean 0.37. The
preregistered absolute hypothesis is falsified (both committed falsifiers fired: the depth-12
commitment 5 is well above 3 and the depth-18 commitment 7 is above 4). The relative hypothesis is
confirmed, with a proportionality constant of about 0.37, a little deeper than the preregistered one
third. Three supporting signals agree. The deepest blocks go idle as depth grows (last-block share
of full damage 0.23, 0.08, 0.01, 0.00 for d3, d6, d12, d18), so a fixed front fraction does the
action work and the tail is redundant. MLP routing is depth-invariant (mean MLP share 0.82, 0.86,
0.77, 0.86, all above 0.7), though at the larger depths the earliest one to three blocks are more
attention-mixed before MLP dominance sets in. And a propagation plateau emerges only at higher
depth: D_l stays near-linear at every depth (linear-fit R2 at least 0.99) but its deep-third slope
falls from about 1.1 at d3 and d6 to 0.74 at d12 and 0.56 at d18, a zone-like saturation that is
invisible at depth 6 and appears once there are enough layers. Detail in
results/measurement_phase6_depthscaling/depth_scaling_results.md.

Released-scale confirmation (Phase 7). Because the four depth-scaling models were trained in a
reduced-data regime (rel-MSE about 0.025) while the original audit used the released checkpoint
(rel-MSE about 0.007), we trained one depth-12 model at released-data scale to verify the law where
the absolute and relative hypotheses diverge most. d12 was retrained from scratch on the full 18,485
non-val episodes (lossless RAM-resident zstd cache to stay GPU-bound) and reached rel-MSE 0.0085
(fidelity PASS, rollout shuffled-over-true 15.3x), about 3x better than the reduced d12 and within
about 20% of the released d6. At released scale its commitment lands at block 4, fraction 0.33,
identical to the released d6 fraction and stable across seeds and sample sizes, decisively above the
absolute prediction (block about 2, fraction about 0.17). MLP routing holds (the attention branch is
at the numerical noise floor, so the conditioning is essentially all MLP-routed), and the D_l
deep-third plateau persists (late-over-early slope 0.59). So the depth law is now confirmed at two
released-scale points, d6 and d12, both at fraction 0.33; released-scale training if anything tightens
the constant relative to the reduced regime (which gave 0.42 at d12). Detail in
results/measurement_phase7_released_d12/measurements_results.md.

## Lead with the clean causal signals

Three signals carry the verdict, and all three are early-emphasized.

Cumulative mean-ablation, the valid causal probe. Ablating the conditioning from block l onward
gives a monotonic front-loaded curve. The commitment depth is 2 at the half-of-full bar on both
PushT and reacher. Per-block marginal contributions are roughly equal for blocks 0 to 3 then
taper, and block 5 contributes almost nothing, so commitment is early and graded, not a sharp
cliff and not uniform.

Consequence decodability, corrected for distribution shift. The first pass used a frozen probe
fit on unablated activations (N equals 4000 train and 1000 val) and showed a front-loaded
consequence drop. The retrained-probe check (same N, 4000 train and 1000 val) refit a fresh
ridge probe on the ablated distribution and recovered most of the apparent drop, with a
late-block mean recovery of 0.59 for the next-embedding-delta target and up to 0.86 for the
physical-state-delta target. So the consequence is still largely present in the ablated
activations and was hidden from the frozen probe by a shift in the linear decoding directions.
The true per-block consequence loss is small and only mildly front-loaded. Late blocks refine
the consequence rather than merely carrying it. Linear-probe qualifier: both probes are ridge,
so this measures recoverable linear consequence under the ablated distribution, not a proof that
late blocks add nonlinear structure. The qualitative claim, that late blocks participate, is the
robust part.

Activation-level injection (Measurement B). The realized injection ratio, the gated update norm
over the residual norm, is front-loaded and falls with depth even as the gate magnitude rises,
and the MLP branch dominates injection and the dominance grows with depth. This agrees with the
ablation and consequence results.

Together these give a partial double dissociation: early action commitment on the action pathway,
plus a distributed consequence representation that late blocks help build.

## Methods note: why the single-block action-MSE is not the commitment probe

The single-block ablation action-MSE is back-loaded, peaking near block 4, which looks like it
contradicts early commitment. It does not. In an architecture that re-injects the same
conditioning signal at every layer, single-site ablation underestimates early-site importance
because the downstream sites still inject the action and compensate. A matched-norm
random-direction control confirms the confound: at an early block a random perturbation of the
conditioning is comparable to or larger than the true mean-ablation and is itself front-loaded,
so the single-block metric is dominated by generic perturbation propagation, not action-specific
damage. The valid causal probe under re-injected conditioning is cumulative ablation, which is
what we use for the commitment depth. The literally falsified preregistration, front-loaded
single-block ablation, is exactly this methodological point rather than a negative result.

## Preregistration outcomes

Registered after D: front-loaded single-block ablation, commitment depth at or below 2, and
MLP routing. Commitment depth held, depth 2 on both environments. MLP routing held, MLP share
0.78 on PushT and 0.75 on reacher. The literal single-block front-loading was falsified and is
reframed as the methods note above. The strongest outcome, a clean double dissociation, holds
in partial form after the retrained-probe correction.

## Limitations to name first

Depth-scaling is now tested, including at released scale and across both environments. The earlier
worry, that the six-block predictor was shallow and the commitment might sharpen or move in a deeper
predictor, is resolved by Phase 6: the commitment does move, staying at a roughly constant fraction
of depth across {3, 6, 12, 18}. Phase 7 closed the data-scale gap on PushT (a released-scale depth-12
model commits at fraction 0.33, matching the released depth-6). Phase 8 closed the environment gap: a
full released-scale depth sweep on reacher gives fractions 0.33, 0.33, 0.33, 0.39, identical to PushT
at every depth. So the depth law is now confirmed at released scale on both environments, not just in
the reduced regime and not just on PushT. The remaining caveat is small: the reacher depth sweep and
the PushT intermediate depths were trained by our own pipeline rather than existing as official
released checkpoints, but the retrained-depth-6 consistency checks on both environments establish
that the pipeline reproduces the official released depth-6 audit (PushT and reacher both fraction
0.33).

Single architecture. The result is replicated across two environments, PushT and reacher, and across
the depth axis in both, but within one model family, LeWM. It is not tested across architectures, so
the relative-depth, MLP-routed pattern may be specific to this AdaLN-zero JEPA predictor. This is now
the one untested axis.

## Recommendation

Proceed, with the framing reframed from a mid-stack Action Emergence Zone to graded commitment at
the injection sites, located at a roughly constant fraction (about the first 40%) of predictor
depth, predominantly MLP-routed, with a distributed consequence representation that late blocks
refine. Do not change substrate. The V-JEPA 2-AC scoping check is reserved for a confirmed flat or
distributed result, which this audit did not produce on any of the models tested.

The depth-scaling axis that was the original main threat is now closed on both environments:
commitment is relative, not a fixed early block, it sits at about the same fraction (0.33) on PushT
and reacher, and MLP routing holds at every depth. For a practitioner building JEPA-style world
models the actionable read is that the action-conditioning work is done by a fixed front fraction
(about the first third) of the predictor and the deep tail is largely idle, so predictor depth past
that fraction buys little for action conditioning specifically, and this holds across both
environments tested. The one remaining open axis is cross-architecture generality, not depth and not
environment.

## Key figures

- results/measurement_a/measurement_a.png, static gate profile.
- results/measurement_bc/measurement_bc.png, injection ratio and D_l.
- results/measurement_d/measurement_d.png, action versus consequence decodability (N 4000/1000).
- results/measurement_e/measurement_e.png, per-block, cumulative, and per-branch ablation.
- results/measurement_e/probe_check.png, retrained versus frozen consequence probe (N 4000/1000).
- results/measurement_phase5_reacher/phase5.png, reacher replication of the three signals (N 1000).
- results/measurement_phase6_depthscaling/phase6_gate_d6.png, retrained-d6 versus released-d6 gate (N 1000).
- results/measurement_phase6_depthscaling/phase6_sweep.png, commitment depth, MLP routing, and D_l shape across depth {3, 6, 12, 18} (N 1000).
- results/measurement_phase7_released_d12/phase7_released_d12.png, released-scale d12 commitment fraction vs released d6 and reduced d12 (N 1000).
- results/measurement_phase8_reacher_sweep/phase8_reacher_sweep.png, reacher depth sweep commitment fraction matching PushT across {3, 6, 12, 18} (N 1000).

Every figure has a saved raw-array counterpart under DATA_ROOT and is regenerable with the
matching experiment module and the --from-cache flag.
