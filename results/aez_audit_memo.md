# AEZ audit memo: action conditioning in LeWM's predictor (final)

Audience: PhD collaborator and Prof. Chen Feng. Status: Measurements A through E, the E
retrained-probe robustness check, the Phase 5 reacher replication, and the Phase 6 depth-scaling
study are complete and reproducible. Scope: the released LeWM checkpoints for PushT and reacher,
plus four from-scratch retrains at predictor depth {3, 6, 12, 18} for the depth-scaling study.
Every probe number states its sample size, since the retrained-probe check showed probe results
are sample-size sensitive.

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
It sits at a roughly constant fraction, about the first 40%, of predictor depth, so the block-2-of-6
result is the depth-6 instance of a fraction that scales with depth rather than a fixed early block.

## Cross-environment status: REPLICATED

The three least-ambiguous signals were re-run on reacher (3D continuous control, action_dim 2),
fidelity-gated first (teacher-forced relative MSE 0.7 percent and 80x a persistence baseline,
rollout cost 91x worse under shuffled actions, on N equals 1000 val clips). All three replicate:
the D_l propagation curve is monotonic and roughly linear, the cumulative-ablation commitment
depth is 2 (identical to PushT), block 5 is again near-idle, and the MLP share of the per-branch
ablation effect is 0.75, above the 0.7 bar. The early-graded-MLP-routed-commitment picture is
not PushT specific.

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

Depth-scaling is now tested, with a caveat. The earlier worry, that the six-block predictor was
shallow and the commitment might sharpen or move in a deeper predictor, is resolved by Phase 6:
the commitment does move, staying at a roughly constant fraction (about 0.37) of depth across
{3, 6, 12, 18}. The remaining caveat is that this law is established in the reduced-data regime and
bridged to released-scale through the retrained-depth-6 consistency gate. It is not directly
verified at released quality at every depth, since only the depth-6 model exists at released scale.

Single architecture. The result is replicated across two environments, PushT and reacher, but
within one model family, LeWM. It is not tested across architectures, so the relative-depth,
MLP-routed pattern may be specific to this AdaLN-zero JEPA predictor.

## Recommendation

Proceed, with the framing reframed from a mid-stack Action Emergence Zone to graded commitment at
the injection sites, located at a roughly constant fraction (about the first 40%) of predictor
depth, predominantly MLP-routed, with a distributed consequence representation that late blocks
refine. Do not change substrate. The V-JEPA 2-AC scoping check is reserved for a confirmed flat or
distributed result, which this audit did not produce on any of the models tested.

The depth-scaling axis that was the original main threat is now closed: commitment is relative, not
a fixed early block, and MLP routing holds at every depth. For a practitioner building JEPA-style
world models the actionable read is that the action-conditioning work is done by a fixed front
fraction of the predictor and the deep tail is largely idle, so predictor depth past that fraction
buys little for action conditioning specifically. The remaining open axis is cross-architecture
generality, not depth.

## Key figures

- results/measurement_a/measurement_a.png, static gate profile.
- results/measurement_bc/measurement_bc.png, injection ratio and D_l.
- results/measurement_d/measurement_d.png, action versus consequence decodability (N 4000/1000).
- results/measurement_e/measurement_e.png, per-block, cumulative, and per-branch ablation.
- results/measurement_e/probe_check.png, retrained versus frozen consequence probe (N 4000/1000).
- results/measurement_phase5_reacher/phase5.png, reacher replication of the three signals (N 1000).
- results/measurement_phase6_depthscaling/phase6_gate_d6.png, retrained-d6 versus released-d6 gate (N 1000).
- results/measurement_phase6_depthscaling/phase6_sweep.png, commitment depth, MLP routing, and D_l shape across depth {3, 6, 12, 18} (N 1000).

Every figure has a saved raw-array counterpart under DATA_ROOT and is regenerable with the
matching experiment module and the --from-cache flag.
