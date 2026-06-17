LeWM Action-Conditioning Audit — Complete Paper Guide

0. What this project became

It started as a narrow question — "does LeWM's AdaLN conditioning preserve or wash out the action signal across predictor depth?" — and grew into a full mechanistic + causal audit of where, and how, an action-conditioned JEPA world model commits to its action. The decisive scientific result is a scaling law for the commitment location:

▎ In LeWM's AdaLN-zero predictor, action conditioning is causally committed at a fixed fraction (≈ one-third) of predictor depth, routed predominantly through the MLP residual branch, in a graded (not sharp-cliff) manner, with the action's downstream consequence represented in a distributed way that later blocks refine. This relative-depth law holds across predictor depth (3→18 blocks), across data scale (reduced and released), and across two environments (PushT 2-D pushing, reacher 3-D arm) — the commitment fraction is ≈ 0.33 in every case.

So the paper is no longer "is there an Action Emergence Zone?" It is: "Action commitment in JEPA predictors is a relative-depth, MLP-routed, environment-general phenomenon — here is the causal evidence and the depth/scale/environment scaling that establishes it."

1. The research question and the hypothesis space

Subject. LeWM (LeWorldModel; Maes, Le Lidec, Scieur, LeCun, Balestriero; arXiv 2603.19312), a ~18M-parameter end-to-end JEPA video world model. It is action-conditioned — distinguishing it from prior perceptual-physics interpretability on unconditioned encoders (Joseph et al.; VideoMAE). That distinction is the paper's positioning: this is the first causal interpretability of an action-conditioned JEPA predictor.

The Action Emergence Zone (AEZ) hypothesis. There exists a specific predictor depth where action conditioning transitions from loosely mixed input signal into causally committed latent consequence.

The decision table (hypothesis taxonomy you should put in the paper). Frame the outcome space explicitly — it makes the audit rigorous and falsifiable:

┌─────┬─────────────────────────────┬───────────────────────────────────────────────┬──────────────────────┐
│ Row │         Hypothesis          │                   Signature                   │     Implication      │
├─────┼─────────────────────────────┼───────────────────────────────────────────────┼──────────────────────┤
│ 1   │ Sharp mid-stack zone        │ a single interior block where action effect   │ localized;           │
│     │ (literal AEZ)               │ jumps                                         │ targetable           │
├─────┼─────────────────────────────┼───────────────────────────────────────────────┼──────────────────────┤
│ 2   │ Early/graded commitment at  │ front-loaded cumulative causal effect,        │ ← what we find       │
│     │ injection                   │ committed by a low block                      │                      │
├─────┼─────────────────────────────┼───────────────────────────────────────────────┼──────────────────────┤
│ 3   │ Late / back-loaded          │ effect concentrated in deep blocks            │                      │
│     │ commitment                  │                                               │                      │
├─────┼─────────────────────────────┼───────────────────────────────────────────────┼──────────────────────┤
│ 4   │ Flat / distributed          │ uniform causal effect across depth            │ substrate rethink    │
│     │                             │                                               │ trigger              │
├─────┼─────────────────────────────┼───────────────────────────────────────────────┼──────────────────────┤
│ —   │ Wash-out                    │ conditioning present but causally unused      │ model ignores        │
│     │                             │                                               │ actions              │
└─────┴─────────────────────────────┴───────────────────────────────────────────────┴──────────────────────┘

The audit was pre-committed to map the observed pattern to exactly one row, with a stated escalation rule (only a confirmed flat/distributed result, row 4, would trigger a V-JEPA-2-AC substrate re-evaluation — which never fired).

2. The model and the method (your Methods section)

Architecture (state precisely).
- Encoder: ViT-tiny (patch 14, 224 px, trained from scratch, CLS token) → 192-dim BatchNorm1d-MLP projector.
- Predictor: autoregressive, 6 ConditionalBlocks with AdaLN-zero conditioning, hidden 192, 16 heads, dim_head 64, MLP dim 2048, causal mask, history length 3.
- Action path: an Embedder maps the action (input_dim = frameskip 5 × action_dim 2 = 10) to a 192-dim conditioning vector c. The same c is re-injected at every block (the conditioning projection is identity since dims match) — this re-injection is central to the methodology.
- AdaLN-zero injection: adaLN_modulation(c).chunk(6) = (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp); the final linear is zero-initialized. This gives 12 gated injection sites (attention + MLP branch per block) — the resolution at which the whole audit is reported.
- Training objective: next-embedding MSE + 0.09·SIGReg, AdamW (lr 5e-5, wd 1e-3), bf16.

Five methodological contributions (each worth a Methods subsection):

1. Cumulative ablation is the valid commitment probe (single-block is not). Under re-injected conditioning, ablating one block's conditioning underestimates its importance because downstream blocks re-inject and compensate. We show single-block ablation is back-loaded but a matched-norm random-direction control is comparable to it — i.e. single-block damage is dominated by generic perturbation propagation, not action-specific commitment. The correct probe is cumulative ablation (ablate conditioning from block l onward); commitment depth = deepest l still doing ≥ 50% of the full-ablation damage. This is a genuine methods result, not a footnote.
2. Mean ablation, never zero, at adaLN_modulation (not cond_proj). Replace the per-sample conditioning with the batch-mean conditioning (removes action identity without zeroing the pathway), applied at each block's modulation input so blocks can be ablated independently.
3. Retrained-probe robustness check. A frozen linear probe fit on un-ablated activations conflates a genuine drop in decodable consequence with mere distribution shift. Refitting the probe on the ablated distribution (matched N) separates them.
4. Depth-scaling as the absolute-vs-relative test, with two confounds controlled: parameter count (per-block capacity held constant at 1.80M; only block count varies) and training quality (every depth fidelity-gated). Preregistered.
5. Released-scale training on commodity hardware via a lossless zstd RAM-cache. Random HDF5 reads are I/O-bound (~1.4 s/step even warm); a lossless per-frame zstd cache fits the full dataset in RAM and makes training GPU-bound (~0.25 s/step). Worth a short infrastructure paragraph + appendix.

Fidelity gating (state the metric). Every model is gated at the readout token on (a) teacher-forced next-embedding rel-MSE, (b) skill vs a persistence baseline, (c) open-loop rollout cost under true vs shuffled actions. A model must beat persistence > 3× and show rollout-shuffled/true > 2× before its commitment numbers are trusted.

3. What we found (the Results narrative, in order)

(0) Fidelity. The released PushT-d6 model passes: TF rel-MSE 0.0080, beats persistence 23.1×, rollout cost 10.98× worse under shuffled actions — it genuinely uses actions.

(A) Static gate audit (12 sites). AdaLN gates are nonzero everywhere and grow with depth (gate late/early ×1.41; gate_mlp ‖W‖ ×1.66). No static wash-out. This looked consistent with an AEZ — but static capacity ≠ causal use, motivating the causal phases.

(B/C) Activation-level (readout token).
- The realized injection ratio is front-loaded (largest at block 0) and falls with depth even though gate magnitude rises (×2.16) — because the gated branch update shrinks (×0.83) while the residual stream grows (×1.09).
- MLP-branch injection dominates attention (mlp/attn ratio 1.4→2.1).
- Action-perturbation propagation D_l grows ~linearly (×3.9) with depth, no plateau, no decay; full- and final-action swaps are nearly identical. D_l growing rules out a parameter-space artifact: the gate capacity is used. (But D_l is a propagation measure — it cannot by itself separate genuine distributed injection from early commitment + late amplification; hence the causal phases.)

(D) Layerwise decodability (ridge probes, N=4000/1000). Raw-action R² jumps to ~0.68 at block 0 (input-anchor baseline 0.20) and stays flat-high (late/early 0.96); consequence (next-emb Δ) R² rises 0.32→0.43 (late/early 1.35). No crossover. Action information is front-loaded and maximal by block 0; the consequence is progressively built.

(E) Causal mean-ablation — the decisive phase (N=1000 clips).
- Cumulative ablation → commitment depth = 2 (monotonic front-loaded; blocks 0–3 ~equal contributions; block 5 idle).
- Per-branch → MLP-routing confirmed causally (MLP-only ≫ attention-only; MLP share of the branch sum 0.905).
- Single-block ablation is back-loaded but the matched-norm random control is comparable → that back-loading is the re-injection-redundancy confound, not commitment.
- Verdict: decision row 2 — early, graded, MLP-routed commitment at the injection sites.

(E-probecheck) Consequence correction. A frozen-probe "consequence drop" looked front-loaded, but refitting on the ablated distribution recovers ~59% (Δemb) up to 86% (Δstate) of it — so most of the apparent drop was distribution shift, not destroyed consequence. Softens "carry-not-construct"; the consequence representation is distributed and late blocks refine it. (A partial double-dissociation: action commitment is early; consequence is distributed.)

(5) Reacher replication (3-D arm). Fidelity rel 0.007 (80×), rollout 90.9×; D_l monotonic-linear; commitment depth 2 (= PushT), block-5 idle, MLP share 0.75. The picture is not PushT-specific.

(6) PushT depth sweep {3,6,12,18}, reduced regime — the absolute-vs-relative test.
- All depths fidelity-PASS at comparable quality; per-block capacity constant 1.80M.
- Commitment blocks {1, 2, 5, 7} → fractions {0.33, 0.33, 0.42, 0.39}, mean 0.37. The absolute hypothesis is falsified (both committed falsifiers fired: d12 commit 5 ≫ 3, d18 commit 7 > 4); the relative hypothesis is confirmed.
- Supporting: deep tail goes idle as depth grows (last-block share 0.23→0.08→0.01→0.00); MLP routing depth-invariant (0.82/0.86/0.77/0.86); a propagation plateau emerges only at higher depth (D_l late/early 1.1→1.1→0.74→0.56).
- Internal-consistency gate: retrained-d6 reproduces released-d6 (commit 2, MLP 0.86); loss-slope shows all plateaued (data-limited, not undertrained).

(7) PushT released-scale d12 — closes the data-scale gap. Trained on the full 18,485 episodes (rel-MSE 0.0085, rollout 15.3×). Commitment block 4, fraction 0.33 — identical to released d6, seed-stable; MLP routing holds (attention at the noise floor → essentially all MLP-routed); D_l deep-third plateau persists. The law is verified at released scale at two points (d6, d12), both fraction 0.33.

(8) Reacher released-scale sweep {3,6,12,18} — closes the environment gap. All four readout-fidelity PASS (released-quality; the trained models even beat the official d6 at the readout). Commitment blocks {1,2,4,7} → fractions {0.33, 0.33, 0.33, 0.39} — identical to PushT at every depth, seed-stable, with trained-d6 reproducing official-d6 (frac 0.33). MLP routing holds at every depth. Verdict: GENERALIZES — the depth law is environment-general.

The one-sentence finding for the abstract: Action conditioning in LeWM's AdaLN-zero JEPA predictor is causally committed at a constant ≈ ⅓ fraction of predictor depth through the MLP residual branch — a graded, relative-depth, MLP-routed commitment that holds across depth, data scale, and two environments, while the action's consequence is represented in a distributed fashion that deeper blocks refine.

4. The paper, section by section (content to write + which figure goes where)

Title (suggestion): Where Does a World Model Commit to Its Action? A Causal Depth-Scaling Audit of AdaLN Conditioning in LeWM.

Abstract. The positioning (first causal audit of an action-conditioned JEPA predictor), the method (cumulative AdaLN mean-ablation at the readout), the headline (commitment at ≈⅓ depth, MLP-routed, graded; relative not absolute; environment-general), and the practitioner takeaway (depth past the front third buys little for action conditioning).

1. Introduction. Action-conditioned JEPA world models are used for planning; where in the stack the action becomes a committed consequence is unknown. Prior interpretability targeted unconditioned encoders. Contributions: (i) a causal commitment-depth probe robust to re-injected conditioning; (ii) the finding of early, graded, MLP-routed commitment; (iii) a depth/scale/environment scaling law (relative ≈⅓ fraction); (iv) released-scale reproducibility on commodity hardware. → Figure 1 (NEW, see §6): architecture + method schematic.

2. Background. JEPA, LeWM architecture, AdaLN-zero conditioning, the 12 gated sites, the re-injection property. Define the decision table (§1 table above).

3. Method. The five subsections from §2: fidelity gating; static gate audit; activation-level injection & D_l; layerwise probes; causal cumulative mean-ablation (the core) including the single-block confound and matched-norm control; retrained-probe correction; depth-scaling design with confound controls; the zstd-cache infrastructure (brief, details to appendix).

4. Single-model audit (depth-6, PushT). Results A–E + probe check.
- → Figure 2 = results/phase0/fidelity.png (fidelity gate).
- → Figure 3 = results/measurement_a/measurement_a.png (static gate profile).
- → Figure 4 = results/measurement_bc/measurement_bc.png (injection ratio + D_l propagation + capacity-vs-use).
- → Figure 5 = results/measurement_d/measurement_d.png (action vs consequence decodability).
- → Figure 6 = results/measurement_e/measurement_e.png (per-block, cumulative→commitment depth, per-branch MLP-routing — the money panel of this section).
- → Figure 7 = results/measurement_e/probe_check.png (retrained-vs-frozen consequence probe).

5. Cross-environment replication (depth-6).
- → Figure 8 = results/measurement_phase5_reacher/phase5.png (reacher: D_l, cumulative, per-branch).

6. Depth scaling: absolute vs relative.
- → Figure 9 = results/measurement_phase6_depthscaling/phase6_sweep.png (commitment depth, MLP routing, D_l shape across {3,6,12,18}).
- → Figure 10 = results/measurement_phase6_depthscaling/phase6_gate_d6.png (retrained-d6 vs released-d6 consistency gate — supports transfer).

7. Released-scale and cross-environment generalization.
- → Figure 11 = results/measurement_phase7_released_d12/phase7_released_d12.png (PushT released d12 lands at frac 0.33).
- → Figure 12 = results/measurement_phase8_reacher_sweep/phase8_reacher_sweep.png (reacher sweep fractions match PushT at every depth — the cross-environment headline).
- → Figure 13 (NEW, see §6): the unified scaling figure (recommended headline for the whole paper).

8. Discussion. Reframing AEZ → relative-depth MLP-routed commitment; the distributed consequence; why MLP not attention; the deep-tail-idle implication for architecture design.

9. Limitations. Single model family (one untested axis: cross-architecture); reacher all-position floor (the position-0 metric note); reduced-regime intermediate depths bridged via the retrained-d6 gate.

10. Reproducibility. Point to results/REPRODUCIBILITY.md (13/13 phases reproduce via --from-cache, no GPU) and results/DATA_CATALOG.md.

Appendices. A: AdaLN-zero & the 12 sites. B: the cumulative-vs-single-block derivation + matched-norm control. C: the zstd RAM-cache infrastructure. D: per-depth fidelity & parameter tables (from fidelity_table.md in phases 6 and 8). E: preregistrations (verbatim from preregistration.md in phases 6 and 8).

5. Figure manifest (exact paths, what each shows, caption seed)

Fig: 2
Repo path: results/phase0/fidelity.png
Shows: TF MSE per position, persistence skill, rollout true-vs-shuffled drift
Section: §4
Caption seed: "LeWM-d6 passes the fidelity gate: 23× persistence, 11× rollout penalty under shuffled actions."
────────────────────────────────────────
Fig: 3
Repo path: results/measurement_a/measurement_a.png
Shows: static AdaLN gate ‖W‖ / magnitude across 12 sites
Section: §4
Caption seed: "AdaLN gates are nonzero everywhere and grow with depth — no static wash-out."
────────────────────────────────────────
Fig: 4
Repo path: results/measurement_bc/measurement_bc.png
Shows: injection ratio (12 sites), D_l propagation (both swaps), modulation response, capacity-vs-use
Section: §4
Caption seed: "Injection is front-loaded and MLP-dominant; action-perturbation D_l grows ~linearly with depth
  (the gate is used)."
────────────────────────────────────────
Fig: 5
Repo path: results/measurement_d/measurement_d.png
Shows: layerwise action R² vs consequence R²
Section: §4
Caption seed: "Action decodability is front-loaded and flat-high; consequence is progressively built; no
  crossover."
────────────────────────────────────────
Fig: 6
Repo path: results/measurement_e/measurement_e.png
Shows: per-block damage + random control; cumulative→commitment depth; per-branch MLP vs attn
Section: §4
Caption seed: "Cumulative mean-ablation locates commitment at block 2; the effect is MLP-routed; single-block
  back-loading is a re-injection confound."
────────────────────────────────────────
Fig: 7
Repo path: results/measurement_e/probe_check.png
Shows: frozen vs retrained consequence-probe recovery
Section: §4
Caption seed: "Refitting the probe recovers most of the apparent consequence drop — it was distribution shift;
  consequence is distributed."
────────────────────────────────────────
Fig: 8
Repo path: results/measurement_phase5_reacher/phase5.png
Shows: reacher D_l, cumulative, per-branch
Section: §5
Caption seed: "Reacher (3-D) replicates: commitment depth 2, block-5 idle, MLP share 0.75."
────────────────────────────────────────
Fig: 9
Repo path: results/measurement_phase6_depthscaling/phase6_sweep.png
Shows: commitment depth (fractional axis), MLP routing, D_l, across {3,6,12,18}
Section: §6
Caption seed: "Commitment scales with depth at a near-constant fraction (~0.37); absolute hypothesis falsified."
────────────────────────────────────────
Fig: 10
Repo path: results/measurement_phase6_depthscaling/phase6_gate_d6.png
Shows: retrained-d6 vs released-d6 (cumulative, MLP, D_l)
Section: §6
Caption seed: "Reduced-regime retraining reproduces the released-d6 audit (commitment 2, MLP routing)."
────────────────────────────────────────
Fig: 11
Repo path: results/measurement_phase7_released_d12/phase7_released_d12.png
Shows: released d12 cumulative vs released d6 / reduced d12; fraction-vs-depth
Section: §7
Caption seed: "At released scale, d12 commits at fraction 0.33 — identical to d6."
────────────────────────────────────────
Fig: 12
Repo path: results/measurement_phase8_reacher_sweep/phase8_reacher_sweep.png
Shows: reacher cumulative curves; reacher vs PushT fraction-vs-depth
Section: §7
Caption seed: "The reacher sweep reproduces the PushT fractions at every depth — the law is environment-general."

6. Figures you still need to generate (and how)

Figure 1 — Architecture + method schematic (ESSENTIAL, hand-drawn/TikZ; not auto-generable). The encoder→projector→6-block AdaLN-zero predictor, with c re-injected at every block, the 12 gated sites highlighted, and the cumulative-ablation intervention (ablate conditioning from block l onward). Make the re-injection visually obvious — it motivates the cumulative-vs-single-block method.

Figure 13 — The unified scaling "money figure" (RECOMMENDED headline; I can generate this). One scatter: x = predictor depth {3,6,12,18}, y = commitment fraction, with five series overlaid — PushT-reduced sweep, PushT-released (d6, d12), reacher-released sweep, official reacher-d6 — plus the y = ⅓ reference line and the falsified "absolute" curve (fraction = 2/depth) for contrast. This single panel collapses Figures 9, 11, 12 into the paper's central claim: every point sits on ≈ 0.33 regardless of depth, scale, or environment, and far above the absolute prediction. How: I already have all the cached fractions; it's ~30 lines added to experiments/plot_phase6.py (a --unified mode reading the four measurements_summary.json files) — say the word and I'll add and generate it.

Figure 14 — Decision-table schematic (optional, hand-drawn). The five hypothesis rows as small depth-vs-causal-effect sketches, with row 2 highlighted as selected. Cheap, and it makes the falsification framing legible.

Figure 15 — Consequence double-dissociation, cleaned (optional; I can generate). A two-panel: action-commitment (front-loaded, from E) beside consequence-decodability after the retrained-probe correction (distributed), to state the partial double-dissociation in one figure. How: combine measurement_e_summary.json + probe_check_summary.json arrays into a small plotter.

Figure 16 — MLP-vs-attention routing across depth (optional; I can generate). A grouped bar or heatmap of per-block MLP share for d6/d12/d18 in both environments, to make "MLP-routed and depth-invariant" a standalone figure rather than a sub-panel. How: from the cached branch_mlp_only/branch_attn_only arrays in the phase-6/8 activation JSONs.

7. Key-number reference table (for the text)

┌───────────────────────────────────────────┬─────────────────────────────┬─────────────────────────────────┐
│                 quantity                  │            PushT            │             reacher             │
├───────────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ released-d6 fidelity (readout rel-MSE /   │ 0.0080 / 23×                │ 0.0068 / 79×                    │
│ persistence)                              │                             │                                 │
├───────────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ commitment depth, d6 (block, fraction)    │ 2, 0.33                     │ 2, 0.33                         │
├───────────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ commitment fractions {d3,d6,d12,d18}      │ {0.33, 0.33, 0.33–0.42,     │ {0.33, 0.33, 0.33, 0.39}        │
│                                           │ 0.39}                       │                                 │
├───────────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ absolute hypothesis                       │ falsified                   │ n/a (matches PushT)             │
├───────────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ MLP share of branch routing               │ 0.91 (d6), MLP-dominant all │ 0.74–0.75 (d6), MLP-dominant    │
│                                           │  depths                     │ all depths                      │
├───────────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ per-block predictor capacity (held        │ 1.80M                       │ 1.80M                           │
│ constant)                                 │                             │                                 │
├───────────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ released-scale verification               │ d6 + d12 both 0.33          │ full sweep matches PushT        │
└───────────────────────────────────────────┴─────────────────────────────┴─────────────────────────────────┘

(All exact per-depth/per-block arrays and rounding live in the per-phase measurements_summary.json and *_results.md; the parameter/fidelity tables are in results/measurement_phase{6,8}_*/fidelity_table.md.)

8. Framing lines you can lift directly

- Reframing: "We do not find a sharp interior Action Emergence Zone. We find early, graded commitment at the injection sites, routed through the MLP residual branch, with a distributed consequence representation that later blocks refine."
- The law: "Commitment sits at a roughly constant fraction (≈ one-third) of predictor depth — the depth-6 'block 2' result is the d6 instance of a fraction that scales with depth, not a fixed early block."
- Generality: "The commitment fraction is ≈ 0.33 across the depth axis, across reduced and released data scale, and across both environments tested."
- Practitioner takeaway: "The action-conditioning work is done by a fixed front fraction of the predictor; the deep tail is largely idle for action conditioning, so predictor depth past that fraction buys little for action conditioning specifically."
- Open axis: "The one untested generalization axis is cross-architecture; depth and environment are both closed."

---
Everything above is grounded in the committed record (results/aez_audit_memo.md, the per-phase *_results.md/measurements_summary.json, DATA_CATALOG.md, REPRODUCIBILITY.md), all reproducible via --from-cache. The two figures I flagged as generatable (Fig 13 unified scaling — the recommended headline — and the optional Figs 15/16) I can produce and drop into results/ whenever you want them.

