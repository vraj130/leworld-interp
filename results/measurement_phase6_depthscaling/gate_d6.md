# Phase 6 retrained-d6 consistency gate (hard gate, run before d3/d12/d18)

The four depth-scaling models were trained in a reduced-data regime (rel-MSE 0.021 to
0.026), meaningfully weaker than the released checkpoint (rel-MSE about 0.007) the original
audit was built on. The cross-depth comparison is internally clean, but its bridge to the
released-model finding rests on one check: does the retrained depth-6 model reproduce the
released depth-6 audit result? This is treated as a gate, not a footnote.

All numbers below are at the readout token, eval() and fp32, mean ablation only, N = 1000
held-out val clips (matched to the released-model audit, which used N = 1000 / 968 episodes).
Measurements ran on CPU because GPU 1 on shannon had fallen off the PCI bus (Unknown Error),
which poisoned CUDA init node-wide; the measurements are small and inference-only, so the CPU
path is numerically equivalent in fp32.

## Gate result: PASS

| signal | released-d6 (PushT) | retrained-d6 (reduced) | gate criterion | verdict |
|---|--:|--:|---|:--|
| commitment depth (50%-of-full bar) | 2 | 2 | within +-1 of 2 | PASS |
| MLP share, mlp/(mlp+attn) | 0.905 | 0.857 | > 0.7 | PASS |
| MLP fraction, mlp/full | 0.538 | 0.641 | (context) MLP-dominant | consistent |
| last block / full (idle check) | 0.05 | 0.08 | last block near-idle | consistent |
| D_l propagation shape | ~linear, no plateau | linear R2 = 1.00, no plateau | shape preserved | consistent |

Per-block detail (l = 0..5):

- cumulative damage (ablate blocks >= l): released [.141 .115 .086 .055 .028 .007];
  retrained [.145 .122 .092 .065 .037 .012]. Same monotone decline, 50% bar crossed at the
  same block, deepest l with >= 50% of full damage is 2 in both.
- MLP share per block: retrained [.68 .79 .85 .94 .92 .96], mean 0.857; released mean 0.905.
  MLP routing dominates at every block in both models.
- D_l final-swap (normalized): retrained [.20 .36 .51 .66 .82 1.00], a straight line
  (linear-fit R2 = 1.00), monotonic, no plateau, matching the released-d6 linear D_l.

Interpretation: the reduced-data regime preserves both the commitment LOCATION (block 2) and
the MLP ROUTING (share well above the 0.7 bar), and the propagation shape. The depth law that
emerges from d3/d12/d18 can therefore be read with confidence and bridged to the released-scale
finding via this consistency, rather than only asserted in the reduced regime.

A note on the N=32 smoke: an initial low-N pass reported MLP share 0.65 (a FAIL), but the
share-of-branch-sum metric is unstable at low N because attn-only damage is tiny (about 1e-4);
at the matched N = 1000 it is 0.857. The gate is evaluated only at N = 1000.

## Loss-slope diagnostic (all four models)

No per-step train loss is persisted; the descent proxy is the val teacher-forced next-emb MSE
logged every 2000 steps during training (the metric tracked at train time). Reported as the
fraction of the whole training descent still occurring in the final 2k and final 6k steps.

| depth | val@30k | best | total descent | final-2k % of descent | final-6k % of descent | status |
|--:|--:|--:|--:|--:|--:|:--|
| 3  | 0.0210 | 0.0209 | 0.3255 | -0.01% | 0.24% | plateaued |
| 6  | 0.0252 | 0.0248 | 0.4052 | -0.09% | 0.13% | plateaued |
| 12 | 0.0234 | 0.0234 | 0.1641 |  0.01% | 1.81% | plateaued |
| 18 | 0.0228 | 0.0225 | 0.3984 | -0.07% | 0.22% | plateaued |

All four had effectively stopped descending by the 30k cutoff: the final 6k steps account for
under 2% of the total descent in every case, and final-2k deltas are near zero or slightly
negative (val MSE bouncing along its floor). Two consequences:

1. Because d6 PASSES the gate and is plateaued (converged on its data budget), the reduced
   regime is a faithful, settled solution, not a half-trained one. The gate result is not a
   transient of an undertrained model.
2. The plateau is at reduced-data fidelity (rel-MSE about 0.025), not at released quality
   (about 0.007). The residual gap to the released checkpoint is a DATA gap, not a STEP gap.
   So if any later depth were to fail a check, the lever would be more episodes, not more
   steps; 30k steps is already enough for this 1500-episode budget.

## Decision

GATE = PASS. Cleared to proceed to d3, d12, d18 and interpret the full depth-scaling curve
(commitment depth absolute index and fraction of total depth, MLP share per depth, D_l shape
per depth), with the depth law bridged to released-scale through this consistency.

Stopping here for review before measuring the other depths, as instructed.
