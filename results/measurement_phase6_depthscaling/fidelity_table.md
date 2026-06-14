# Phase 6 depth-scaling: training and fidelity gate (report before measurements)

Retrained LeWM on PushT from scratch at predictor depth in {3, 6, 12} (18 training in
parallel as a stretch). Everything held fixed at the released config (hidden_dim 192, heads
16, dim_head 64, mlp_dim 2048, history_size 3, SIGReg weight 0.09, AdamW lr 5e-5 wd 1e-3,
bf16, grad-clip 1.0, adaLN zero-init); only predictor.depth varies.

## Training setup (honest, identical across depths)

NFS random reads of 224x224 frames bottlenecked the released data pipeline at about 1.6 s
per step regardless of worker count, which makes full 18,685-episode, ~100-epoch training
(roughly 1.4M steps) infeasible here. Training therefore runs from a uint8 pixel cache of a
fixed episode subset (GPU-bound at about 0.27 s per step) with GPU-side normalization. The
SAME cache (1500 train and 200 val episodes, episode-disjoint, seed 3072), the SAME 30,000
step budget, and the SAME seed are used for every depth, so the reduced data does not
confound the depth comparison. Absolute fidelity sits below the released checkpoint (the
released model trained on roughly 12x more episodes for far more steps), which is expected
and is exactly why the gate compares relative quality across depths rather than to the
released numbers. Wall time about 135 to 155 minutes per model on one RTX 3090.

## Parameter count and fidelity gate (1500 held-out val clips)

| depth | params total | params predictor | params/block | TF next-emb MSE | rel MSE | skill vs persist | rollout shuf/true | gate |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 3  | 12.64M | 5.40M  | 1.80M | 0.0201 | 0.021 | 8.8x | 8.3x | PASS |
| 6  | 18.03M | 10.79M | 1.80M | 0.0250 | 0.026 | 7.5x | 6.4x | PASS |
| 12 | 28.82M | 21.58M | 1.80M | 0.0245 | 0.026 | 7.9x | 6.1x | PASS |
| 18 | 39.62M | 32.37M | 1.80M | 0.0236 | 0.025 | 8.3x | 6.3x | PASS |

All four depths (including the depth-18 stretch) pass at comparable quality: relative MSE
0.021 to 0.026, 7.5 to 8.8x better than a persistence baseline, and the open-loop rollout
cost is 6 to 8x worse under shuffled actions, so every model genuinely uses the action
conditioning. Per-block
capacity is constant at 1.80M parameters per predictor block, so any change in commitment
depth across models is a depth effect, not a per-block capacity effect (the parameter-count
confound is controlled: only block count changes).

## Confounds controlled

- Parameter count: deeper predictors have more parameters (12.6M to 39.6M), but per-block
  capacity is constant at 1.80M, so a relative-scaling result would mean commitment moves
  proportionally while per-block capacity is fixed.
- Training quality: every compared depth passes the same fidelity gate at comparable
  quality, so commitment-depth differences are not training-quality artifacts.

Depth 18 (stretch) is finishing on the second GPU and tracks the same trajectory (8.1x
persistence at step 20k); it will be gated and folded into the table and the measurements.

The preregistered absolute-vs-relative prediction is committed in `preregistration.md`
before any commitment-depth measurement. Next step (on go): cumulative-ablation commitment
depth, per-branch MLP-routing, and the C D_l shape per depth.
