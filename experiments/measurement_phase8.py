"""Phase 8 measurements: reacher depth sweep vs PushT (depth law cross-environment).

For each fidelity-passing reacher depth, the three clean signals (cumulative-ablation commitment
depth, per-branch MLP share, D_l shape) at N=1000, readout token, eval+fp32, mean ablation, on the
reacher held-out val zstd cache. Headline: do the reacher commitment fractions cluster near the
PushT constant (~0.33) across depths?

Robustness: commitment block re-measured across seeds at the key depths (especially d12).
Internal-consistency: reacher d6 should reproduce the Phase 5 released-reacher d6 (commit 2,
fraction 0.33, MLP share 0.75).

    uv run python -m experiments.measurement_phase8 --depths 3 6 12 18 --device cuda:0
    uv run python -m experiments.measurement_phase8 --from-cache --depths 3 6 12 18
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from leworld_interp import data as D
from leworld_interp import paths
from leworld_interp import pixelcache as PC
from leworld_interp.model import set_seed
from experiments.measurement_phase6 import measure_depth, summarize_depth, READOUT
from experiments.train_sweep import env_h5, cache_dirs, FRAMESKIP
from experiments.sweep_fidelity import load_val_batch

ENV = "reacher"
CKPT_ROOT = paths.CHECKPOINTS / f"depth_scaling_{ENV}"
ARR8 = paths.ACTIVATIONS / f"measurement_phase8_{ENV}_sweep"
RES8 = paths.RESULTS / f"measurement_phase8_{ENV}_sweep"

# PushT reference fractions (Phase 6 reduced sweep + Phase 7 released)
PUSHT_REDUCED = {3: (1, 0.33), 6: (2, 0.33), 12: (5, 0.42), 18: (7, 0.39)}
PUSHT_RELEASED = {6: (2, 0.33), 12: (4, 0.33)}
# Phase 5 released reacher d6 reference
P5_REACHER_D6 = {"commit": 2, "frac": 0.33, "mlp_share": 0.75}
BAND = (0.28, 0.40)   # PushT-constant band


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--depths", type=int, nargs="+", default=[3, 6, 12, 18])
    pa.add_argument("--n", type=int, default=1000)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--robust-depths", type=int, nargs="+", default=[6, 12])
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--from-cache", action="store_true")
    args = pa.parse_args()
    paths.ensure(ARR8, RES8)

    if args.from_cache:
        raw = {d: json.loads((ARR8 / f"depth_{d}.json").read_text()) for d in args.depths}
        robustness = json.loads((ARR8 / "robustness.json").read_text())
    else:
        _, va_dir = cache_dirs(ENV)
        am2, as2 = D.compute_action_stats(env_h5(ENV))
        amean, astd = PC.action_znorm(am2, as2, FRAMESKIP)
        amean, astd = amean.to(args.device), astd.to(args.device)
        px, action_raw = load_val_batch(va_dir, args.n, args.seed)
        print(f"[phase8 measure:{ENV}] depths={args.depths} on {px.size(0)} val clips "
              f"(readout {READOUT}, eval+fp32, mean ablation)")
        raw = {}
        for d in args.depths:
            set_seed(args.seed)
            r = measure_depth(d, px, action_raw, amean, astd, args.device, args.seed, ckpt_root=CKPT_ROOT)
            (ARR8 / f"depth_{d}.json").write_text(json.dumps(r, indent=2))
            raw[d] = r
            print(f"  depth {d:2d}: baseline TF MSE={r['baseline_tf_mse']:.5f}")
        # robustness across seeds at key depths
        robustness = []
        for d in args.robust_depths:
            for sd in (0, 1, 2):
                set_seed(sd)
                pxx, axx = load_val_batch(va_dir, args.n, sd)
                r = measure_depth(d, pxx, axx, amean, astd, args.device, sd, ckpt_root=CKPT_ROOT)
                sm = summarize_depth(r)
                robustness.append({"depth": d, "seed": sd, "commit": sm["commitment_depth_abs"],
                                   "frac": sm["commitment_depth_frac"], "mlp_share": sm["mlp_share"]})
                print(f"  robustness d{d} seed={sd}: commit={sm['commitment_depth_abs']} "
                      f"frac={sm['commitment_depth_frac']} mlp_share={sm['mlp_share']}")
        (ARR8 / "robustness.json").write_text(json.dumps(robustness, indent=2))

    summaries = {d: summarize_depth(raw[d]) for d in args.depths}
    fracs = [summaries[d]["commitment_depth_frac"] for d in args.depths]
    in_band = [BAND[0] <= f <= BAND[1] for f in fracs]
    mlp_ok = all(summaries[d]["mlp_share"] > 0.7 for d in args.depths)
    relative = max(summaries[d]["commitment_depth_abs"] for d in args.depths) > \
        min(summaries[d]["commitment_depth_abs"] for d in args.depths)  # commit grows with depth
    if all(in_band) and mlp_ok and relative:
        verdict = "GENERALIZES"
    elif relative:
        verdict = "PARTIAL (relative law, constant off 0.33)"
    else:
        verdict = "DIVERGES"

    out = {"env": ENV, "n_clips": args.n, "band": BAND, "verdict": verdict,
           "summaries": [summaries[d] for d in args.depths],
           "pusht_reduced": PUSHT_REDUCED, "pusht_released": PUSHT_RELEASED,
           "p5_reacher_d6_ref": P5_REACHER_D6, "robustness": robustness}
    (RES8 / "measurements_summary.json").write_text(json.dumps(out, indent=2))

    # ---- report ----
    print(f"\n================ PHASE 8: REACHER DEPTH SWEEP (N={args.n}) ================")
    print(f"{'depth':>5} {'params(M)':>9} {'commit':>7} {'frac':>6} {'PushT frac':>11} "
          f"{'MLPshare':>9} {'attnfloor':>10} {'D_l R2':>7} {'D_l le/ea':>10}")
    for d in args.depths:
        s = summaries[d]
        attn = np.asarray(raw[d]["branch_attn_only"], float)
        floor = "yes" if attn.min() < 1e-4 else "no"
        pf = PUSHT_RELEASED.get(d, PUSHT_REDUCED.get(d, ("", "")))[1]
        lo = s["D_l_late_over_early_slope"]
        print(f"{d:>5} {s['params_total_M']:>9.2f} {s['commitment_depth_abs']:>7} "
              f"{s['commitment_depth_frac']:>6.2f} {pf:>11} {s['mlp_share']:>9.2f} {floor:>10} "
              f"{s['D_l_linear_r2']:>7.2f} {(f'{lo:.2f}' if lo is not None else 'n/a'):>10}")
    print("\nrobustness (commit block / fraction across seeds at key depths):")
    for r in robustness:
        print(f"  d{r['depth']} seed={r['seed']}: commit={r['commit']} frac={r['frac']} mlp_share={r['mlp_share']}")
    print(f"\nreacher fractions: {[round(f,3) for f in fracs]} | in PushT band {BAND}: {in_band}")
    print(f"internal-consistency: reacher d6 commit={summaries.get(6,{}).get('commitment_depth_abs','?')} "
          f"vs Phase5 released-reacher-d6 commit={P5_REACHER_D6['commit']}")
    print(f"\nVERDICT: {verdict}")
    print("==========================================================================\n")
    print(f"saved {RES8/'measurements_summary.json'}")


if __name__ == "__main__":
    main()
