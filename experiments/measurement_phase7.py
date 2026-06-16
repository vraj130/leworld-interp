"""Phase 7 measurements: released-scale d12 vs reduced-d12 and released-d6.

The reviewer gap was that the Phase 6 depth law lived in the reduced-data regime, verified at
released scale only at d6. This measures the released-scale d12 (rel-MSE 0.0085, fidelity PASS)
on the same three clean signals and asks whether its commitment fraction lands in the
reduced-regime band (~0.35-0.45) -> depth law VERIFIED at two released-scale points (d6, d12);
or moves toward the absolute prediction (block ~2, frac ~0.17) -> data scale shifts commitment.

Signals (N=1000, readout token, eval+fp32, mean ablation):
  (2) cumulative-ablation commitment depth: absolute block + fraction of total depth.
  (3) per-branch MLP share, with the d12 noise-floor caveat (attn-only damage can approach
      machine epsilon at higher depth, making the share ratio fragile).
  (C) D_l propagation shape / deep-third slope (does the emergent plateau persist at released scale).

Robustness: the commitment block is re-checked across seeds and N to confirm it is not a
knife-edge of the 50% threshold.

    uv run python -m experiments.measurement_phase7 --device cuda:0
    uv run python -m experiments.measurement_phase7 --from-cache
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
from experiments.depth_fidelity import load_val_batch
from experiments.measurement_phase6 import measure_depth, summarize_depth, READOUT

RELEASED_ROOT = paths.CHECKPOINTS / "depth_scaling_released"
REDUCED_D12 = paths.ACTIVATIONS / "measurement_phase6_depthscaling" / "depth_12.json"
REL_E = paths.RESULTS / "measurement_e" / "measurement_e_summary.json"
ARR7 = paths.ACTIVATIONS / "measurement_phase7_released_d12"
RES7 = paths.RESULTS / "measurement_phase7_released_d12"
DEPTH = 12
BAND = (0.33, 0.50)   # reduced-regime / relative band for the commitment fraction


def released_d6_ref():
    s = json.load(open(REL_E))
    p2 = np.asarray(s["part2_cumulative"]["mse_increase"], float)
    commit = max([l for l in range(len(p2)) if p2[l] >= 0.5 * p2[0]])
    return {"commit": commit, "frac": round(commit / len(p2), 3),
            "mlp_share": round(s["part3_per_branch"]["mlp_share_of_branch_sum_mean"], 3)}


def attn_floor_note(raw):
    attn = np.asarray(raw["branch_attn_only"], float)
    mn = float(attn.min())
    return {"attn_only_min": mn, "attn_only_per_block": [round(x, 5) for x in attn.tolist()],
            "near_machine_floor": bool(mn < 1e-4)}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--n", type=int, default=1000)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--from-cache", action="store_true")
    args = pa.parse_args()
    paths.ensure(ARR7, RES7)
    arr_path = ARR7 / "depth_12_released.json"
    out_summary = RES7 / "measurements_summary.json"

    if args.from_cache:
        raw = json.loads(arr_path.read_text())
        robustness = json.loads((ARR7 / "robustness.json").read_text())
    else:
        am2, as2 = D.compute_action_stats(paths.PUSHT_H5)
        amean, astd = PC.action_znorm(am2, as2, 5)
        amean, astd = amean.to(args.device), astd.to(args.device)

        # primary measurement
        set_seed(args.seed)
        px, action_raw = load_val_batch(args.n, args.seed)
        print(f"[phase7] released d12 primary: N={px.size(0)} seed={args.seed} "
              f"(readout {READOUT}, eval+fp32, mean ablation)")
        raw = measure_depth(DEPTH, px, action_raw, amean, astd, args.device, args.seed,
                            ckpt_root=RELEASED_ROOT)
        arr_path.write_text(json.dumps(raw, indent=2))

        # robustness: vary seed and N, record commitment block + fraction + MLP share
        robustness = []
        for (sd, n) in [(args.seed, args.n), (1, args.n), (2, args.n), (args.seed, 1500)]:
            set_seed(sd)
            pxx, axx = load_val_batch(n, sd)
            r = measure_depth(DEPTH, pxx, axx, amean, astd, args.device, sd, ckpt_root=RELEASED_ROOT)
            sm = summarize_depth(r)
            robustness.append({"seed": sd, "n": n, "commit": sm["commitment_depth_abs"],
                               "frac": sm["commitment_depth_frac"], "mlp_share": sm["mlp_share"]})
            print(f"  robustness seed={sd} N={n}: commit={sm['commitment_depth_abs']} "
                  f"frac={sm['commitment_depth_frac']} mlp_share={sm['mlp_share']}")
        (ARR7 / "robustness.json").write_text(json.dumps(robustness, indent=2))

    s = summarize_depth(raw)
    d6 = released_d6_ref()
    red = summarize_depth(json.loads(REDUCED_D12.read_text()))
    floor = attn_floor_note(raw)

    frac = s["commitment_depth_frac"]
    verified = (BAND[0] - 0.03) <= frac <= (BAND[1] + 0.05) and s["mlp_share"] > 0.7
    verdict = "VERIFIED" if verified else "DIVERGED"

    out = {
        "n_clips": args.n, "depth": DEPTH, "scale": "released",
        "released_d12": s, "reduced_d12": red, "released_d6_ref": d6,
        "attn_floor": floor, "robustness": robustness,
        "commitment_band": BAND, "verdict": verdict,
    }
    out_summary.write_text(json.dumps(out, indent=2))

    # ---- report ----
    print("\n================ PHASE 7: RELEASED-SCALE d12 MEASUREMENTS ================")
    print(f"N = {args.n} held-out val clips; readout token; eval()+fp32; mean ablation\n")
    print(f"{'model':<22} {'scale':<9} {'commit':>7} {'frac':>6} {'MLPshare':>9} {'D_l R2':>7} {'D_l late/early':>14}")
    rd6 = f"{'released d6 (ref)':<22} {'released':<9} {d6['commit']:>7} {d6['frac']:>6.2f} {d6['mlp_share']:>9.2f} {'~1.0':>7} {'~1.0':>14}"
    lo = s["D_l_late_over_early_slope"]
    rd12 = (f"{'released d12 (PH7)':<22} {'released':<9} {s['commitment_depth_abs']:>7} "
            f"{s['commitment_depth_frac']:>6.2f} {s['mlp_share']:>9.2f} {s['D_l_linear_r2']:>7.2f} "
            f"{(f'{lo:.2f}' if lo is not None else 'n/a'):>14}")
    rlo = red["D_l_late_over_early_slope"]
    rred = (f"{'reduced d12 (PH6)':<22} {'reduced':<9} {red['commitment_depth_abs']:>7} "
            f"{red['commitment_depth_frac']:>6.2f} {red['mlp_share']:>9.2f} {red['D_l_linear_r2']:>7.2f} "
            f"{(f'{rlo:.2f}' if rlo is not None else 'n/a'):>14}")
    print(rd6); print(rd12); print(rred)

    print(f"\ncumulative damage / full (released d12, l=0..{DEPTH-1}):")
    cum = np.asarray(raw["cumulative_mse_increase"], float); cum = cum / cum[0]
    print("  " + " ".join(f"{x:.2f}" for x in cum))
    print(f"  last-block/full={s['last_block_frac_of_full']:.3f}")

    print("\nMLP-share noise-floor check (released d12):")
    print(f"  attn-only per block: {floor['attn_only_per_block']}")
    print(f"  attn-only min={floor['attn_only_min']:.5f}  near machine floor (<1e-4): {floor['near_machine_floor']}")

    print("\nrobustness (commitment block / fraction / MLP share across seeds and N):")
    for r in robustness:
        print(f"  seed={r['seed']} N={r['n']}: commit={r['commit']} frac={r['frac']} mlp_share={r['mlp_share']}")

    print(f"\nVERDICT: {verdict}  (released-d12 frac {frac} vs band {BAND}; "
          f"reduced-d12 {red['commitment_depth_frac']}, released-d6 {d6['frac']})")
    print("==========================================================================\n")
    print(f"saved {out_summary}")


if __name__ == "__main__":
    main()
