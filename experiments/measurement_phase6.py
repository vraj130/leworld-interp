"""Phase 6 depth-scaling measurements on the retrained checkpoints.

Runs the two clean causal signals plus the propagation-shape curve on each
fidelity-passing depth checkpoint, at the readout token, eval()+fp32, seeded,
N matched to the PushT audit (default 1000 val clips). Mean ablation only.

  (2) cumulative ablation -> commitment depth (deepest l where ablating blocks >= l
      still does >= 50% of full damage). Reports absolute block index AND fraction
      of total depth (the absolute-vs-relative discriminator), plus a last-block-idle
      check.
  (3) per-branch routing -> MLP share of the branch sum (mean over blocks of
      mlp / (mlp + attn) damage). Tests whether MLP-routing is depth-specific.
  (C) D_l action-perturbation propagation at readout (final-action swap), for SHAPE
      (linear / plateau). A plateau that only appears at higher depth is itself a finding.

The retrained depth-6 model is the internal-consistency GATE: it must reproduce the
released-model audit targets (commitment depth ~2 within +-1, MLP share > 0.7) for the
reduced-regime depth comparison to bridge to the released-scale finding.

    uv run python -m experiments.measurement_phase6 --depths 6            # the gate
    uv run python -m experiments.measurement_phase6 --depths 3 6 12 18    # full sweep
    uv run python -m experiments.measurement_phase6 --from-cache --depths 3 6 12 18
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from leworld_interp import data as D
from leworld_interp import paths
from leworld_interp import pixelcache as PC
from leworld_interp.model import build_lewm, set_seed
from experiments.depth_fidelity import CKPT_ROOT, VAL_CACHE, encode, load_val_batch
from experiments.measurement_e import tf_run, rollout_final, HISTORY_SIZE, READOUT, ROLL_TARGET

RES_DIR = paths.RESULTS / "measurement_phase6_depthscaling"
ARR = paths.ACTIVATIONS / "measurement_phase6_depthscaling"
EPS = 1e-8

# released depth-6 audit values (the consistency-gate targets)
RELEASED_COMMIT = 2
RELEASED_MLP_SHARE = 0.75
COMMIT_THRESH = 0.5      # 50%-of-full bar (matches Measurement E)


def measure_depth(depth, px, action_raw, amean, astd, dev, seed, ckpt_root=CKPT_ROOT):
    cfg_p = ckpt_root / f"depth_{depth}" / "config.json"
    w_p = ckpt_root / f"depth_{depth}" / "weights.pt"
    model, cfg = build_lewm(cfg_p, w_p, device=dev, dtype=torch.float32)
    assert not model.training
    nb = len(model.predictor.transformer.layers)
    assert nb == depth, f"loaded {nb} blocks, expected {depth}"
    n_total = sum(p.numel() for p in model.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())

    emb, act_emb = encode(model, px, action_raw, amean, astd, dev)
    N = emb.size(0)
    tf_tgt = emb[:, HISTORY_SIZE]
    roll_tgt = emb[:, ROLL_TARGET]

    def mse_tf(pred):
        return float((pred[:, READOUT] - tf_tgt).pow(2).mean().item())

    def cost_roll(final):
        return float((final - roll_tgt).pow(2).sum(-1).mean().item())

    pred0, _ = tf_run(model, emb, act_emb, None, capture=False)
    mse_base = mse_tf(pred0)
    cost_base = cost_roll(rollout_final(model, emb, act_emb, None))

    # --- (2) cumulative ablation (blocks >= l) ---
    p2_mse = np.zeros(nb)
    p2_cost = np.zeros(nb)
    for l in range(nb):
        pred, _ = tf_run(model, emb, act_emb, lambda a, l=l: a.mean_ablate(list(range(l, nb))), capture=False)
        p2_mse[l] = mse_tf(pred) - mse_base
        p2_cost[l] = cost_roll(rollout_final(model, emb, act_emb,
                                             lambda a, l=l: a.mean_ablate(list(range(l, nb))))) - cost_base

    # --- (3) per-branch MLP vs attn ---
    p3 = {"mlp": np.zeros(nb), "attn": np.zeros(nb), "full": np.zeros(nb)}
    for l in range(nb):
        for br in ("mlp", "attn", "full"):
            pred, _ = tf_run(model, emb, act_emb, lambda a, l=l, br=br: a.branch_ablate(l, br), capture=False)
            p3[br][l] = mse_tf(pred) - mse_base

    # --- (C) D_l propagation (final-action swap) ---
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed + 7))
    act_final = act_emb.clone()
    act_final[:, READOUT] = act_emb[perm][:, READOUT]   # swap only the last action token (cleanest under causal mask)
    _, snap_t = tf_run(model, emb, act_emb, None, capture=True)
    _, snap_x = tf_run(model, emb, act_final, None, capture=True)
    D_l = np.zeros(nb)
    for l in range(nb):
        ht = snap_t[l]["x_out"][:, READOUT]
        D_l[l] = float(((ht - snap_x[l]["x_out"][:, READOUT]).norm(dim=-1)
                        / ht.norm(dim=-1).clamp_min(EPS)).mean().item())

    return {
        "depth": depth, "nb": nb, "n_clips": int(N),
        "params_total": int(n_total), "params_predictor": int(n_pred),
        "params_total_M": round(n_total / 1e6, 3), "params_predictor_M": round(n_pred / 1e6, 3),
        "baseline_tf_mse": mse_base, "baseline_rollout_cost": cost_base,
        "cumulative_mse_increase": p2_mse.tolist(),
        "cumulative_cost_increase": p2_cost.tolist(),
        "branch_mlp_only": p3["mlp"].tolist(),
        "branch_attn_only": p3["attn"].tolist(),
        "branch_full": p3["full"].tolist(),
        "D_l_final_swap": D_l.tolist(),
    }


def summarize_depth(r):
    nb = r["nb"]
    p2 = np.asarray(r["cumulative_mse_increase"], float)
    full = p2[0] if p2[0] > 0 else max(p2.max(), 1e-9)
    commit = 0
    for l in range(nb):
        if p2[l] >= COMMIT_THRESH * full:
            commit = l
    mlp = np.asarray(r["branch_mlp_only"], float)
    attn = np.asarray(r["branch_attn_only"], float)
    share_per_block = mlp / np.clip(mlp + attn, 1e-9, None)
    mlp_share = float(share_per_block.mean())
    # last-block-idle check: cumulative damage from ablating only the final block
    last_block_share = float(p2[nb - 1] / max(full, 1e-9))
    # D_l shape
    dl = np.asarray(r["D_l_final_swap"], float)
    dln = dl / max(dl.max(), 1e-9)
    # linearity: R^2 of a straight-line fit of D_l vs block index
    x = np.arange(nb)
    if nb >= 3:
        coef = np.polyfit(x, dl, 1)
        resid = dl - np.polyval(coef, x)
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((dl - dl.mean()) ** 2).sum())
        lin_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        # plateau: late slope vs early slope (does growth flatten in the deep third?)
        third = max(1, nb // 3)
        early_slope = (dl[third] - dl[0]) / max(third, 1)
        late_slope = (dl[-1] - dl[-1 - third]) / max(third, 1)
        plateau_ratio = float(late_slope / early_slope) if early_slope > 1e-9 else float("nan")
    else:
        lin_r2, plateau_ratio = float("nan"), float("nan")
    return {
        "depth": r["depth"], "n_clips": r["n_clips"],
        "params_total_M": r["params_total_M"], "params_predictor_M": r["params_predictor_M"],
        "commitment_depth_abs": int(commit),
        "commitment_depth_frac": round(commit / nb, 3),
        "cumulative_curve": [round(v, 5) for v in p2.tolist()],
        "last_block_frac_of_full": round(last_block_share, 3),
        "mlp_share": round(mlp_share, 3),
        "mlp_share_per_block": [round(v, 3) for v in share_per_block.tolist()],
        "D_l_final_swap": [round(v, 4) for v in dl.tolist()],
        "D_l_normalized": [round(v, 3) for v in dln.tolist()],
        "D_l_linear_r2": round(lin_r2, 3),
        "D_l_late_over_early_slope": round(plateau_ratio, 3) if plateau_ratio == plateau_ratio else None,
    }


def loss_slope_table(depths):
    """Loss-slope diagnostic from each checkpoint's logged val-TF-MSE history.

    No per-step train loss is persisted; the descent proxy is the val_tf_mse logged
    every 2000 steps during training (the metric tracked at train time). Reports
    whether the curve was still descending at the 30k cutoff or had plateaued.
    """
    rows = []
    for d in depths:
        mp = CKPT_ROOT / f"depth_{d}" / "train_meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        hist = meta.get("history", [])
        if len(hist) < 4:
            continue
        v = np.array([h["val_tf_mse"] for h in hist], float)
        steps = [h["step"] for h in hist]
        first, last, best = float(v[0]), float(v[-1]), float(v.min())
        total_descent = first - best
        final_2k_delta = float(v[-2] - v[-1])                       # +: still falling
        # final ~6k (last 3 eval intervals)
        final_6k_delta = float(v[-4] - v[-1]) if len(v) >= 4 else float("nan")
        frac_2k = final_2k_delta / max(total_descent, 1e-12)
        frac_6k = final_6k_delta / max(total_descent, 1e-12)
        # "still descending" if >=2% of the whole descent is still happening in the last 6k
        descending = bool(frac_6k > 0.02)
        rows.append({
            "depth": d, "n_evals": len(v), "first_step": steps[0], "last_step": steps[-1],
            "val_first": round(first, 5), "val_last": round(last, 5), "val_best": round(best, 5),
            "total_descent": round(total_descent, 5),
            "final_2k_delta": round(final_2k_delta, 6),
            "final_2k_frac_of_descent": round(frac_2k, 4),
            "final_6k_delta": round(final_6k_delta, 6),
            "final_6k_frac_of_descent": round(frac_6k, 4),
            "status": "still descending" if descending else "plateaued",
        })
    return rows


def gate_verdict(d6_summary):
    commit_ok = abs(d6_summary["commitment_depth_abs"] - RELEASED_COMMIT) <= 1
    mlp_ok = d6_summary["mlp_share"] > 0.7
    return {
        "retrained_d6_commitment_depth": d6_summary["commitment_depth_abs"],
        "released_target_commitment_depth": RELEASED_COMMIT,
        "commitment_within_pm1": bool(commit_ok),
        "retrained_d6_mlp_share": d6_summary["mlp_share"],
        "released_target_mlp_share": RELEASED_MLP_SHARE,
        "mlp_share_above_0p7": bool(mlp_ok),
        "GATE": "PASS" if (commit_ok and mlp_ok) else "FAIL",
    }


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--depths", type=int, nargs="+", required=True)
    pa.add_argument("--n", type=int, default=1000)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--from-cache", action="store_true")
    args = pa.parse_args()
    paths.ensure(RES_DIR, ARR)
    out_summary = RES_DIR / "measurements_summary.json"

    if args.from_cache:
        raw = {int(d): json.loads((ARR / f"depth_{d}.json").read_text()) for d in args.depths}
    else:
        set_seed(args.seed)
        am2, as2 = D.compute_action_stats(paths.PUSHT_H5)
        amean, astd = PC.action_znorm(am2, as2, 5)
        amean, astd = amean.to(args.device), astd.to(args.device)
        px, action_raw = load_val_batch(args.n, args.seed)
        print(f"[phase6 measure] depths={args.depths} on {px.size(0)} held-out val clips "
              f"(readout token {READOUT}, eval+fp32, seed {args.seed})")
        raw = {}
        for d in args.depths:
            set_seed(args.seed)
            r = measure_depth(d, px, action_raw, amean, astd, args.device, args.seed)
            (ARR / f"depth_{d}.json").write_text(json.dumps(r, indent=2))
            raw[d] = r
            print(f"  depth {d:2d}: baseline TF MSE={r['baseline_tf_mse']:.5f}")

    summaries = {int(d): summarize_depth(raw[int(d)]) for d in args.depths}
    losses = loss_slope_table(sorted({3, 6, 12, 18} | set(args.depths)))
    out = {"n_clips": args.n, "commit_threshold": COMMIT_THRESH,
           "summaries": [summaries[int(d)] for d in args.depths],
           "loss_slope": losses}
    if 6 in [int(d) for d in args.depths]:
        out["retrained_d6_gate"] = gate_verdict(summaries[6])
    out_summary.write_text(json.dumps(out, indent=2))

    # ---- report ----
    print("\n================ PHASE 6 DEPTH-SCALING MEASUREMENTS ================")
    print(f"N = {args.n} held-out val clips per depth; readout token; eval()+fp32; mean ablation\n")
    print(f"{'depth':>5} {'params(M)':>9} {'commit':>7} {'frac':>5} {'lastblk':>7} "
          f"{'MLPshare':>8} {'D_l lin R2':>10} {'D_l late/early':>13}")
    for d in args.depths:
        s = summaries[int(d)]
        lo = s["D_l_late_over_early_slope"]
        print(f"{s['depth']:>5} {s['params_total_M']:>9.2f} {s['commitment_depth_abs']:>7} "
              f"{s['commitment_depth_frac']:>5.2f} {s['last_block_frac_of_full']:>7.2f} "
              f"{s['mlp_share']:>8.2f} {s['D_l_linear_r2']:>10.2f} "
              f"{(f'{lo:.2f}' if lo is not None else 'n/a'):>13}")

    print("\nloss-slope diagnostic (logged val-TF-MSE every 2k steps; descent proxy):")
    print(f"{'depth':>5} {'val@30k':>8} {'best':>8} {'tot.desc':>9} {'final2k%':>9} {'final6k%':>9} {'status':>17}")
    for r in losses:
        print(f"{r['depth']:>5} {r['val_last']:>8.4f} {r['val_best']:>8.4f} {r['total_descent']:>9.4f} "
              f"{100*r['final_2k_frac_of_descent']:>8.2f}% {100*r['final_6k_frac_of_descent']:>8.2f}% "
              f"{r['status']:>17}")

    if "retrained_d6_gate" in out:
        g = out["retrained_d6_gate"]
        print("\n---------------- RETRAINED-d6 CONSISTENCY GATE ----------------")
        print(f"  commitment depth: retrained={g['retrained_d6_commitment_depth']} "
              f"vs released target={g['released_target_commitment_depth']}  "
              f"(within +-1: {g['commitment_within_pm1']})")
        print(f"  MLP share:        retrained={g['retrained_d6_mlp_share']:.2f} "
              f"vs released target={g['released_target_mlp_share']}  "
              f"(>0.7: {g['mlp_share_above_0p7']})")
        print(f"  GATE = {g['GATE']}")
    print("===================================================================\n")
    print(f"saved {out_summary}")


if __name__ == "__main__":
    main()
