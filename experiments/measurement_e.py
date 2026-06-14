"""Phase 4, Measurement E -- causal mean-ablation of per-block adaLN conditioning.

The decisive early-vs-distributed test. Mean ablation only (replace the conditioning
into a block's adaLN_modulation with the batch-mean action embedding; never zero, never
at cond_proj). Model in eval(), fp32, seeded. The encoder is run ONCE; every ablation is
a cheap predictor re-run on cached embeddings.

Two metrics per ablation:
  * teacher-forced readout next-emb MSE increase (action-pathway damage)
  * open-loop rollout-to-goal cost increase (planning cost; the embedding-space cost the
    CEM planner minimizes), 5-step, true actions, conditioning ablated throughout.

Part 1 (per-block, 3 curves): action MSE damage; consequence-decodability drop (reuse the
  cached Measurement-D probes on the ablated block's readout residual -- Addition 1);
  matched-norm random-direction control (Addition 2).
Part 2 (cumulative, ablate blocks >= l): commitment depth = deepest l still destroying perf.
Part 3 (per-branch): MLP-chunks-only vs attn-chunks-only vs full-output, testing B's routing.

    uv run python -m experiments.measurement_e
    uv run python -m experiments.measurement_e --from-cache
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from leworld_interp import data as D
from leworld_interp import paths
from leworld_interp.ablate import AdaLNAblator
from leworld_interp.hooks import BlockCapture
from leworld_interp.model import build_lewm, set_seed

HISTORY_SIZE = 3
READOUT = HISTORY_SIZE - 1
NUM_STEPS = 8
ROLL_TARGET = 7          # 5-step open-loop rollout (frames 3..7 from a 3-frame context)
ALPHAS = np.logspace(-2, 5, 15)   # must match Measurement D
PROBE_TARGETS = ("demb", "dstate")

ACT_DIR = paths.ACTIVATIONS / "measurement_e"
RES_DIR = paths.RESULTS / "measurement_e"
D_CACHE = paths.ACTIVATIONS / "measurement_d" / "probe_cache.npz"
ARR_PATH = ACT_DIR / "e_arrays.npz"
SUMMARY_PATH = RES_DIR / "measurement_e_summary.json"
PLOT_PATH = RES_DIR / "measurement_e.png"


# ---------- model runners (ablation armed via arm_fn) ----------
@torch.inference_mode()
def tf_run(model, emb, act_emb, arm_fn, capture):
    abl = AdaLNAblator(model)
    if arm_fn:
        arm_fn(abl)
    cap = BlockCapture(model) if capture else None
    try:
        if cap:
            with cap:
                pred = model.predict(emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE])
                snap = cap.snapshot()
        else:
            pred = model.predict(emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE])
            snap = None
    finally:
        abl.clear()
    return pred, snap


@torch.inference_mode()
def rollout_final(model, emb, act_emb, arm_fn, target=ROLL_TARGET):
    abl = AdaLNAblator(model)
    if arm_fn:
        arm_fn(abl)
    try:
        emb_list = list(emb[:, :HISTORY_SIZE].unbind(1))
        for frame in range(HISTORY_SIZE, target + 1):
            t = frame - HISTORY_SIZE
            emb_trunc = torch.stack(emb_list[t:], dim=1)
            act_trunc = act_emb[:, t:frame]
            emb_list.append(model.predict(emb_trunc, act_trunc)[:, -1])
    finally:
        abl.clear()
    return emb_list[target]


def reproduce_probes(depths):
    """Reproduce the exact cached Measurement-D ridge probes (same data, deterministic)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    d = np.load(D_CACHE)
    probes = {}
    for di in depths:
        sc = StandardScaler().fit(d[f"ftr_{di}"])
        Xs = sc.transform(d[f"ftr_{di}"])
        for t in PROBE_TARGETS:
            m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Xs, d[f"ttr_{t}"])
            probes[(di, t)] = (sc, m)
    return probes


def cons_r2(probes, di, t, feat, target):
    from sklearn.metrics import r2_score

    sc, m = probes[(di, t)]
    pred = m.predict(sc.transform(feat))
    return float(r2_score(target, pred, multioutput="uniform_average"))


def compute(args):
    set_seed(args.seed)
    dev = args.device
    model, cfg = build_lewm(paths.LEWM_PUSHT_CONFIG, paths.LEWM_PUSHT_WEIGHTS, device=dev, dtype=torch.float32)
    assert not model.training
    nb = len(model.predictor.transformer.layers)

    probe = D.build_dataset(paths.PUSHT_H5, num_steps=1, frameskip=1, normalize=False)
    action_dim = probe.get_dim("action")
    frameskip = cfg["action_encoder"]["input_dim"] // action_dim
    am, asd = D.compute_action_stats(paths.PUSHT_H5)
    ds = D.build_dataset(paths.PUSHT_H5, num_steps=NUM_STEPS, frameskip=frameskip,
                         action_mean=am, action_std=asd)
    _, val = D.split_indices(len(ds), seed=3072, val_frac=0.1)
    sel = np.sort(np.random.default_rng(args.seed).choice(val, size=min(args.n, len(val)), replace=False))
    batch = D.load_batch(ds, sel)
    n_ep = len(np.unique(batch["episode_idx"].numpy()))
    print(f"[E] {len(sel)} val clips ({n_ep} episodes), {NUM_STEPS}-frame windows, "
          f"action_dim={action_dim} frameskip={frameskip}")

    # --- encode once ---
    pixels, action, state = batch["pixels"], batch["action"], batch["state"]
    embs, acts = [], []
    with torch.inference_mode():
        for i in range(0, pixels.size(0), args.batch_size):
            sl = slice(i, i + args.batch_size)
            embs.append(model.encode({"pixels": pixels[sl].to(dev)})["emb"].float())
            acts.append(model.action_encoder(action[sl].to(dev)).float())
    emb = torch.cat(embs, 0)        # (N,8,192) on device
    act_emb = torch.cat(acts, 0)    # (N,8,192) on device
    tf_tgt = emb[:, HISTORY_SIZE]                       # frame-3 embedding
    roll_tgt = emb[:, ROLL_TARGET]
    cons_tgt = {
        "demb": (emb[:, HISTORY_SIZE] - emb[:, READOUT]).cpu().numpy(),
        "dstate": (state[:, HISTORY_SIZE] - state[:, READOUT]).numpy(),
    }

    probes = reproduce_probes(list(range(1, nb + 1)))

    def mse_tf(pred):
        return float((pred[:, READOUT] - tf_tgt).pow(2).mean().item())

    def cost_roll(final):
        return float((final - roll_tgt).pow(2).sum(-1).mean().item())

    # --- baseline ---
    pred0, snap0 = tf_run(model, emb, act_emb, None, capture=True)
    mse_base = mse_tf(pred0)
    cost_base = cost_roll(rollout_final(model, emb, act_emb, None))
    cons_base = {t: np.array([cons_r2(probes, di, t, snap0[di - 1]["x_out"][:, READOUT].cpu().numpy(), cons_tgt[t])
                              for di in range(1, nb + 1)]) for t in PROBE_TARGETS}
    print(f"[E] baseline: TF MSE={mse_base:.5f}  rollout cost={cost_base:.4f}")

    # --- Part 1: per-block ---
    p1_mse = np.zeros(nb)
    p1_cost = np.zeros(nb)
    p1_consdrop = {t: np.zeros(nb) for t in PROBE_TARGETS}
    p1_ctrl_mse = np.zeros(nb)
    units = [torch.randn(emb.size(0), HISTORY_SIZE, emb.size(-1),
                         generator=torch.Generator().manual_seed(args.seed + 100 + d))
             for d in range(args.ctrl_draws)]
    for l in range(nb):
        pred, snap = tf_run(model, emb, act_emb, lambda a, l=l: a.mean_ablate([l]), capture=True)
        p1_mse[l] = mse_tf(pred) - mse_base
        p1_cost[l] = cost_roll(rollout_final(model, emb, act_emb, lambda a, l=l: a.mean_ablate([l]))) - cost_base
        feat = snap[l]["x_out"][:, READOUT].cpu().numpy()
        for t in PROBE_TARGETS:
            p1_consdrop[t][l] = cons_base[t][l] - cons_r2(probes, l + 1, t, feat, cons_tgt[t])
        # matched-norm random control (avg over draws)
        ms = []
        for u in units:
            predc, _ = tf_run(model, emb, act_emb, lambda a, l=l, u=u: a.random_ablate([l], u), capture=False)
            ms.append(mse_tf(predc) - mse_base)
        p1_ctrl_mse[l] = float(np.mean(ms))

    # --- Part 2: cumulative (blocks >= l) ---
    p2_mse = np.zeros(nb)
    p2_cost = np.zeros(nb)
    for l in range(nb):
        pred, _ = tf_run(model, emb, act_emb, lambda a, l=l: a.mean_ablate(list(range(l, nb))), capture=False)
        p2_mse[l] = mse_tf(pred) - mse_base
        p2_cost[l] = cost_roll(rollout_final(model, emb, act_emb,
                                             lambda a, l=l: a.mean_ablate(list(range(l, nb))))) - cost_base

    # --- Part 3: per-branch ---
    p3 = {"mlp": np.zeros(nb), "attn": np.zeros(nb), "full": np.zeros(nb)}
    for l in range(nb):
        for br in ("mlp", "attn", "full"):
            pred, _ = tf_run(model, emb, act_emb, lambda a, l=l, br=br: a.branch_ablate(l, br), capture=False)
            p3[br][l] = mse_tf(pred) - mse_base

    paths.ensure(ACT_DIR)
    np.savez(
        ARR_PATH, nb=nb, n=len(sel), n_ep=n_ep, mse_base=mse_base, cost_base=cost_base,
        cons_base_demb=cons_base["demb"], cons_base_dstate=cons_base["dstate"],
        p1_mse=p1_mse, p1_cost=p1_cost, p1_ctrl_mse=p1_ctrl_mse,
        p1_consdrop_demb=p1_consdrop["demb"], p1_consdrop_dstate=p1_consdrop["dstate"],
        p2_mse=p2_mse, p2_cost=p2_cost,
        p3_mlp=p3["mlp"], p3_attn=p3["attn"], p3_full=p3["full"],
    )
    return _load_cache()


def _load_cache():
    d = np.load(ARR_PATH)
    return {k: (d[k] if d[k].ndim else d[k].item()) for k in d.files}


def summarize(R, thresh=0.5):
    nb = int(R["nb"])
    p1m, p1c = R["p1_mse"], R["p1_cost"]
    ctrl = R["p1_ctrl_mse"]
    cd = R["p1_consdrop_demb"]
    p2m = R["p2_mse"]

    def shape(curve):
        c = np.asarray(curve, float)
        if c.max() <= 0:
            return "no effect"
        peak = int(np.argmax(c))
        early = c[: max(1, nb // 3)].mean()
        late = c[-max(1, nb // 3):].mean()
        if peak <= 1 and late < 0.5 * early:
            return f"FRONT-LOADED (peak block {peak}, late={late:.2g} vs early={early:.2g})"
        if (c.max() - c.min()) / c.max() < 0.35:
            return "FLAT / distributed"
        if peak >= nb - 2 and early < 0.5 * late:
            return f"BACK-LOADED (peak block {peak})"
        return f"graded (peak block {peak}, early={early:.2g} late={late:.2g})"

    # commitment depth: deepest l where cumulative damage >= thresh * full(l=0)
    full = p2m[0] if p2m[0] > 0 else max(p2m.max(), 1e-9)
    commit = 0
    for l in range(nb):
        if p2m[l] >= thresh * full:
            commit = l
    # double dissociation: action front-loaded vs consequence-drop distributed
    action_front = (np.argmax(p1m) <= 1 and p1m[-max(1, nb // 3):].mean() < 0.5 * p1m[: max(1, nb // 3)].mean())
    cd_pos = cd.clip(min=0)
    cons_distributed = (cd_pos.min() > 0.2 * max(cd_pos.max(), 1e-9)) and (cd_pos[-max(1, nb // 3):].mean() > 0.4 * cd_pos[:max(1, nb // 3)].mean())
    double_diss = bool(action_front and cons_distributed)
    # control: is true-ablation front-loading action-specific?
    ctrl_flat_or_diff = bool(np.argmax(ctrl) != np.argmax(p1m) or ctrl.max() < 0.5 * p1m.max())
    # MLP routing: does mlp-only reproduce most of full, and dominate attn-only?
    mlp_frac = (R["p3_mlp"] / np.clip(R["p3_full"], 1e-9, None)).tolist()
    mlp_vs_attn = (R["p3_mlp"] / np.clip(R["p3_mlp"] + R["p3_attn"], 1e-9, None))  # share of branch sum
    mlp_share_mean = float(np.mean(mlp_vs_attn))

    return {
        "n_clips": int(R["n"]), "n_episodes": int(R["n_ep"]),
        "baseline_tf_mse": float(R["mse_base"]), "baseline_rollout_cost": float(R["cost_base"]),
        "part1_per_block": {
            "action_mse_increase": p1m.tolist(),
            "planning_cost_increase": p1c.tolist(),
            "consequence_drop_demb": cd.tolist(),
            "consequence_drop_dstate": R["p1_consdrop_dstate"].tolist(),
            "random_control_mse_increase": ctrl.tolist(),
            "action_damage_shape": shape(p1m),
            "consequence_drop_shape": shape(np.clip(cd, 0, None)),
            "random_control_shape": shape(np.clip(ctrl, 0, None)),
        },
        "part2_cumulative": {
            "mse_increase": p2m.tolist(),
            "cost_increase": R["p2_cost"].tolist(),
            "commitment_depth": int(commit),
            "threshold_frac_of_full": thresh,
        },
        "part3_per_branch": {
            "mlp_only_mse_increase": R["p3_mlp"].tolist(),
            "attn_only_mse_increase": R["p3_attn"].tolist(),
            "full_output_mse_increase": R["p3_full"].tolist(),
            "mlp_fraction_of_full": mlp_frac,
            "mlp_fraction_mean": float(np.mean(mlp_frac)),
            "mlp_share_of_branch_sum_mean": mlp_share_mean,
        },
        "prereg_outcomes": {
            "front_loaded_per_block": bool(action_front),
            "commitment_depth_le_2": bool(commit <= 2),
            "mlp_routing_dominant": bool(mlp_share_mean > 0.7),
            "random_control_rules_out_AR_artifact": ctrl_flat_or_diff,
            "double_dissociation": double_diss,
        },
    }


def make_plot(R, s, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = int(R["nb"])
    depth = np.arange(nb)
    p1 = s["part1_per_block"]
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))

    def nz(v):
        v = np.clip(np.asarray(v, float), 0, None)
        return v / max(v.max(), 1e-9)

    ax[0, 0].plot(depth, nz(p1["action_mse_increase"]), "-o", color="#d62728", lw=2.4, label="action MSE damage")
    ax[0, 0].plot(depth, nz(p1["consequence_drop_demb"]), "-s", color="#2ca02c", lw=2.4, label="consequence-R² drop")
    ax[0, 0].plot(depth, nz(p1["random_control_mse_increase"]), "--^", color="#7f7f7f", lw=2.0, label="random-dir control")
    ax[0, 0].set_xlabel("ablated block (depth)"); ax[0, 0].set_ylabel("normalized to each max")
    ax[0, 0].set_title("(1) per-block: action vs consequence vs control\n(shape)")
    ax[0, 0].set_xticks(depth); ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(depth, p1["action_mse_increase"], "-o", color="#d62728", lw=2.4, label="true mean-ablation")
    ax[0, 1].plot(depth, p1["random_control_mse_increase"], "--^", color="#7f7f7f", lw=2.0, label="matched-norm random")
    ax[0, 1].set_xlabel("ablated block (depth)"); ax[0, 1].set_ylabel("TF next-emb MSE increase")
    ax[0, 1].set_title("(1) per-block action damage (raw)\ntrue vs control")
    ax[0, 1].set_xticks(depth); ax[0, 1].legend(fontsize=8)

    p2 = s["part2_cumulative"]
    ax[1, 0].plot(depth, p2["mse_increase"], "-o", color="#1f77b4", lw=2.4, label="cumulative MSE increase")
    ax[1, 0].axvline(p2["commitment_depth"], color="k", ls="--", lw=1.3,
                     label=f"commitment depth = {p2['commitment_depth']}")
    ax[1, 0].axhline(p2["threshold_frac_of_full"] * p2["mse_increase"][0], color="#aaa", ls=":", lw=1)
    ax[1, 0].set_xlabel("ablate blocks ≥ l"); ax[1, 0].set_ylabel("TF MSE increase")
    ax[1, 0].set_title("(2) cumulative ablation → commitment depth")
    ax[1, 0].set_xticks(depth); ax[1, 0].legend(fontsize=8)

    p3 = s["part3_per_branch"]
    w = 0.38
    ax[1, 1].bar(depth - w / 2, p3["mlp_only_mse_increase"], w, color="#d62728", label="MLP-chunks only")
    ax[1, 1].bar(depth + w / 2, p3["attn_only_mse_increase"], w, color="#1f77b4", label="attn-chunks only")
    ax[1, 1].plot(depth, p3["full_output_mse_increase"], "k_-", lw=0, marker="_", ms=18, label="full (both)")
    ax[1, 1].set_xlabel("ablated block (depth)"); ax[1, 1].set_ylabel("TF MSE increase")
    ax[1, 1].set_title(f"(3) per-branch (MLP frac of full ≈ {p3['mlp_fraction_mean']:.2f})")
    ax[1, 1].set_xticks(depth); ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(s):
    p1, p2, p3, po = s["part1_per_block"], s["part2_cumulative"], s["part3_per_branch"], s["prereg_outcomes"]
    print("\n================ MEASUREMENT E (causal mean-ablation) ================")
    print(f"clips={s['n_clips']} episodes={s['n_episodes']}  "
          f"baseline TF MSE={s['baseline_tf_mse']:.5f}  rollout cost={s['baseline_rollout_cost']:.4f}")
    def row(n, v, f="{:8.4f}"):
        print(f"  {n:<26}", " ".join(f.format(x) for x in v))
    print("\nPart 1 (per-block, l=0..5):")
    row("action MSE damage", p1["action_mse_increase"])
    row("planning cost damage", p1["planning_cost_increase"])
    row("consequence drop (Δemb)", p1["consequence_drop_demb"])
    row("random-dir control MSE", p1["random_control_mse_increase"])
    print(f"  shapes: action={p1['action_damage_shape']}")
    print(f"          consequence-drop={p1['consequence_drop_shape']}")
    print(f"          control={p1['random_control_shape']}")
    print("\nPart 2 (cumulative ablate ≥ l):")
    row("cumulative MSE increase", p2["mse_increase"])
    print(f"  commitment depth = {p2['commitment_depth']}  (deepest l with ≥{p2['threshold_frac_of_full']:.0%} of full damage)")
    print("\nPart 3 (per-branch MSE increase):")
    row("MLP-only", p3["mlp_only_mse_increase"]); row("attn-only", p3["attn_only_mse_increase"])
    row("full (both)", p3["full_output_mse_increase"])
    print(f"  MLP fraction of full (mean) = {p3['mlp_fraction_mean']:.2f}")
    print("\nPREREG OUTCOMES:")
    for k, v in po.items():
        print(f"  [{'YES' if v else 'no ':>3}] {k}: {v}")
    print("=====================================================================\n")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--n", type=int, default=1000)
    pa.add_argument("--batch-size", type=int, default=256)
    pa.add_argument("--ctrl-draws", type=int, default=3)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:1")
    pa.add_argument("--from-cache", action="store_true")
    args = pa.parse_args()
    R = _load_cache() if args.from_cache else compute(args)
    summary = summarize(R)
    paths.ensure(RES_DIR)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    make_plot(R, summary, PLOT_PATH)
    report(summary)


if __name__ == "__main__":
    main()
