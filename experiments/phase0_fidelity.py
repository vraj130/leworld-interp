"""Phase 0 fidelity gate for the LeWM AEZ audit.

Before any audit number is trusted, prove the loaded model reproduces known
behavior. Two checks, both on held-out PushT clips, in ``eval()`` + fp32, seeded:

  (A) Teacher-forced next-embedding MSE -- the exact training objective
      (``lejepa_forward``): encode ``history_size`` frames, predict, compare to
      the next encoded embeddings. Reported overall (matches training) and per
      readout position, against a persistence baseline.

  (B) Open-loop rollout with TRUE vs within-batch SHUFFLED actions, using the
      model's own ``rollout`` method. The true-action rollout should track the
      encoder's future embeddings far better than the shuffled-action rollout --
      this confirms the predictor + rollout machinery is sane and genuinely uses
      the action conditioning. The final-step divergence is the embedding-space
      plan cost the CEM planner minimizes.

Raw per-clip arrays are written under DATA_ROOT (regenerable plots); a scalar
summary JSON and a plot go under results/. Run::

    uv run python -m experiments.phase0_fidelity
    uv run python -m experiments.phase0_fidelity --from-cache   # replot only
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from leworld_interp import data as D
from leworld_interp import paths
from leworld_interp.model import build_lewm, set_seed

HISTORY_SIZE = 3  # ctx_len (config: history_size)
NUM_PREDS = 1     # config: num_preds


@torch.inference_mode()
def encode_windows(model, pixels, actions, device, chunk):
    """Encode (B,T,C,H,W) pixels [+ (B,T,A) actions] -> emb (B,T,D) [, act_emb]."""
    embs, acts = [], []
    for i in range(0, pixels.size(0), chunk):
        info = {"pixels": pixels[i : i + chunk].to(device)}
        if actions is not None:
            info["action"] = actions[i : i + chunk].to(device)
        out = model.encode(info)
        embs.append(out["emb"].float().cpu())
        if actions is not None:
            acts.append(out["act_emb"].float().cpu())
    emb = torch.cat(embs, 0)
    return (emb, torch.cat(acts, 0)) if actions is not None else emb


@torch.inference_mode()
def teacher_forced(model, emb, act_emb, device, chunk):
    """Predict next embeddings from the first HISTORY_SIZE frames; per-clip,
    per-position squared error vs the encoder's next embeddings."""
    ctx_emb, ctx_act = emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE]
    tgt = emb[:, NUM_PREDS : NUM_PREDS + HISTORY_SIZE]  # emb[:, 1:1+3]
    se = []  # (B, pos) mean-over-dim squared error
    for i in range(0, ctx_emb.size(0), chunk):
        pred = model.predict(ctx_emb[i : i + chunk].to(device), ctx_act[i : i + chunk].to(device))
        se.append((pred.float().cpu() - tgt[i : i + chunk]).pow(2).mean(dim=-1))
    return torch.cat(se, 0).numpy(), tgt.numpy()  # (B, HISTORY_SIZE)


@torch.inference_mode()
def rollout_future(model, ctx_pixels, action_seq, device, chunk):
    """Open-loop rollout from H=ctx context frames; returns predicted emb
    trajectory (B, L+1, D) where L = action_seq length."""
    outs = []
    for i in range(0, ctx_pixels.size(0), chunk):
        info = {"pixels": ctx_pixels[i : i + chunk].unsqueeze(1).to(device)}  # (b,1,H,C,H,W)
        acts = action_seq[i : i + chunk].unsqueeze(1).to(device)             # (b,1,L,A)
        info = model.rollout(info, acts)
        outs.append(info["predicted_emb"][:, 0].float().cpu())              # (b, L+1, D)
    return torch.cat(outs, 0)


def run_measurements(args):
    set_seed(args.seed)
    device = args.device
    out_dir = paths.ACTIVATIONS / "phase0"
    paths.ensure(out_dir)

    model, cfg = build_lewm(
        paths.LEWM_PUSHT_CONFIG, paths.LEWM_PUSHT_WEIGHTS, device=device, dtype=torch.float32
    )
    assert not model.training, "model must be in eval() for the audit"

    # --- frameskip / action_dim read from data + config, never hardcoded ---
    probe = D.build_dataset(paths.PUSHT_H5, num_steps=1, frameskip=1, normalize=False)
    action_dim = probe.get_dim("action")
    act_input_dim = cfg["action_encoder"]["input_dim"]
    assert act_input_dim % action_dim == 0, (act_input_dim, action_dim)
    frameskip = act_input_dim // action_dim
    print(f"[cfg] action_dim={action_dim}  action_encoder.input_dim={act_input_dim}  "
          f"=> frameskip={frameskip}  history_size={HISTORY_SIZE}  num_preds={NUM_PREDS}")

    action_mean, action_std = D.compute_action_stats(paths.PUSHT_H5)

    # ============ (A) teacher-forced next-embedding MSE ============
    ds_tf = D.build_dataset(
        paths.PUSHT_H5, num_steps=HISTORY_SIZE + NUM_PREDS, frameskip=frameskip,
        action_mean=action_mean, action_std=action_std,
    )
    _, val_tf = D.split_indices(len(ds_tf), seed=3072, val_frac=0.1)
    sel_tf = np.random.default_rng(args.seed).choice(val_tf, size=min(args.n_tf, len(val_tf)), replace=False)
    batch_tf = D.load_batch(ds_tf, sel_tf)
    n_ep_tf = len(np.unique(batch_tf["episode_idx"].numpy()))
    print(f"[A] teacher-forced on {len(sel_tf)} val clips spanning {n_ep_tf} episodes")

    emb, act_emb = encode_windows(model, batch_tf["pixels"], batch_tf["action"], device, args.batch_size)
    se_tf, tgt = teacher_forced(model, emb, act_emb, device, args.batch_size)  # (B,3)

    tgt_energy = float((tgt ** 2).mean())                       # mean target emb energy / dim
    # persistence baseline: predict emb[:,t+1] with emb[:,t]
    persist = ((emb[:, :HISTORY_SIZE] - emb[:, NUM_PREDS : NUM_PREDS + HISTORY_SIZE]) ** 2).mean(dim=-1).numpy()
    tf_mse_overall = float(se_tf.mean())                        # == training pred_loss
    tf_mse_per_pos = se_tf.mean(0)                              # (3,)
    persist_overall = float(persist.mean())
    persist_per_pos = persist.mean(0)

    # ============ (B) open-loop rollout: true vs shuffled actions ============
    horizon = args.horizon
    num_steps_ro = HISTORY_SIZE + horizon
    ds_ro = D.build_dataset(
        paths.PUSHT_H5, num_steps=num_steps_ro, frameskip=frameskip,
        action_mean=action_mean, action_std=action_std,
    )
    _, val_ro = D.split_indices(len(ds_ro), seed=3072, val_frac=0.1)
    sel_ro = np.random.default_rng(args.seed + 1).choice(val_ro, size=min(args.n_rollout, len(val_ro)), replace=False)
    batch_ro = D.load_batch(ds_ro, sel_ro)
    n_ep_ro = len(np.unique(batch_ro["episode_idx"].numpy()))
    print(f"[B] rollout on {len(sel_ro)} val clips spanning {n_ep_ro} episodes, horizon={horizon}")

    pixels_ro, actions_ro = batch_ro["pixels"], batch_ro["action"]  # (B, num_steps, ...)
    true_emb = encode_windows(model, pixels_ro, None, device, args.batch_size)  # (B, num_steps, D)
    ctx_pixels = pixels_ro[:, :HISTORY_SIZE]                                     # (B, H, C, H, W)

    perm = torch.randperm(actions_ro.size(0), generator=torch.Generator().manual_seed(args.seed + 2))
    pred_true = rollout_future(model, ctx_pixels, actions_ro, device, args.batch_size)         # (B, L+1, D)
    pred_shuf = rollout_future(model, ctx_pixels, actions_ro[perm], device, args.batch_size)

    # aligned future frames live at indices H .. num_steps-1
    fut = slice(HISTORY_SIZE, num_steps_ro)
    tf_true = true_emb[:, fut]                       # (B, horizon, D)
    eps = 1e-8

    def drift(pred):
        p = pred[:, fut]                              # (B, horizon, D)
        num = (p - tf_true).pow(2).sum(dim=-1)        # (B, horizon)
        den = tf_true.pow(2).sum(dim=-1).clamp_min(eps)
        return (num / den).numpy()                    # normalized per-clip, per-horizon

    drift_true = drift(pred_true)                     # (B, horizon)
    drift_shuf = drift(pred_shuf)
    # final-step plan cost (== model.criterion's sum-MSE) at the last real frame
    cost_true = (pred_true[:, num_steps_ro - 1] - true_emb[:, num_steps_ro - 1]).pow(2).sum(-1).numpy()
    cost_shuf = (pred_shuf[:, num_steps_ro - 1] - true_emb[:, num_steps_ro - 1]).pow(2).sum(-1).numpy()

    # ---- persist raw arrays (regenerable) ----
    np.savez(
        out_dir / "fidelity_arrays.npz",
        se_tf=se_tf, persist=persist, tgt_energy=tgt_energy,
        drift_true=drift_true, drift_shuf=drift_shuf,
        cost_true=cost_true, cost_shuf=cost_shuf,
        horizon=horizon, frameskip=frameskip, action_dim=action_dim,
    )

    summary = {
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(paths.LEWM_PUSHT_WEIGHTS),
        "frameskip": int(frameskip),
        "action_dim": int(action_dim),
        "history_size": HISTORY_SIZE,
        "num_preds": NUM_PREDS,
        "n_tf_clips": int(len(sel_tf)),
        "n_tf_episodes": int(n_ep_tf),
        "n_rollout_clips": int(len(sel_ro)),
        "n_rollout_episodes": int(n_ep_ro),
        "horizon": horizon,
        "teacher_forced": {
            "next_emb_mse_overall": tf_mse_overall,
            "next_emb_mse_per_position": tf_mse_per_pos.tolist(),
            "readout_pos_mse": float(tf_mse_per_pos[-1]),
            "persistence_mse_overall": persist_overall,
            "persistence_mse_per_position": persist_per_pos.tolist(),
            "target_emb_energy_per_dim": tgt_energy,
            "relative_mse_overall": tf_mse_overall / tgt_energy,
            "skill_vs_persistence": persist_overall / tf_mse_overall,
        },
        "rollout": {
            "drift_true_per_horizon": drift_true.mean(0).tolist(),
            "drift_shuf_per_horizon": drift_shuf.mean(0).tolist(),
            "final_cost_true": float(cost_true.mean()),
            "final_cost_shuf": float(cost_shuf.mean()),
            "shuf_over_true_final": float(cost_shuf.mean() / max(cost_true.mean(), eps)),
        },
    }
    res_dir = paths.RESULTS / "phase0"
    paths.ensure(res_dir)
    (res_dir / "fidelity_summary.json").write_text(json.dumps(summary, indent=2))
    make_plot(summary, drift_true, drift_shuf, res_dir / "fidelity.png")
    report(summary)
    return summary


def make_plot(summary, drift_true, drift_shuf, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tf = summary["teacher_forced"]
    pos = np.arange(HISTORY_SIZE)
    ax[0].bar(pos - 0.2, tf["next_emb_mse_per_position"], 0.4, label="LeWM predictor")
    ax[0].bar(pos + 0.2, tf["persistence_mse_per_position"], 0.4, label="persistence")
    ax[0].set_xlabel("predicted position (causal context length)")
    ax[0].set_ylabel("next-embedding MSE")
    ax[0].set_title("(A) teacher-forced next-emb MSE")
    ax[0].set_xticks(pos)
    ax[0].legend()

    h = np.arange(1, drift_true.shape[1] + 1)
    ax[1].plot(h, drift_true.mean(0), "-o", label="true actions")
    ax[1].plot(h, drift_shuf.mean(0), "-s", label="shuffled actions")
    ax[1].set_xlabel("rollout horizon (steps ahead)")
    ax[1].set_ylabel("normalized rollout divergence")
    ax[1].set_title("(B) open-loop rollout vs encoder")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(s):
    tf, ro = s["teacher_forced"], s["rollout"]
    print("\n================ PHASE 0 FIDELITY ================")
    print(f"teacher-forced next-emb MSE (== training pred_loss): {tf['next_emb_mse_overall']:.5f}")
    print(f"  per position {tf['next_emb_mse_per_position']}  (readout={tf['readout_pos_mse']:.5f})")
    print(f"  target emb energy/dim={tf['target_emb_energy_per_dim']:.5f}  "
          f"relative MSE={tf['relative_mse_overall']:.4f}")
    print(f"  persistence MSE={tf['persistence_mse_overall']:.5f}  "
          f"=> model beats persistence by {tf['skill_vs_persistence']:.2f}x")
    print(f"rollout final plan cost: true={ro['final_cost_true']:.4f}  shuffled={ro['final_cost_shuf']:.4f}  "
          f"(shuffled/true={ro['shuf_over_true_final']:.2f}x)")
    print(f"rollout drift true   : {[round(x,4) for x in ro['drift_true_per_horizon']]}")
    print(f"rollout drift shuffle: {[round(x,4) for x in ro['drift_shuf_per_horizon']]}")
    print("==================================================\n")


def replot_from_cache():
    arr = np.load(paths.ACTIVATIONS / "phase0" / "fidelity_arrays.npz")
    summary = json.loads((paths.RESULTS / "phase0" / "fidelity_summary.json").read_text())
    make_plot(summary, arr["drift_true"], arr["drift_shuf"], paths.RESULTS / "phase0" / "fidelity.png")
    report(summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-tf", type=int, default=512)
    p.add_argument("--n-rollout", type=int, default=256)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--from-cache", action="store_true", help="re-plot from saved arrays, no model")
    args = p.parse_args()
    if args.from_cache:
        replot_from_cache()
    else:
        run_measurements(args)


if __name__ == "__main__":
    main()
