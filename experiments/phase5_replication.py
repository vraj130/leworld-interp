"""Phase 5 -- 3D-environment replication of the three least-ambiguous signals.

Scoped replication (NOT the full audit). Fidelity-gated first, then:
  (1) C: D_l action-perturbation propagation at the readout (full-swap and final-swap).
  (2) E: cumulative mean-ablation (ablate blocks >= l) -> commitment depth, block-5 idle.
  (3) E: per-branch mean-ablation (MLP chunks vs attn chunks) -> MLP share.

Mean ablation only; readout position; eval()+fp32; seeded. The model/data are selected by
--env (default reacher). action_dim and frameskip are read from the data/config, never
hardcoded. Outputs go to results/measurement_phase5_<env>/ and
DATA_ROOT/activations/measurement_phase5_<env>/. --from-cache replots without the model.

    uv run python -m experiments.phase5_replication --env reacher
    uv run python -m experiments.phase5_replication --env reacher --from-cache
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
from experiments.measurement_e import rollout_final, tf_run, HISTORY_SIZE, READOUT, ROLL_TARGET

NUM_STEPS = 8
EPS = 1e-8


def env_paths(env):
    cdir = paths.CHECKPOINTS / f"lewm-{env}"
    ddir = paths.DATASETS / f"lewm-{env}"
    h5s = sorted(ddir.glob("*.h5"))
    h5 = h5s[0] if h5s else ddir / f"{env}.h5"
    return cdir / "config.json", cdir / "weights.pt", h5


def out_dirs(env):
    return (paths.ACTIVATIONS / f"measurement_phase5_{env}",
            paths.RESULTS / f"measurement_phase5_{env}")


@torch.inference_mode()
def encode(model, pixels, action, dev, bs):
    embs, acts = [], []
    for i in range(0, pixels.size(0), bs):
        sl = slice(i, i + bs)
        embs.append(model.encode({"pixels": pixels[sl].to(dev)})["emb"].float())
        acts.append(model.action_encoder(action[sl].to(dev)).float())
    return torch.cat(embs, 0), torch.cat(acts, 0)


@torch.inference_mode()
def measure_C(model, emb, act_true, act_full, act_final):
    """D_l at readout for full-swap and final-swap. emb/act_* are (N, T>=3, 192)."""
    cap = BlockCapture(model)
    nb = len(model.predictor.transformer.layers)
    with cap:
        model.predictor(emb[:, :HISTORY_SIZE], act_true[:, :HISTORY_SIZE]); st = cap.snapshot()
        model.predictor(emb[:, :HISTORY_SIZE], act_full[:, :HISTORY_SIZE]); sf = cap.snapshot()
        model.predictor(emb[:, :HISTORY_SIZE], act_final[:, :HISTORY_SIZE]); sx = cap.snapshot()
    r = READOUT
    Dfull, Dfinal = [], []
    for bi in range(nb):
        ht = st[bi]["x_out"][:, r]
        Dfull.append(float(((ht - sf[bi]["x_out"][:, r]).norm(dim=-1) / ht.norm(dim=-1).clamp_min(EPS)).mean()))
        Dfinal.append(float(((ht - sx[bi]["x_out"][:, r]).norm(dim=-1) / ht.norm(dim=-1).clamp_min(EPS)).mean()))
    return Dfull, Dfinal


def compute(args):
    set_seed(args.seed)
    dev = args.device
    cfg_p, w_p, h5 = env_paths(args.env)
    if not h5.exists():
        raise FileNotFoundError(f"{h5} not found; extract the dataset first")
    model, cfg = build_lewm(cfg_p, w_p, device=dev, dtype=torch.float32)
    assert not model.training
    nb = len(model.predictor.transformer.layers)

    probe = D.build_dataset(h5, num_steps=1, frameskip=1, normalize=False, keys_to_load=["pixels", "action"])
    action_dim = probe.get_dim("action")
    frameskip = cfg["action_encoder"]["input_dim"] // action_dim
    am, asd = D.compute_action_stats(h5)
    ds = D.build_dataset(h5, num_steps=NUM_STEPS, frameskip=frameskip,
                         action_mean=am, action_std=asd, keys_to_load=["pixels", "action"])
    _, val = D.split_indices(len(ds), seed=3072, val_frac=0.1)
    sel = np.sort(np.random.default_rng(args.seed).choice(val, size=min(args.n, len(val)), replace=False))
    batch = D.load_batch(ds, sel)
    print(f"[P5:{args.env}] {len(sel)} val clips of {len(ds)} (action_dim={action_dim} frameskip={frameskip})")

    pixels, action = batch["pixels"], batch["action"]
    emb, act_emb = encode(model, pixels, action, dev, args.batch_size)
    N = emb.size(0)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed + 7))
    act_full_raw = action[perm]
    act_final_raw = action.clone(); act_final_raw[:, READOUT] = action[perm][:, READOUT]
    act_full = model.action_encoder(act_full_raw.to(dev)).float()
    act_final = model.action_encoder(act_final_raw.to(dev)).float()

    tf_tgt = emb[:, HISTORY_SIZE]
    roll_tgt = emb[:, ROLL_TARGET]

    # ---- FIDELITY ----
    pred0, _ = tf_run(model, emb, act_emb, None, capture=False)
    tf_mse = float((pred0[:, READOUT] - tf_tgt).pow(2).mean())
    persist = float((emb[:, READOUT] - tf_tgt).pow(2).mean())
    cost_true = float((rollout_final(model, emb, act_emb, None) - roll_tgt).pow(2).sum(-1).mean())
    act_shuf = model.action_encoder(act_full_raw.to(dev)).float()
    cost_shuf = float((rollout_final(model, emb, act_shuf, None) - roll_tgt).pow(2).sum(-1).mean())
    tgt_energy = float((tf_tgt ** 2).mean())
    fidelity = {
        "tf_next_emb_mse": tf_mse, "persistence_mse": persist,
        "skill_vs_persistence": persist / max(tf_mse, EPS),
        "target_emb_energy_per_dim": tgt_energy, "relative_mse": tf_mse / tgt_energy,
        "rollout_cost_true": cost_true, "rollout_cost_shuffled": cost_shuf,
        "shuf_over_true": cost_shuf / max(cost_true, EPS),
    }
    passed = (fidelity["skill_vs_persistence"] > 3.0 and fidelity["shuf_over_true"] > 2.0
              and fidelity["relative_mse"] < 0.2)
    fidelity["passed"] = bool(passed)
    print(f"[P5:{args.env}] fidelity: TF MSE={tf_mse:.5f} (rel {fidelity['relative_mse']:.3f}, "
          f"{fidelity['skill_vs_persistence']:.1f}x persistence); rollout true={cost_true:.3f} "
          f"shuf={cost_shuf:.3f} ({fidelity['shuf_over_true']:.1f}x) -> "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        print("[P5] FIDELITY FAILED -- not trusting replication numbers. Stopping.")
        return {"env": args.env, "fidelity": fidelity, "failed": True}

    # ---- (1) C: D_l ----
    Dfull, Dfinal = measure_C(model, emb, act_emb, act_full, act_final)

    # ---- (2) E cumulative + (3) per-branch ----
    base_mse = tf_mse
    cum = np.zeros(nb)
    for l in range(nb):
        pred, _ = tf_run(model, emb, act_emb, lambda a, l=l: a.mean_ablate(list(range(l, nb))), capture=False)
        cum[l] = float((pred[:, READOUT] - tf_tgt).pow(2).mean()) - base_mse
    p3 = {b: np.zeros(nb) for b in ("mlp", "attn", "full")}
    for l in range(nb):
        for b in ("mlp", "attn", "full"):
            pred, _ = tf_run(model, emb, act_emb, lambda a, l=l, b=b: a.branch_ablate(l, b), capture=False)
            p3[b][l] = float((pred[:, READOUT] - tf_tgt).pow(2).mean()) - base_mse

    adir, _ = out_dirs(args.env)
    paths.ensure(adir)
    np.savez(adir / "p5_arrays.npz", nb=nb, n=N, action_dim=action_dim, frameskip=frameskip,
             D_full=np.array(Dfull), D_final=np.array(Dfinal), cum=cum,
             p3_mlp=p3["mlp"], p3_attn=p3["attn"], p3_full=p3["full"],
             **{f"fid_{k}": v for k, v in fidelity.items()})
    return _load_cache(args.env)


def _load_cache(env):
    adir, _ = out_dirs(env)
    d = np.load(adir / "p5_arrays.npz")
    fid = {k[4:]: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files if k.startswith("fid_")}
    return {"env": env, "nb": int(d["nb"]), "n": int(d["n"]),
            "action_dim": int(d["action_dim"]), "frameskip": int(d["frameskip"]),
            "D_full": d["D_full"], "D_final": d["D_final"], "cum": d["cum"],
            "p3": {"mlp": d["p3_mlp"], "attn": d["p3_attn"], "full": d["p3_full"]},
            "fidelity": fid, "failed": False}


def summarize(R, pusht_k=2, thresh=0.5):
    nb = R["nb"]
    cum = R["cum"]
    Df = R["D_full"]
    full = cum[0] if cum[0] > 0 else max(cum.max(), 1e-9)
    commit = 0
    for l in range(nb):
        if cum[l] >= thresh * full:
            commit = l
    # D_l monotonic + roughly linear?
    incr = np.diff(np.concatenate([[0.0], Df]))
    monotonic = bool(np.all(incr > -0.01))
    mlp_share = float(np.mean(R["p3"]["mlp"] / np.clip(R["p3"]["mlp"] + R["p3"]["attn"], 1e-9, None)))
    block5_idle = bool(cum[-1] < 0.15 * full)
    replicates = bool(monotonic and abs(commit - pusht_k) <= 1 and mlp_share > 0.7)
    return {
        "env": R["env"], "n_clips": R["n"], "action_dim": R["action_dim"], "frameskip": R["frameskip"],
        "fidelity": R["fidelity"],
        "C_D_l_full": Df.tolist(), "C_D_l_final": R["D_final"].tolist(),
        "C_monotonic": monotonic,
        "E_cumulative_mse_increase": cum.tolist(),
        "E_commitment_depth": int(commit), "E_block5_near_idle": block5_idle,
        "E_mlp_share_of_branch_sum": mlp_share,
        "E_per_branch": {b: R["p3"][b].tolist() for b in R["p3"]},
        "pusht_k": pusht_k,
        "replicates": replicates,
        "verdict": ("REPLICATES: same row (early graded commitment, MLP-routed), commitment depth "
                    f"{commit} within +-1 of PushT k={pusht_k}, MLP share {mlp_share:.2f}>0.7."
                    if replicates else
                    f"DIVERGES: commitment depth {commit} (PushT {pusht_k}), MLP share {mlp_share:.2f}, "
                    f"D_l monotonic={monotonic}. Report the divergence plainly."),
    }


def make_plot(s, nb, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = np.arange(nb)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    xx = np.arange(-1, nb)
    ax[0].plot(xx, np.concatenate([[0.0], s["C_D_l_full"]]), "-o", color="#9467bd", lw=2.3, label="full-swap")
    ax[0].plot(xx, np.concatenate([[0.0], s["C_D_l_final"]]), "-s", color="#2ca02c", lw=2.3, label="final-swap")
    ax[0].set_xlabel("after block (−1=input)"); ax[0].set_ylabel("D_l at readout")
    ax[0].set_title(f"(1) C: D_l propagation [{s['env']}]"); ax[0].set_xticks(xx); ax[0].legend(fontsize=8)

    ax[1].plot(depth, s["E_cumulative_mse_increase"], "-o", color="#1f77b4", lw=2.3)
    ax[1].axvline(s["E_commitment_depth"], color="k", ls="--", lw=1.2,
                  label=f"commitment depth {s['E_commitment_depth']}")
    ax[1].set_xlabel("ablate blocks ≥ l"); ax[1].set_ylabel("TF MSE increase")
    ax[1].set_title("(2) E cumulative ablation"); ax[1].set_xticks(depth); ax[1].legend(fontsize=8)

    w = 0.38
    ax[2].bar(depth - w / 2, s["E_per_branch"]["mlp"], w, color="#d62728", label="MLP chunks")
    ax[2].bar(depth + w / 2, s["E_per_branch"]["attn"], w, color="#1f77b4", label="attn chunks")
    ax[2].set_xlabel("ablated block"); ax[2].set_ylabel("TF MSE increase")
    ax[2].set_title(f"(3) per-branch (MLP share {s['E_mlp_share_of_branch_sum']:.2f})")
    ax[2].set_xticks(depth); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def report(s):
    print("\n========= PHASE 5 REPLICATION (%s) =========" % s["env"])
    f = s["fidelity"]
    print(f"clips={s['n_clips']} action_dim={s['action_dim']} frameskip={s['frameskip']}")
    print(f"FIDELITY: TF MSE rel={f['relative_mse']:.3f} ({f['skill_vs_persistence']:.1f}x persistence); "
          f"rollout shuf/true={f['shuf_over_true']:.1f}x -> {'PASS' if f['passed'] else 'FAIL'}")
    print(f"(1) C D_l full : {[round(x,3) for x in s['C_D_l_full']]}  monotonic={s['C_monotonic']}")
    print(f"    C D_l final: {[round(x,3) for x in s['C_D_l_final']]}")
    print(f"(2) E cumulative: {[round(x,4) for x in s['E_cumulative_mse_increase']]}")
    print(f"    commitment depth={s['E_commitment_depth']} (PushT k={s['pusht_k']})  block5 near-idle={s['E_block5_near_idle']}")
    print(f"(3) per-branch MLP share of branch sum = {s['E_mlp_share_of_branch_sum']:.2f}")
    print(f"\nVERDICT: {s['verdict']}")
    print("=============================================\n")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--env", default="reacher")
    pa.add_argument("--n", type=int, default=1000)
    pa.add_argument("--batch-size", type=int, default=256)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:1")
    pa.add_argument("--from-cache", action="store_true")
    args = pa.parse_args()
    R = _load_cache(args.env) if args.from_cache else compute(args)
    if R.get("failed"):
        print("Fidelity failed; see fidelity dict.")
        print(json.dumps(R["fidelity"], indent=2))
        return
    summary = summarize(R)
    _, rdir = out_dirs(args.env)
    paths.ensure(rdir)
    (rdir / "phase5_summary.json").write_text(json.dumps(summary, indent=2))
    make_plot(summary, R["nb"], rdir / "phase5.png")
    report(summary)


if __name__ == "__main__":
    main()
