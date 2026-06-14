"""Phase 3, Measurement D -- layerwise ridge probes (action vs consequence).

At each predictor block output (plus the pre-block-0 input as an anchor), at the
readout token, we linearly decode two things from the residual stream:

  * raw action   -- the 10-dim conditioning the model sees at the readout (action[2],
                    = frameskip * action_dim, read from the data)
  * consequence  -- the predictor's own target, the next-state embedding delta
                    emb[3]-emb[2] (192-dim); plus the physical state delta (7-dim)

D lives in decodable-content space (C lived in propagation space). The AEZ signature
is a CROSSOVER: raw-action decodability flat/falling with depth while consequence
decodability rises; the crossover depth k is the candidate commitment depth.

The pre-block-0 input anchor is a control for PushT being EXPERT data: the expert
policy makes action correlate with state, so some action R^2 is decodable from the
state embedding before any conditioning is injected. Action decodability *above* that
baseline is the conditioning's contribution.

One model pass yields all depths (hooks), so the 6-depth sweep is CPU-bound ridge fits;
GPU parallelism is moot. Probe inputs are cached so --from-cache refits without the model.

    uv run python -m experiments.measurement_d
    uv run python -m experiments.measurement_d --from-cache
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from leworld_interp import data as D
from leworld_interp import paths
from leworld_interp.hooks import BlockCapture
from leworld_interp.model import build_lewm, set_seed

HISTORY_SIZE = 3
READOUT = HISTORY_SIZE - 1
ALPHAS = np.logspace(-2, 5, 15)

ACT_DIR = paths.ACTIVATIONS / "measurement_d"
RES_DIR = paths.RESULTS / "measurement_d"
CACHE = ACT_DIR / "probe_cache.npz"
PROBE_W = ACT_DIR / "probe_weights.npz"
SUMMARY_PATH = RES_DIR / "measurement_d_summary.json"
PLOT_PATH = RES_DIR / "measurement_d.png"


@torch.inference_mode()
def extract(model, batch, device, bs):
    """One forward sweep -> residual features at every depth (readout token) + targets."""
    nb = len(model.predictor.transformer.layers)
    feats = [[] for _ in range(nb + 1)]  # 0 = input anchor, 1..nb = after block i-1
    t_action, t_demb, t_dstate = [], [], []
    pixels, action, state = batch["pixels"], batch["action"], batch["state"]
    r = READOUT
    cap = BlockCapture(model)
    with cap:
        for i in range(0, pixels.size(0), bs):
            sl = slice(i, i + bs)
            emb = model.encode({"pixels": pixels[sl].to(device)})["emb"].float()  # (b,4,192)
            c = model.action_encoder(action[sl, :HISTORY_SIZE].to(device)).float()  # (b,3,192)
            model.predictor(emb[:, :HISTORY_SIZE], c)
            snap = cap.snapshot()
            feats[0].append(snap[0]["x_in"][:, r].cpu())
            for bi in range(nb):
                feats[bi + 1].append(snap[bi]["x_out"][:, r].cpu())
            t_action.append(action[sl, r].clone())                       # (b,10) z-scored
            t_demb.append((emb[:, HISTORY_SIZE] - emb[:, r]).cpu())       # (b,192)
            t_dstate.append((state[sl, HISTORY_SIZE] - state[sl, r]).clone())  # (b,7)
    feats = [torch.cat(f).numpy().astype(np.float32) for f in feats]
    tgt = {
        "action": torch.cat(t_action).numpy().astype(np.float32),
        "demb": torch.cat(t_demb).numpy().astype(np.float32),
        "dstate": torch.cat(t_dstate).numpy().astype(np.float32),
    }
    return feats, tgt


def build_cache(args):
    set_seed(args.seed)
    device = args.device
    model, cfg = build_lewm(paths.LEWM_PUSHT_CONFIG, paths.LEWM_PUSHT_WEIGHTS,
                            device=device, dtype=torch.float32)
    assert not model.training
    probe = D.build_dataset(paths.PUSHT_H5, num_steps=1, frameskip=1, normalize=False)
    action_dim = probe.get_dim("action")
    frameskip = cfg["action_encoder"]["input_dim"] // action_dim
    am, asd = D.compute_action_stats(paths.PUSHT_H5)
    ds = D.build_dataset(paths.PUSHT_H5, num_steps=HISTORY_SIZE + 1, frameskip=frameskip,
                         action_mean=am, action_std=asd)
    tr, va = D.split_indices(len(ds), seed=3072, val_frac=0.1)
    rng = np.random.default_rng(args.seed)
    sel_tr = np.sort(rng.choice(tr, size=min(args.n_train, len(tr)), replace=False))
    sel_va = np.sort(rng.choice(va, size=min(args.n_val, len(va)), replace=False))
    print(f"[D] extracting features: {len(sel_tr)} train / {len(sel_va)} val clips "
          f"(action_dim={action_dim}, frameskip={frameskip})")
    btr = D.load_batch(ds, sel_tr)
    bva = D.load_batch(ds, sel_va)
    ftr, ttr = extract(model, btr, device, args.batch_size)
    fva, tva = extract(model, bva, device, args.batch_size)
    paths.ensure(ACT_DIR)
    np.savez(
        CACHE, nb=len(ftr) - 1, action_dim=action_dim, frameskip=frameskip,
        **{f"ftr_{i}": ftr[i] for i in range(len(ftr))},
        **{f"fva_{i}": fva[i] for i in range(len(fva))},
        ttr_action=ttr["action"], ttr_demb=ttr["demb"], ttr_dstate=ttr["dstate"],
        tva_action=tva["action"], tva_demb=tva["demb"], tva_dstate=tva["dstate"],
    )
    return _pack(ftr, fva, ttr, tva, len(ftr) - 1)


def _pack(ftr, fva, ttr, tva, nb):
    return dict(nb=nb, ftr=ftr, fva=fva, ttr=ttr, tva=tva)


def _load_cache():
    d = np.load(CACHE)
    nb = int(d["nb"])
    ftr = [d[f"ftr_{i}"] for i in range(nb + 1)]
    fva = [d[f"fva_{i}"] for i in range(nb + 1)]
    ttr = {k: d[f"ttr_{k}"] for k in ("action", "demb", "dstate")}
    tva = {k: d[f"tva_{k}"] for k in ("action", "demb", "dstate")}
    return _pack(ftr, fva, ttr, tva, nb)


def fit_probes(R):
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    nb = R["nb"]
    depths = nb + 1  # input + nb blocks
    targets = ("action", "demb", "dstate")
    r2 = {t: np.zeros(depths) for t in targets}
    r2_var = {t: np.zeros(depths) for t in targets}
    weights = {}
    for di in range(depths):
        Xtr, Xva = R["ftr"][di], R["fva"][di]
        sc = StandardScaler().fit(Xtr)
        Xtr2, Xva2 = sc.transform(Xtr), sc.transform(Xva)
        for t in targets:
            Ytr, Yva = R["ttr"][t], R["tva"][t]
            mdl = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Xtr2, Ytr)
            pred = mdl.predict(Xva2)
            r2[t][di] = float(r2_score(Yva, pred, multioutput="uniform_average"))
            r2_var[t][di] = float(r2_score(Yva, pred, multioutput="variance_weighted"))
            weights[f"coef_d{di}_{t}"] = mdl.coef_.astype(np.float32)
    paths.ensure(ACT_DIR)
    np.savez(PROBE_W, **weights, **{f"r2_{t}": r2[t] for t in targets})
    return r2, r2_var


def analyze(r2, nb):
    a = r2["action"]      # depth 0 = input anchor, 1..nb = block outputs
    cz = r2["demb"]
    blocks = np.arange(nb)            # block output indices 0..nb-1 -> depth idx 1..nb
    ab = a[1:]                        # action R2 at block outputs
    cb = cz[1:]                       # consequence R2 at block outputs
    base = float(a[0])               # input anchor (expert state->action leakage)

    # crossover among block outputs: first block where consequence overtakes action
    cross = None
    for i in range(nb):
        if cb[i] > ab[i] and (i == 0 or cb[i - 1] <= ab[i - 1]):
            cross = i
            break
    action_peak = int(np.argmax(ab))
    action_late_over_early = float(ab[-1] / max(ab[0], 1e-8))
    conseq_late_over_early = float(cb[-1] / max(cb[0], 1e-8))
    action_maximal_early = bool(ab[0] >= 0.95 * ab.max())  # near-maximal at first block

    # preregistered E prediction (committed; null in parentheses)
    if cross is not None:
        prereg = (f"crossover at block k={cross}: preregister per-block ablation peaks at/just "
                  f"before k={cross}; cumulative-from-{cross} destroys performance while "
                  f"cumulative-from-{cross + 1} does not.")
        pred_id = "crossover_k"
    elif action_maximal_early and conseq_late_over_early > 1.15 and action_late_over_early < 1.05:
        prereg = ("action decodability is already ~maximal at block 0 (jumps from the input "
                  "baseline and stays flat-high) while only consequence rises with depth -> "
                  "preregister FRONT-LOADED per-block ablation (early ablation hurts most; "
                  "early commitment, decision row 2). NULL = flat (distributed, row 4).")
        pred_id = "front_loaded"
    elif abs(action_late_over_early - 1.0) < 0.12 and abs(conseq_late_over_early - 1.0) < 0.12:
        prereg = ("no crossover, both roughly flat across depth -> preregister FLAT per-block "
                  "ablation effect (genuinely distributed, row 4).")
        pred_id = "flat"
    else:
        prereg = ("no clean crossover; action stays decodable while consequence also high. "
                  "Preregister FRONT-LOADED per-block ablation as primary, flat as null.")
        pred_id = "front_loaded_or_flat"

    return {
        "input_anchor_action_r2": base,
        "action_r2_block_outputs": ab.tolist(),
        "consequence_demb_r2_block_outputs": cb.tolist(),
        "consequence_dstate_r2_block_outputs": r2["dstate"][1:].tolist(),
        "action_peak_block": action_peak,
        "action_late_over_early": action_late_over_early,
        "consequence_late_over_early": conseq_late_over_early,
        "crossover_block_k": cross,
        "preregistered_E_prediction": prereg,
        "preregistered_E_id": pred_id,
    }


def make_plot(r2, summary, nb, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    xd = np.arange(-1, nb)  # -1 = input anchor, 0..nb-1 = block outputs
    ax[0].axhline(summary["input_anchor_action_r2"], color="#888", ls=":", lw=1.2,
                  label="action R² input baseline (expert leakage)")
    ax[0].plot(xd, r2["action"], "-o", color="#1f77b4", lw=2.4, label="raw action R²")
    ax[0].plot(xd, r2["demb"], "-s", color="#d62728", lw=2.4, label="consequence (next-emb Δ) R²")
    ax[0].plot(xd, r2["dstate"], "--^", color="#2ca02c", lw=1.8, label="phys state Δ R²")
    if summary["crossover_block_k"] is not None:
        ax[0].axvline(summary["crossover_block_k"], color="k", ls="--", lw=1.2,
                      label=f"crossover k={summary['crossover_block_k']}")
    ax[0].set_xlabel("residual after block (−1 = predictor input)")
    ax[0].set_ylabel("validation R²")
    ax[0].set_title("(D) layerwise decodability: action vs consequence")
    ax[0].set_xticks(xd)
    ax[0].legend(fontsize=8)

    # gap (consequence − action) across block outputs
    blk = np.arange(nb)
    gap = np.asarray(summary["consequence_demb_r2_block_outputs"]) - np.asarray(summary["action_r2_block_outputs"])
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].plot(blk, gap, "-o", color="#9467bd", lw=2.4)
    ax[1].set_xlabel("predictor block (depth)")
    ax[1].set_ylabel("R²(consequence) − R²(action)")
    ax[1].set_title("(D) consequence-minus-action decodability gap")
    ax[1].set_xticks(blk)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(s):
    print("\n================ MEASUREMENT D (layerwise probes) ================")
    print(f"input-anchor action R² (expert state→action leakage baseline): {s['input_anchor_action_r2']:.3f}")
    print("block:               ", " ".join(f"{i:>6d}" for i in range(len(s['action_r2_block_outputs']))))
    print("action R²            ", " ".join(f"{v:6.3f}" for v in s["action_r2_block_outputs"]))
    print("consequence(Δemb) R² ", " ".join(f"{v:6.3f}" for v in s["consequence_demb_r2_block_outputs"]))
    print("phys state Δ R²      ", " ".join(f"{v:6.3f}" for v in s["consequence_dstate_r2_block_outputs"]))
    print(f"\naction peak block={s['action_peak_block']}  action late/early={s['action_late_over_early']:.2f}  "
          f"consequence late/early={s['consequence_late_over_early']:.2f}")
    print(f"crossover block k = {s['crossover_block_k']}")
    print(f"\nPREREGISTERED E PREDICTION [{s['preregistered_E_id']}]:\n  {s['preregistered_E_prediction']}")
    print("==================================================================\n")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--n-train", type=int, default=4000)
    pa.add_argument("--n-val", type=int, default=1000)
    pa.add_argument("--batch-size", type=int, default=256)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:1")
    pa.add_argument("--from-cache", action="store_true")
    args = pa.parse_args()
    R = _load_cache() if args.from_cache else build_cache(args)
    r2, r2_var = fit_probes(R)
    summary = analyze(r2, R["nb"])
    summary["r2_variance_weighted"] = {t: r2_var[t].tolist() for t in r2_var}
    paths.ensure(RES_DIR)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    make_plot(r2, summary, R["nb"], PLOT_PATH)
    report(summary)


if __name__ == "__main__":
    main()
