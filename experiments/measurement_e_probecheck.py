"""Measurement E robustness check -- retrained-probe test on the single-dissociation claim.

E's consequence-drop applied the FROZEN Measurement-D probe (fit on unablated activations)
to ablated activations. Confound: under ablation the linear consequence subspace can shift,
so a frozen probe shows a drop even if the consequence is still linearly recoverable in the
ablated activations (the frozen directions are just wrong). That would falsely read as
"late blocks carry, do not construct."

Test: for each block (focus b3..b5), retrain a fresh ridge consequence probe (Delta-emb and
phys-state-Delta) on the ABLATED-distribution activations at that block and compare its val
R^2 to (a) the frozen probe on the same ablated activations and (b) the unablated baseline.

  * retrained ~= frozen, both << baseline  -> consequence genuinely lost -> carry-not-construct HOLDS.
  * retrained jumps back toward baseline    -> frozen missed recoverable consequence -> SOFTEN.

This stays inside the E analysis. eval()+fp32, seeded. Ablated activations are extracted and
cached here (the first E pass saved only scalar curves).

    uv run python -m experiments.measurement_e_probecheck
    uv run python -m experiments.measurement_e_probecheck --from-cache
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
ALPHAS = np.logspace(-2, 5, 15)
TARGETS = ("demb", "dstate")

ACT_DIR = paths.ACTIVATIONS / "measurement_e"
RES_DIR = paths.RESULTS / "measurement_e"
D_CACHE = paths.ACTIVATIONS / "measurement_d" / "probe_cache.npz"
CACHE = ACT_DIR / "probecheck_cache.npz"
SUMMARY_PATH = RES_DIR / "probe_check_summary.json"
PLOT_PATH = RES_DIR / "probe_check.png"


@torch.inference_mode()
def encode_all(model, batch, dev, bs):
    pixels, action, state = batch["pixels"], batch["action"], batch["state"]
    embs, acts = [], []
    for i in range(0, pixels.size(0), bs):
        sl = slice(i, i + bs)
        embs.append(model.encode({"pixels": pixels[sl].to(dev)})["emb"].float())
        acts.append(model.action_encoder(action[sl, :HISTORY_SIZE].to(dev)).float())
    emb = torch.cat(embs, 0)
    act_emb = torch.cat(acts, 0)
    tgt = {
        "demb": (emb[:, HISTORY_SIZE] - emb[:, READOUT]).cpu().numpy().astype(np.float32),
        "dstate": (state[:, HISTORY_SIZE] - state[:, READOUT]).numpy().astype(np.float32),
    }
    return emb, act_emb, tgt


@torch.inference_mode()
def residuals(model, emb, act_emb, arm_fn):
    """Per-block readout residual (nb, N, 192) under an optional ablation arm_fn."""
    cap = BlockCapture(model)
    abl = AdaLNAblator(model)
    if arm_fn:
        arm_fn(abl)
    try:
        with cap:
            model.predict(emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE])
            snap = cap.snapshot()
    finally:
        abl.clear()
    return [snap[i]["x_out"][:, READOUT].cpu().numpy().astype(np.float32) for i in range(len(snap))]


def build_cache(args):
    set_seed(args.seed)
    dev = args.device
    model, cfg = build_lewm(paths.LEWM_PUSHT_CONFIG, paths.LEWM_PUSHT_WEIGHTS, device=dev, dtype=torch.float32)
    assert not model.training
    nb = len(model.predictor.transformer.layers)
    probe = D.build_dataset(paths.PUSHT_H5, num_steps=1, frameskip=1, normalize=False)
    frameskip = cfg["action_encoder"]["input_dim"] // probe.get_dim("action")
    am, asd = D.compute_action_stats(paths.PUSHT_H5)
    ds = D.build_dataset(paths.PUSHT_H5, num_steps=HISTORY_SIZE + 1, frameskip=frameskip,
                         action_mean=am, action_std=asd)
    tr, va = D.split_indices(len(ds), seed=3072, val_frac=0.1)
    rng = np.random.default_rng(args.seed)
    sel_tr = np.sort(rng.choice(tr, size=min(args.n_train, len(tr)), replace=False))
    sel_va = np.sort(rng.choice(va, size=min(args.n_val, len(va)), replace=False))
    print(f"[E-probecheck] {len(sel_tr)} train / {len(sel_va)} val clips")
    btr, bva = D.load_batch(ds, sel_tr), D.load_batch(ds, sel_va)
    emb_tr, ace_tr, ytr = encode_all(model, btr, dev, args.batch_size)
    emb_va, ace_va, yva = encode_all(model, bva, dev, args.batch_size)

    # unablated (baseline) residuals + per-block single-ablation residuals
    base_tr = residuals(model, emb_tr, ace_tr, None)
    base_va = residuals(model, emb_va, ace_va, None)
    abl_tr = [residuals(model, emb_tr, ace_tr, (lambda a, l=l: a.mean_ablate([l])))[l] for l in range(nb)]
    abl_va = [residuals(model, emb_va, ace_va, (lambda a, l=l: a.mean_ablate([l])))[l] for l in range(nb)]

    paths.ensure(ACT_DIR)
    np.savez(
        CACHE, nb=nb,
        ytr_demb=ytr["demb"], ytr_dstate=ytr["dstate"], yva_demb=yva["demb"], yva_dstate=yva["dstate"],
        **{f"base_tr_{i}": base_tr[i] for i in range(nb)},
        **{f"base_va_{i}": base_va[i] for i in range(nb)},
        **{f"abl_tr_{l}": abl_tr[l] for l in range(nb)},
        **{f"abl_va_{l}": abl_va[l] for l in range(nb)},
    )
    return _load_cache()


def _load_cache():
    d = np.load(CACHE)
    nb = int(d["nb"])
    return {
        "nb": nb,
        "ytr": {t: d[f"ytr_{t}"] for t in TARGETS},
        "yva": {t: d[f"yva_{t}"] for t in TARGETS},
        "base_tr": [d[f"base_tr_{i}"] for i in range(nb)],
        "base_va": [d[f"base_va_{i}"] for i in range(nb)],
        "abl_tr": [d[f"abl_tr_{l}"] for l in range(nb)],
        "abl_va": [d[f"abl_va_{l}"] for l in range(nb)],
    }


def _frozen_probe(depth, target):
    """Reproduce the exact cached Measurement-D probe (scaler + RidgeCV) for a depth/target."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    d = np.load(D_CACHE)
    sc = StandardScaler().fit(d[f"ftr_{depth}"])
    m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(d[f"ftr_{depth}"]), d[f"ttr_{target}"])
    return sc, m


def _fit_fresh(Xtr, Ytr):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(Xtr), Ytr)
    return sc, m


def _r2(sc, m, X, Y):
    from sklearn.metrics import r2_score

    return float(r2_score(Y, m.predict(sc.transform(X)), multioutput="uniform_average"))


def analyze(R):
    nb = R["nb"]
    out = {t: {"baseline": [], "frozen_ablated": [], "retrained_ablated": [], "recovery_frac": []} for t in TARGETS}
    for t in TARGETS:
        for l in range(nb):
            depth = l + 1
            sc_f, m_f = _frozen_probe(depth, t)
            base = _r2(sc_f, m_f, R["base_va"][l], R["yva"][t])
            froz = _r2(sc_f, m_f, R["abl_va"][l], R["yva"][t])
            sc_r, m_r = _fit_fresh(R["abl_tr"][l], R["ytr"][t])
            retr = _r2(sc_r, m_r, R["abl_va"][l], R["yva"][t])
            rec = (retr - froz) / max(base - froz, 1e-6)
            out[t]["baseline"].append(base)
            out[t]["frozen_ablated"].append(froz)
            out[t]["retrained_ablated"].append(retr)
            out[t]["recovery_frac"].append(float(np.clip(rec, -1, 2)))

    late = slice(3, nb)  # b3..b5
    late_rec_demb = float(np.mean(out["demb"]["recovery_frac"][late]))
    if late_rec_demb < 0.33:
        verdict = ("late-block recovery is small: a retrained probe does not beat the frozen probe on "
                   "ablated activations, both stay well below baseline. The consequence is genuinely lost "
                   "under ablation -> single-dissociation / carry-not-construct HOLDS; keep E wording.")
        fired = "hold"
    elif late_rec_demb > 0.5:
        verdict = ("late-block recovery is large: a retrained probe recovers most of the frozen drop, so "
                   "the frozen-probe drop was largely distribution shift, not consequence loss. SOFTEN the "
                   "claim to 'late blocks add mostly linear/nonlinear consequence refinement'.")
        fired = "soften"
    else:
        verdict = ("late-block recovery is partial: frozen-probe drop is part real loss, part distribution "
                   "shift. Soften wording slightly to 'late blocks mostly carry, with some recoverable "
                   "refinement'.")
        fired = "partial"
    return {
        "n_train": int(R["abl_tr"][0].shape[0]), "n_val": int(R["abl_va"][0].shape[0]),
        "by_target": out,
        "late_recovery_frac_demb_mean": late_rec_demb,
        "branch_fired": fired,
        "verdict": verdict,
    }


def make_plot(s, nb, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = np.arange(nb)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, t in enumerate(TARGETS):
        o = s["by_target"][t]
        ax[j].plot(depth, o["baseline"], "-o", color="#2ca02c", lw=2.2, label="baseline (unablated)")
        ax[j].plot(depth, o["frozen_ablated"], "-s", color="#d62728", lw=2.2, label="frozen probe on ablated")
        ax[j].plot(depth, o["retrained_ablated"], "-^", color="#1f77b4", lw=2.2, label="retrained on ablated")
        ax[j].set_xlabel("ablated block (depth)"); ax[j].set_ylabel("val R²")
        ax[j].set_title(f"consequence probe: {t}")
        ax[j].set_xticks(depth); ax[j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def report(s):
    print("\n========= E ROBUSTNESS: retrained vs frozen consequence probe =========")
    print(f"train={s['n_train']} val={s['n_val']}")
    for t in TARGETS:
        o = s["by_target"][t]
        print(f"\n[{t}]  block:        " + " ".join(f"{i:>7d}" for i in range(s_nb(s))))
        print("  baseline (unabl)     " + " ".join(f"{v:7.3f}" for v in o["baseline"]))
        print("  frozen on ablated    " + " ".join(f"{v:7.3f}" for v in o["frozen_ablated"]))
        print("  retrained on ablated " + " ".join(f"{v:7.3f}" for v in o["retrained_ablated"]))
        print("  recovery fraction    " + " ".join(f"{v:7.2f}" for v in o["recovery_frac"]))
    print(f"\nlate-block (b3..b5) mean recovery (Δemb) = {s['late_recovery_frac_demb_mean']:.2f}  -> {s['branch_fired'].upper()}")
    print(f"{s['verdict']}")
    print("=======================================================================\n")


def s_nb(s):
    return len(s["by_target"]["demb"]["baseline"])


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--n-train", type=int, default=3000)
    pa.add_argument("--n-val", type=int, default=1000)
    pa.add_argument("--batch-size", type=int, default=256)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:1")
    pa.add_argument("--from-cache", action="store_true")
    args = pa.parse_args()
    R = _load_cache() if args.from_cache else build_cache(args)
    summary = analyze(R)
    paths.ensure(RES_DIR)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    make_plot(summary, R["nb"], PLOT_PATH)
    report(summary)


if __name__ == "__main__":
    main()
