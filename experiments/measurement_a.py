"""Phase 1, Measurement A -- static adaLN gate audit (no data, no forward pass).

For each of the 6 ConditionalBlocks, the final adaLN Linear is zero-initialised at
the start of training, so the learned norm of each output chunk is a direct fossil
record of how much that conditioning site grew. We split the Linear's weight/bias
into the 6 chunks (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
and report the Frobenius norm of each, at **12-site resolution**: the gate_msa and
gate_mlp chunks (one per attn / mlp injection site, x6 blocks) directly scale each
residual update and are highlighted.

Chunks that stayed near zero in late blocks are wash-out evidence found before a
single forward pass. The static norm is necessary-not-sufficient: it shows the gate
can vary with conditioning, not that downstream layers causally use it (-> C, E).

    uv run python -m experiments.measurement_a
    uv run python -m experiments.measurement_a --from-cache   # replot only
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from leworld_interp import adaln, paths
from leworld_interp.model import build_lewm, set_seed

ACT_DIR = paths.ACTIVATIONS / "measurement_a"
RES_DIR = paths.RESULTS / "measurement_a"
ARR_PATH = ACT_DIR / "adaln_chunk_norms.npz"
SUMMARY_PATH = RES_DIR / "measurement_a_summary.json"
PLOT_PATH = RES_DIR / "measurement_a.png"


def compute():
    set_seed(0)
    model, _ = build_lewm(
        paths.LEWM_PUSHT_CONFIG, paths.LEWM_PUSHT_WEIGHTS, device="cpu", dtype=torch.float32
    )
    assert not model.training
    norms = adaln.static_adaln_norms(model)
    paths.ensure(ACT_DIR)
    np.savez(
        ARR_PATH,
        w_fro=norms["w_fro"],
        b_l2=norms["b_l2"],
        b_mean=norms["b_mean"],
        chunk_names=np.array(norms["chunk_names"]),
        dim=norms["dim"],
        n_blocks=norms["n_blocks"],
    )
    return norms


def summarize(norms: dict) -> dict:
    names = norms["chunk_names"]
    w, bl2, bmean = norms["w_fro"], norms["b_l2"], norms["b_mean"]
    nb = norms["n_blocks"]
    ci = {n: i for i, n in enumerate(names)}

    # 12-site table: per block, the two gated injection sites
    sites = []
    for b in range(nb):
        for site in adaln.SITES:
            sites.append({
                "site": f"b{b}.{site['branch']}",
                "block": b,
                "branch": site["branch"],
                "gate_w_fro": float(w[b, ci[site["gate"]]]),
                "gate_b_l2": float(bl2[b, ci[site["gate"]]]),
                "gate_b_mean": float(bmean[b, ci[site["gate"]]]),
                "shift_w_fro": float(w[b, ci[site["shift"]]]),
                "scale_w_fro": float(w[b, ci[site["scale"]]]),
            })

    gm = w[:, ci["gate_msa"]]
    gp = w[:, ci["gate_mlp"]]
    gate_sum = gm + gp  # per block, combined gate weight-Fro
    total = w.sum(axis=1)  # per block, total adaLN weight-Fro budget
    gate_share = (gate_sum / total).tolist()

    early = float(gate_sum[: max(1, nb // 3)].mean())
    late = float(gate_sum[-max(1, nb // 3):].mean())

    # qualitative static read (final verdict deferred to C/E)
    late_to_early = late / early if early else float("nan")
    near_zero = float(gate_sum.min()) < 0.15 * float(gate_sum.max())
    if near_zero and float(np.argmin(gate_sum)) >= nb - 2:
        read = "late gate(s) near zero -> static early-commitment signature"
    elif late_to_early > 1.15:
        read = "gates grow into mid/late depth -> consistent with an AEZ (not early wash-out)"
    elif late_to_early < 0.85:
        read = "gates concentrated early, fade with depth -> early-commitment signature"
    else:
        read = "gate magnitude roughly flat across depth -> distributed conditioning"

    return {
        "checkpoint": str(paths.LEWM_PUSHT_WEIGHTS),
        "metric": "Frobenius norm of each 192-wide adaLN output chunk (weight); bias L2 reported too",
        "chunk_order": names,
        "n_blocks": nb,
        "w_fro_per_block_per_chunk": w.tolist(),
        "b_l2_per_block_per_chunk": bl2.tolist(),
        "sites_12": sites,
        "gate_msa_w_fro_by_depth": gm.tolist(),
        "gate_mlp_w_fro_by_depth": gp.tolist(),
        "gate_share_of_block_budget": gate_share,
        "gate_sum_early_mean": early,
        "gate_sum_late_mean": late,
        "gate_late_over_early": late_to_early,
        "static_read": read,
    }


def make_plot(norms, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(norms["chunk_names"])
    w = norms["w_fro"]
    bl2 = norms["b_l2"]
    nb = norms["n_blocks"]
    depth = np.arange(nb)
    ci = {n: i for i, n in enumerate(names)}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

    # Panel A: all 6 chunk weight-Frobenius norms; gates highlighted
    style = {
        "shift_msa": ("#bbbbbb", "--", 1.3),
        "scale_msa": ("#888888", "--", 1.3),
        "gate_msa": ("#1f77b4", "-", 2.6),
        "shift_mlp": ("#ddccbb", "--", 1.3),
        "scale_mlp": ("#bba98a", "--", 1.3),
        "gate_mlp": ("#d62728", "-", 2.6),
    }
    for n in names:
        c, ls, lw = style[n]
        ax[0].plot(depth, w[:, ci[n]], ls, color=c, lw=lw,
                   marker="o" if n in adaln.GATE_CHUNKS else None, label=n)
    ax[0].set_xlabel("predictor block (depth)")
    ax[0].set_ylabel("adaLN chunk weight  ‖·‖_F")
    ax[0].set_title("(A) per-chunk adaLN weight norm vs depth\n(gates highlighted)")
    ax[0].set_xticks(depth)
    ax[0].legend(fontsize=8, ncol=2)

    # Panel B: 12-site gate profile (weight-Fro solid, bias-L2 dashed)
    ax[1].plot(depth, w[:, ci["gate_msa"]], "-o", color="#1f77b4", lw=2.4, label="gate_msa  ‖W‖_F")
    ax[1].plot(depth, w[:, ci["gate_mlp"]], "-o", color="#d62728", lw=2.4, label="gate_mlp  ‖W‖_F")
    ax[1].plot(depth, bl2[:, ci["gate_msa"]], "--s", color="#1f77b4", lw=1.6, alpha=0.7, label="gate_msa  ‖b‖")
    ax[1].plot(depth, bl2[:, ci["gate_mlp"]], "--s", color="#d62728", lw=1.6, alpha=0.7, label="gate_mlp  ‖b‖")
    ax[1].set_xlabel("predictor block (depth)")
    ax[1].set_ylabel("gate chunk norm")
    ax[1].set_title("(B) 12-site gate profile\n(attn=gate_msa, mlp=gate_mlp, per block)")
    ax[1].set_xticks(depth)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(summary):
    print("\n================ MEASUREMENT A (static adaLN gate audit) ================")
    print("chunk order:", summary["chunk_order"])
    print("\n12-site table (gate_w_fro = action-conditional gate magnitude; b = bias):")
    print(f"  {'site':<9} {'gate ‖W‖_F':>11} {'gate ‖b‖':>9} {'gate b̄':>8} {'shift ‖W‖':>10} {'scale ‖W‖':>10}")
    for s in summary["sites_12"]:
        print(f"  {s['site']:<9} {s['gate_w_fro']:>11.4f} {s['gate_b_l2']:>9.4f} "
              f"{s['gate_b_mean']:>8.4f} {s['shift_w_fro']:>10.4f} {s['scale_w_fro']:>10.4f}")
    print("\ngate_msa ‖W‖_F by depth:", [round(x, 4) for x in summary["gate_msa_w_fro_by_depth"]])
    print("gate_mlp ‖W‖_F by depth:", [round(x, 4) for x in summary["gate_mlp_w_fro_by_depth"]])
    print("gate share of block adaLN budget:", [round(x, 3) for x in summary["gate_share_of_block_budget"]])
    print(f"gate(msa+mlp) early-mean={summary['gate_sum_early_mean']:.4f}  "
          f"late-mean={summary['gate_sum_late_mean']:.4f}  late/early={summary['gate_late_over_early']:.2f}")
    print(f"\nSTATIC READ: {summary['static_read']}")
    print("(static = necessary, not sufficient; causal verdict comes from C and E)")
    print("=========================================================================\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from-cache", action="store_true")
    args = p.parse_args()
    if args.from_cache:
        d = np.load(ARR_PATH, allow_pickle=True)
        norms = {
            "w_fro": d["w_fro"], "b_l2": d["b_l2"], "b_mean": d["b_mean"],
            "chunk_names": [str(x) for x in d["chunk_names"]],
            "dim": int(d["dim"]), "n_blocks": int(d["n_blocks"]),
        }
    else:
        norms = compute()
    summary = summarize(norms)
    paths.ensure(RES_DIR)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    make_plot(norms, PLOT_PATH)
    report(summary)


if __name__ == "__main__":
    main()
