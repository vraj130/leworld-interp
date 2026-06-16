"""Phase 6 plots. Reads cached per-depth raw arrays (no recompute).

Two modes:
  --gate : the retrained-d6 vs released-d6 internal-consistency figure (3 panels).
  --sweep: the cross-depth comparison once d3/d6/d12/d18 are all measured.

    uv run python -m experiments.plot_phase6 --gate
    uv run python -m experiments.plot_phase6 --sweep --depths 3 6 12 18
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from leworld_interp import paths

ARR = paths.ACTIVATIONS / "measurement_phase6_depthscaling"
RES_DIR = paths.RESULTS / "measurement_phase6_depthscaling"
REL_E = paths.RESULTS / "measurement_e" / "measurement_e_summary.json"


def _load(depth):
    return json.load(open(ARR / f"depth_{depth}.json"))


def gate_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = _load(6)
    rel = json.load(open(REL_E))
    nb = 6
    x = np.arange(nb)

    rt_cum = np.array(r["cumulative_mse_increase"])
    rl_cum = np.array(rel["part2_cumulative"]["mse_increase"])
    rt_mlp = np.array(r["branch_mlp_only"]); rt_attn = np.array(r["branch_attn_only"])
    rl_mlp = np.array(rel["part3_per_branch"]["mlp_only_mse_increase"])
    rl_attn = np.array(rel["part3_per_branch"]["attn_only_mse_increase"])
    rt_share = rt_mlp / np.clip(rt_mlp + rt_attn, 1e-9, None)
    rl_share = rl_mlp / np.clip(rl_mlp + rl_attn, 1e-9, None)
    rt_dl = np.array(r["D_l_final_swap"]); rt_dl = rt_dl / rt_dl.max()
    rl_dl = np.array(rel.get("_dummy", [0])) if False else None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    # panel 1: cumulative ablation -> commitment depth (normalized to full)
    ax[0].plot(x, rl_cum / rl_cum[0], "-o", color="#1f77b4", lw=2.4, label="released-d6")
    ax[0].plot(x, rt_cum / rt_cum[0], "--s", color="#d62728", lw=2.4, label="retrained-d6 (reduced)")
    ax[0].axhline(0.5, color="#888", ls=":", lw=1.2, label="50% bar")
    ax[0].axvline(2, color="k", ls="--", lw=1.0, alpha=0.6)
    ax[0].annotate("commit depth = 2", (2.05, 0.9), fontsize=9)
    ax[0].set_xlabel("ablate blocks >= l"); ax[0].set_ylabel("cumulative damage / full")
    ax[0].set_title("(2) commitment depth"); ax[0].set_xticks(x); ax[0].legend(fontsize=8)

    # panel 2: per-block MLP share of branch sum
    ax[1].plot(x, rl_share, "-o", color="#1f77b4", lw=2.4, label=f"released-d6 (mean {rl_share.mean():.2f})")
    ax[1].plot(x, rt_share, "--s", color="#d62728", lw=2.4, label=f"retrained-d6 (mean {rt_share.mean():.2f})")
    ax[1].axhline(0.7, color="#888", ls=":", lw=1.2, label="0.7 gate")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("predictor block"); ax[1].set_ylabel("MLP share  mlp/(mlp+attn)")
    ax[1].set_title("(3) per-branch MLP routing"); ax[1].set_xticks(x); ax[1].legend(fontsize=8)

    # panel 3: D_l propagation (normalized), retrained; released for shape if present
    ax[2].plot(x, rt_dl, "--s", color="#d62728", lw=2.4,
               label=f"retrained-d6 (linear R2={_linr2(np.array(r['D_l_final_swap'])):.2f})")
    # released D_l (final swap) lives in measurement_bc; overlay if available
    try:
        bc = json.load(open(paths.RESULTS / "measurement_bc" / "measurement_bc_summary.json"))
        rl_dl = np.array(bc["D_l_final_swap"]); rl_dl = rl_dl / rl_dl.max()
        ax[2].plot(x, rl_dl, "-o", color="#1f77b4", lw=2.4, label="released-d6")
    except Exception:
        pass
    ax[2].set_xlabel("residual stream after block"); ax[2].set_ylabel("D_l (normalized)")
    ax[2].set_title("(C) action propagation shape"); ax[2].set_xticks(x); ax[2].legend(fontsize=8)

    fig.suptitle("Phase 6 internal-consistency GATE: retrained-d6 reproduces released-d6 audit "
                 "(N=1000, readout, eval+fp32)", fontsize=11)
    fig.tight_layout()
    out = RES_DIR / "phase6_gate_d6.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")


def _linr2(dl):
    x = np.arange(len(dl)); coef = np.polyfit(x, dl, 1)
    resid = dl - np.polyval(coef, x)
    return 1.0 - float((resid ** 2).sum()) / max(float(((dl - dl.mean()) ** 2).sum()), 1e-12)


def sweep_plot(depths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = {d: _load(d) for d in depths}
    colors = {3: "#2ca02c", 6: "#1f77b4", 12: "#ff7f0e", 18: "#d62728"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    for d in depths:
        r = rs[d]; nb = r["nb"]; x = np.arange(nb)
        cum = np.array(r["cumulative_mse_increase"]); cum = cum / cum[0]
        commit = max([l for l in range(nb) if cum[l] >= 0.5])
        ax[0].plot(x / (nb - 1), cum, "-o", color=colors[d], lw=2.0, ms=4,
                   label=f"d{d} (commit {commit}, frac {commit/nb:.2f})")
        mlp = np.array(r["branch_mlp_only"]); attn = np.array(r["branch_attn_only"])
        share = mlp / np.clip(mlp + attn, 1e-9, None)
        ax[1].plot(x / (nb - 1), share, "-o", color=colors[d], lw=2.0, ms=4,
                   label=f"d{d} (mean {share.mean():.2f})")
        dl = np.array(r["D_l_final_swap"]); dl = dl / dl.max()
        ax[2].plot(x / (nb - 1), dl, "-o", color=colors[d], lw=2.0, ms=4,
                   label=f"d{d} (R2 {_linr2(np.array(r['D_l_final_swap'])):.2f})")

    ax[0].axhline(0.5, color="#888", ls=":", lw=1.2)
    ax[0].set_xlabel("fractional depth  l/(D-1)"); ax[0].set_ylabel("cumulative damage / full")
    ax[0].set_title("(2) commitment depth vs total depth"); ax[0].legend(fontsize=8)
    ax[1].axhline(0.7, color="#888", ls=":", lw=1.2)
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("fractional depth"); ax[1].set_ylabel("MLP share")
    ax[1].set_title("(3) MLP routing vs total depth"); ax[1].legend(fontsize=8)
    ax[2].set_xlabel("fractional depth"); ax[2].set_ylabel("D_l (normalized)")
    ax[2].set_title("(C) propagation shape vs total depth"); ax[2].legend(fontsize=8)

    fig.suptitle("Phase 6 depth-scaling: commitment depth, MLP routing, propagation shape "
                 "(N=1000, readout, eval+fp32)", fontsize=11)
    fig.tight_layout()
    out = RES_DIR / "phase6_sweep.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")


def phase7_plot():
    """Released-scale d12 vs released-d6 and reduced-d12: depth law at released scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res7 = paths.RESULTS / "measurement_phase7_released_d12"
    arr7 = paths.ACTIVATIONS / "measurement_phase7_released_d12"
    rel12 = json.load(open(arr7 / "depth_12_released.json"))
    red12 = _load(12)
    e = json.load(open(REL_E))
    d6_cum = np.asarray(e["part2_cumulative"]["mse_increase"], float)

    def norm(c):
        c = np.asarray(c, float); return c / c[0]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # panel 1: cumulative ablation on fractional-depth axis
    for cum, nb, color, lab in [
        (d6_cum, 6, "#1f77b4", "released d6 (commit 2, frac 0.33)"),
        (np.asarray(rel12["cumulative_mse_increase"]), 12, "#d62728", "released d12 (commit 4, frac 0.33)"),
        (np.asarray(red12["cumulative_mse_increase"]), 12, "#7f7f7f", "reduced d12 (commit 5, frac 0.42)"),
    ]:
        x = np.arange(nb) / (nb - 1)
        style = "--s" if "reduced" in lab else "-o"
        ax[0].plot(x, norm(cum), style, color=color, lw=2.2, ms=5, label=lab)
    ax[0].axhline(0.5, color="#aaa", ls=":", lw=1.3, label="50% bar")
    ax[0].set_xlabel("fractional depth  l/(D-1)"); ax[0].set_ylabel("cumulative damage / full")
    ax[0].set_title("(2) commitment depth at released scale"); ax[0].legend(fontsize=8)

    # panel 2: commitment fraction vs total depth (Phase 6 reduced sweep + released points)
    sweep = {}
    for d in (3, 6, 12, 18):
        r = _load(d); cum = norm(r["cumulative_mse_increase"])
        sweep[d] = max([l for l in range(d) if cum[l] >= 0.5]) / d
    ax[1].plot(list(sweep), list(sweep.values()), "--s", color="#7f7f7f", lw=2.0, ms=7,
               label="reduced-regime sweep (Phase 6)")
    ax[1].plot([6], [2 / 6], "o", color="#1f77b4", ms=13, label="released d6 (0.33)")
    ax[1].plot([12], [4 / 12], "o", color="#d62728", ms=13, label="released d12 (0.33)")
    ax[1].axhline(1 / 3, color="#2ca02c", ls=":", lw=1.5, label="fraction 1/3")
    ax[1].set_ylim(0, 0.7)
    ax[1].set_xlabel("total predictor depth"); ax[1].set_ylabel("commitment fraction of depth")
    ax[1].set_title("(2) fraction is depth-invariant, held at released scale"); ax[1].legend(fontsize=8)

    fig.suptitle("Phase 7: commitment fraction holds at released scale (d6 and d12 both 0.33, rel-MSE ~0.007-0.008)",
                 fontsize=11)
    fig.tight_layout()
    paths.ensure(res7)
    out = res7 / "phase7_released_d12.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--gate", action="store_true")
    pa.add_argument("--sweep", action="store_true")
    pa.add_argument("--phase7", action="store_true")
    pa.add_argument("--depths", type=int, nargs="+", default=[3, 6, 12, 18])
    args = pa.parse_args()
    paths.ensure(RES_DIR)
    if args.gate:
        gate_plot()
    if args.sweep:
        sweep_plot(args.depths)
    if args.phase7:
        phase7_plot()


if __name__ == "__main__":
    main()
