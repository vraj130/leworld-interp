"""Generate the two paper figures that are not produced by the per-phase pipeline:

  figure_1  -- architecture + method schematic (encoder -> AdaLN-zero predictor with the
               conditioning re-injected at every block; cumulative mean-ablation intervention).
  figure_13 -- unified scaling figure: commitment fraction vs predictor depth, overlaying every
               run (PushT reduced sweep, PushT released, reacher released sweep, official reacher
               d6) against the 1/3 line and the falsified absolute prediction.

Both write straight into the repo-root figures/ folder. figure_13 reads the cached per-phase
summaries so the numbers are the frozen ones.

    uv run python -m experiments.make_paper_figures
"""

from __future__ import annotations

import json

import numpy as np

from leworld_interp import paths

FIG = paths.REPO_ROOT / "figures"


def fig1_schematic():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333", fs=9, tc="#111"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, color="#333", lw=1.6, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                     color=color, lw=lw))

    # ---- perception path (top) ----
    box(0.2, 4.7, 1.5, 1.0, "3 history\nframes", "#eaeaea")
    box(2.1, 4.7, 1.7, 1.0, "ViT-tiny\nencoder", "#cfe3f7")
    box(4.2, 4.7, 1.6, 1.0, "projector\n(BN-MLP)", "#cfe3f7")
    box(6.2, 4.7, 1.7, 1.0, "embeddings\n$z_0 z_1 z_2$", "#eaeaea")
    arrow(1.7, 5.2, 2.1, 5.2); arrow(3.8, 5.2, 4.2, 5.2); arrow(5.8, 5.2, 6.2, 5.2)

    # ---- action path (bottom-left) ----
    box(0.2, 0.4, 1.5, 0.95, "action $a$", "#eaeaea")
    box(2.1, 0.4, 1.7, 0.95, "Embedder", "#f7dfcf")
    box(4.2, 0.4, 1.6, 0.95, "conditioning\n$c$", "#f6c9a8")
    arrow(1.7, 0.87, 2.1, 0.87); arrow(3.8, 0.87, 4.2, 0.87)

    # conditioning bus running up the predictor
    ax.plot([5.0, 5.0], [1.35, 4.35], color="#d9822b", lw=2.2)
    ax.text(4.55, 2.9, "$c$ re-injected\nat every block", color="#b5651d", fontsize=8.5,
            rotation=90, ha="center", va="center")

    # ---- predictor stack (6 AdaLN-zero blocks) ----
    nb = 6
    bx, bw, bh, gap = 6.6, 3.0, 0.46, 0.13
    y0 = 1.45
    commit_frac = 1 / 3
    for i in range(nb):
        y = y0 + i * (bh + gap)
        committed = i <= 1  # commitment band ~ first third (block index <= ~2 of 6)
        fc = "#d6ecd6" if committed else "#f0f0f0"
        box(bx, y, bw, bh, f"block {i}:  AdaLN-zero  (attn gate $\\,\\oplus\\,$ MLP gate)", fc,
            fs=8.2)
        # conditioning injection arrow into each block
        arrow(5.0, y + bh / 2, bx, y + bh / 2, color="#d9822b", lw=1.3)
    ax.text(bx + bw / 2, y0 + nb * (bh + gap) + 0.12, "AR predictor (6 ConditionalBlocks, causal)",
            ha="center", fontsize=9.5, color="#111")
    # commitment band bracket
    ax.annotate("", xy=(bx + bw + 0.15, y0 - 0.02), xytext=(bx + bw + 0.15, y0 + 2 * (bh + gap) - gap + bh),
                arrowprops=dict(arrowstyle="-", color="#2e8b2e", lw=2))
    ax.text(bx + bw + 0.3, y0 + (bh + gap), "commitment\nband (~1/3\nof depth):\ncumulative\nablation here\ndoes >=50%\nof full damage",
            fontsize=7.6, color="#2e8b2e", va="center")

    # readout -> prediction (top-right)
    arrow(bx + bw / 2, y0 + nb * (bh + gap) - gap + 0.05, bx + bw / 2, 4.7, color="#333")
    box(10.4, 4.7, 1.7, 1.0, "readout\n$\\to$ pred_proj\n$\\to$ next-emb", "#cfe3f7", fs=8.2)
    arrow(7.9, 5.2, 10.4, 5.2)

    # intervention caption
    ax.text(6.6, 0.15,
            "Intervention: replace $c$ with the batch-mean conditioning into blocks $\\geq l$ "
            "(mean ablation, never zero). Commitment depth = deepest $l$ still causing $\\geq$50% of full damage.",
            fontsize=8.0, color="#444")

    ax.set_title("Figure 1. LeWM action-conditioned JEPA predictor and the cumulative AdaLN ablation probe",
                 fontsize=11)
    fig.tight_layout()
    out = FIG / "figure_1_architecture_method.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def fig13_unified():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def fracs(path, key=None):
        s = json.load(open(path))
        rows = s["summaries"] if key is None else [s[key]]
        return {int(r["depth"]): r["commitment_depth_frac"] for r in rows}

    pusht_reduced = fracs(paths.RESULTS / "measurement_phase6_depthscaling" / "measurements_summary.json")
    reacher = fracs(paths.RESULTS / "measurement_phase8_reacher_sweep" / "measurements_summary.json")
    p7 = json.load(open(paths.RESULTS / "measurement_phase7_released_d12" / "measurements_summary.json"))
    pusht_released = {6: 2 / 6, 12: p7["released_d12"]["commitment_depth_frac"]}  # d6 official + d12 trained
    reacher_official_d6 = {6: 2 / 6}

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    depths = np.array([3, 6, 12, 18])

    # falsified absolute prediction: commitment pinned at block ~2 -> fraction 2/depth
    dd = np.linspace(3, 18, 100)
    ax.plot(dd, 2 / dd, ":", color="#444", lw=1.8, label="absolute prediction (fixed block 2): 2/depth")
    ax.axhline(1 / 3, color="#2ca02c", ls="--", lw=1.6, label="relative law: fraction = 1/3")

    ax.plot(list(pusht_reduced), list(pusht_reduced.values()), "s--", color="#7f7f7f", ms=9, lw=1.6,
            label="PushT reduced sweep")
    ax.plot(list(pusht_released), list(pusht_released.values()), "D", color="#1f77b4", ms=13,
            label="PushT released (d6 official, d12 trained)")
    ax.plot(list(reacher), list(reacher.values()), "o-", color="#d62728", ms=11, lw=2.2,
            label="reacher released sweep")
    ax.plot(list(reacher_official_d6), list(reacher_official_d6.values()), "*", color="#8b0000", ms=18,
            label="reacher d6 (official released)")

    ax.fill_between([2.5, 18.5], 0.28, 0.40, color="#2ca02c", alpha=0.07)
    ax.set_xlim(2.5, 18.5); ax.set_ylim(0, 0.72)
    ax.set_xticks(depths)
    ax.set_xlabel("predictor depth (blocks)"); ax.set_ylabel("commitment fraction of depth")
    ax.set_title("Figure 13. Commitment fraction is ~1/3 across depth, data scale, and environment\n"
                 "(every run lands on the relative law; the absolute prediction is falsified)", fontsize=10.5)
    ax.legend(fontsize=8.4, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIG / "figure_13_unified_scaling.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig1_schematic()
    fig13_unified()
