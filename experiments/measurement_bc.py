"""Phase 2, Measurements B and C -- activation-level action conditioning.

Paired forward passes through the predictor on held-out PushT clips (eval(), fp32,
seeded), all reported at the **readout token** (last position, index 2 of a 3-frame
context) and at **12-site resolution** (attn/mlp injection site per block).

Counterfactuals share the SAME states, only the action changes:
  * full-swap   a'_full  -- within-batch permutation of the whole action history
  * final-swap  a'_final -- permute ONLY the last action token (cleanest under the
                            causal mask); uses the same permutation as full-swap for
                            that token so the two variants are comparable.

(B) per site: mean|gate|, injection ratio ||gate ⊙ branch_out|| / ||x|| (the realized
    fraction of the residual update that is action-conditioned), and the modulation
    response delta_{shift,scale,gate} = ||chunk(a) - chunk(a'_full)|| at the readout.
(C) headline curve: D_l = ||h_l(a) - h_l(a')|| / ||h_l(a)|| after each block, for both
    swap variants, with ||a-a'|| and ||c-c'|| reported so D_l is per-unit-perturbation
    interpretable.

    uv run python -m experiments.measurement_bc
    uv run python -m experiments.measurement_bc --from-cache
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from leworld_interp import data as D
from leworld_interp import paths
from leworld_interp.adaln import CHUNK_NAMES, GATE_CHUNKS
from leworld_interp.adaln import split_chunks
from leworld_interp.hooks import BlockCapture
from leworld_interp.model import build_lewm, set_seed

HISTORY_SIZE = 3
READOUT = HISTORY_SIZE - 1  # last position (causal)
EPS = 1e-8

ACT_DIR = paths.ACTIVATIONS / "measurement_bc"
RES_DIR = paths.RESULTS / "measurement_bc"
ARR_PATH = ACT_DIR / "bc_arrays.npz"
SUMMARY_PATH = RES_DIR / "measurement_bc_summary.json"
PLOT_PATH = RES_DIR / "measurement_bc.png"


@torch.inference_mode()
def _encode_emb(model, pixels, device):
    return model.encode({"pixels": pixels.to(device)})["emb"].float()


@torch.inference_mode()
def _act_emb(model, action, device):
    return model.action_encoder(action.to(device)).float()


def _vstack(d):
    return {k: np.concatenate(v, 0) for k, v in d.items()}


def compute(args):
    set_seed(args.seed)
    device = args.device
    model, cfg = build_lewm(paths.LEWM_PUSHT_CONFIG, paths.LEWM_PUSHT_WEIGHTS,
                            device=device, dtype=torch.float32)
    assert not model.training

    probe = D.build_dataset(paths.PUSHT_H5, num_steps=1, frameskip=1, normalize=False)
    action_dim = probe.get_dim("action")
    frameskip = cfg["action_encoder"]["input_dim"] // action_dim
    action_mean, action_std = D.compute_action_stats(paths.PUSHT_H5)

    ds = D.build_dataset(paths.PUSHT_H5, num_steps=HISTORY_SIZE, frameskip=frameskip,
                         action_mean=action_mean, action_std=action_std)
    _, val = D.split_indices(len(ds), seed=3072, val_frac=0.1)
    sel = np.random.default_rng(args.seed).choice(val, size=min(args.n_clips, len(val)), replace=False)
    batch = D.load_batch(ds, sel)
    n_ep = len(np.unique(batch["episode_idx"].numpy()))
    print(f"[BC] {len(sel)} val clips spanning {n_ep} episodes; readout token = position {READOUT}")

    pixels = batch["pixels"]               # (N,3,C,H,W)
    a = batch["action"]                    # (N,3,10) z-scored
    N = a.size(0)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed + 7))
    a_full = a[perm]
    a_final = a.clone()
    a_final[:, READOUT] = a[perm][:, READOUT]   # swap only the last action token

    nb = len(model.predictor.transformer.layers)
    # per-sample accumulators
    _Bkeys = ("inj_attn", "inj_mlp", "absgate_msa", "absgate_mlp",
              "upd_attn", "upd_mlp", "xin_norm", "xmid_norm", "xout_norm",
              "branch_attn_norm", "branch_mlp_norm")
    accB = {k: [[] for _ in range(nb)] for k in _Bkeys}
    accDelta = {name: [[] for _ in range(nb)] for name in CHUNK_NAMES}
    accDeltaRel = {name: [[] for _ in range(nb)] for name in CHUNK_NAMES}
    accC = {v: [[] for _ in range(nb)] for v in ("full", "final")}
    # position-resolved D (mean later): sum over samples, per (block, position, variant)
    Dpos = {v: np.zeros((nb, HISTORY_SIZE)) for v in ("full", "final")}
    pert = {k: [] for k in ("da_full", "da_final", "dc_full_ro", "dc_final_ro", "dc_full_win")}

    cap = BlockCapture(model)
    checked = False
    with cap:
        for i in range(0, N, args.batch_size):
            sl = slice(i, i + args.batch_size)
            emb = _encode_emb(model, pixels[sl], device)
            c_t = _act_emb(model, a[sl], device)
            c_f = _act_emb(model, a_full[sl], device)
            c_x = _act_emb(model, a_final[sl], device)

            model.predictor(emb, c_t); snap_t = cap.snapshot()
            model.predictor(emb, c_f); snap_f = cap.snapshot()
            model.predictor(emb, c_x); snap_x = cap.snapshot()

            r = READOUT
            for bi in range(nb):
                ct = split_chunks(snap_t[bi]["adaln"][:, r])
                cf = split_chunks(snap_f[bi]["adaln"][:, r])
                attn_t = snap_t[bi]["attn_out"][:, r]
                mlp_t = snap_t[bi]["mlp_out"][:, r]
                xin_t = snap_t[bi]["x_in"][:, r]
                gated_attn = ct["gate_msa"] * attn_t
                xmid_t = xin_t + gated_attn
                gated_mlp = ct["gate_mlp"] * mlp_t

                if not checked:
                    recon = (xmid_t + gated_mlp)
                    assert torch.allclose(recon, snap_t[bi]["x_out"][:, r], atol=1e-4), \
                        "hook reconstruction mismatch"
                ua, um = gated_attn.norm(dim=-1), gated_mlp.norm(dim=-1)
                xin_n, xmid_n = xin_t.norm(dim=-1), xmid_t.norm(dim=-1)
                accB["inj_attn"][bi].append((ua / xin_n.clamp_min(EPS)).cpu().numpy())
                accB["inj_mlp"][bi].append((um / xmid_n.clamp_min(EPS)).cpu().numpy())
                accB["absgate_msa"][bi].append(ct["gate_msa"].abs().mean(dim=-1).cpu().numpy())
                accB["absgate_mlp"][bi].append(ct["gate_mlp"].abs().mean(dim=-1).cpu().numpy())
                accB["upd_attn"][bi].append(ua.cpu().numpy())
                accB["upd_mlp"][bi].append(um.cpu().numpy())
                accB["xin_norm"][bi].append(xin_n.cpu().numpy())
                accB["xmid_norm"][bi].append(xmid_n.cpu().numpy())
                accB["xout_norm"][bi].append(snap_t[bi]["x_out"][:, r].norm(dim=-1).cpu().numpy())
                accB["branch_attn_norm"][bi].append(attn_t.norm(dim=-1).cpu().numpy())
                accB["branch_mlp_norm"][bi].append(mlp_t.norm(dim=-1).cpu().numpy())

                for name in CHUNK_NAMES:
                    dch = (ct[name] - cf[name]).norm(dim=-1)
                    accDelta[name][bi].append(dch.cpu().numpy())
                    accDeltaRel[name][bi].append((dch / ct[name].norm(dim=-1).clamp_min(EPS)).cpu().numpy())

                ht = snap_t[bi]["x_out"][:, r]
                Dfull = (ht - snap_f[bi]["x_out"][:, r]).norm(dim=-1) / ht.norm(dim=-1).clamp_min(EPS)
                Dfinal = (ht - snap_x[bi]["x_out"][:, r]).norm(dim=-1) / ht.norm(dim=-1).clamp_min(EPS)
                accC["full"][bi].append(Dfull.cpu().numpy())
                accC["final"][bi].append(Dfinal.cpu().numpy())

                # position-resolved D (all 3 positions), summed for later mean
                for p in range(HISTORY_SIZE):
                    htp = snap_t[bi]["x_out"][:, p]
                    Dpos["full"][bi, p] += ((htp - snap_f[bi]["x_out"][:, p]).norm(dim=-1)
                                            / htp.norm(dim=-1).clamp_min(EPS)).sum().item()
                    Dpos["final"][bi, p] += ((htp - snap_x[bi]["x_out"][:, p]).norm(dim=-1)
                                             / htp.norm(dim=-1).clamp_min(EPS)).sum().item()
            checked = True

            # perturbation magnitudes (this minibatch)
            b = emb.size(0)
            pert["da_full"].append((a[sl] - a_full[sl]).reshape(b, -1).norm(dim=-1).numpy())
            pert["da_final"].append((a[sl] - a_final[sl]).reshape(b, -1).norm(dim=-1).numpy())
            pert["dc_full_ro"].append((c_t[:, r] - c_f[:, r]).norm(dim=-1).cpu().numpy())
            pert["dc_final_ro"].append((c_t[:, r] - c_x[:, r]).norm(dim=-1).cpu().numpy())
            pert["dc_full_win"].append((c_t - c_f).reshape(b, -1).norm(dim=-1).cpu().numpy())

    # stack
    B = {k: np.stack([np.concatenate(accB[k][bi]) for bi in range(nb)]) for k in accB}        # (nb, N)
    Delta = {k: np.stack([np.concatenate(accDelta[k][bi]) for bi in range(nb)]) for k in CHUNK_NAMES}
    DeltaRel = {k: np.stack([np.concatenate(accDeltaRel[k][bi]) for bi in range(nb)]) for k in CHUNK_NAMES}
    C = {v: np.stack([np.concatenate(accC[v][bi]) for bi in range(nb)]) for v in accC}        # (nb, N)
    Dpos = {v: Dpos[v] / N for v in Dpos}
    P = _vstack(pert)

    paths.ensure(ACT_DIR)
    np.savez(
        ARR_PATH, nb=nb, n=N, readout=READOUT,
        D_full=C["full"], D_final=C["final"],
        Dpos_full=Dpos["full"], Dpos_final=Dpos["final"],
        **{f"B_{k}": B[k] for k in B},
        **{f"delta_{k}": Delta[k] for k in CHUNK_NAMES},
        **{f"deltarel_{k}": DeltaRel[k] for k in CHUNK_NAMES},
        **{f"pert_{k}": P[k] for k in P},
    )
    return dict(nb=nb, N=N, n_ep=n_ep, B=B, Delta=Delta, DeltaRel=DeltaRel, C=C, Dpos=Dpos, P=P)


def _load_cache():
    d = np.load(ARR_PATH, allow_pickle=True)
    nb = int(d["nb"])
    B = {k[len("B_"):]: d[k] for k in d.files if k.startswith("B_")}
    C = {"full": d["D_full"], "final": d["D_final"]}
    Delta = {k: d[f"delta_{k}"] for k in CHUNK_NAMES}
    DeltaRel = {k: d[f"deltarel_{k}"] for k in CHUNK_NAMES}
    Dpos = {"full": d["Dpos_full"], "final": d["Dpos_final"]}
    P = {k[len("pert_"):]: d[k] for k in d.files if k.startswith("pert_")}
    return dict(nb=nb, N=int(d["n"]), n_ep=None, B=B, Delta=Delta, DeltaRel=DeltaRel, C=C, Dpos=Dpos, P=P)


def summarize(R):
    nb = R["nb"]
    B, C, Delta, P = R["B"], R["C"], R["Delta"], R["P"]
    m = lambda x: [float(v) for v in x.mean(axis=1)]  # per-block mean over samples

    inj_attn, inj_mlp = m(B["inj_attn"]), m(B["inj_mlp"])
    Dfull, Dfinal = m(C["full"]), m(C["final"])
    mlp_attn_ratio = [inj_mlp[i] / max(inj_attn[i], EPS) for i in range(nb)]
    le = lambda v: float(v[-1] / max(v[0], EPS))  # late/early ratio

    # Disentangle "gate capacity (param) vs realized use": the injection RATIO can be
    # flat even when the gate fires harder, if the residual-stream norm grows. So
    # report gate magnitude, raw gated-update norm, residual norm, the ratio, AND the
    # causal effect D_l together, and only call "artifact" if the effect fails to grow.
    cap_vs_use = {
        "mean_abs_gate_mlp_late_over_early": le(m(B["absgate_mlp"])),
        "raw_update_mlp_late_over_early": le(m(B["upd_mlp"])),
        "residual_xmid_norm_late_over_early": le(m(B["xmid_norm"])),
        "injection_ratio_mlp_late_over_early": le(inj_mlp),
        "D_l_final_late_over_early": le(Dfinal),
    }
    try:
        a_sum = json.loads((paths.RESULTS / "measurement_a" / "measurement_a_summary.json").read_text())
        cap_vs_use["gate_mlp_param_capacity_late_over_early"] = float(
            a_sum["gate_mlp_w_fro_by_depth"][-1] / a_sum["gate_mlp_w_fro_by_depth"][0])
    except FileNotFoundError:
        pass
    eff_grows = cap_vs_use["D_l_final_late_over_early"] > 1.2
    if eff_grows:
        cap_vs_use["verdict"] = (
            "NOT a parameter-space artifact: action propagation D_l grows with depth. "
            "Injection-ratio flatness is residual-norm growth (x"
            f"{cap_vs_use['residual_xmid_norm_late_over_early']:.2f}), not an unused gate "
            f"(raw mlp update x{cap_vs_use['raw_update_mlp_late_over_early']:.2f}).")
    else:
        cap_vs_use["verdict"] = (
            "WARNING: gate capacity present but D_l does not grow -> possible "
            "parameter-space artifact / wash-out; corroborate at E.")

    def trend_read(curve):
        c = np.asarray(curve)
        early = c[: max(1, nb // 3)].mean()
        late = c[-max(1, nb // 3):].mean()
        peak = int(np.argmax(c))
        rng = (c.max() - c.min()) / max(c.max(), EPS)
        if rng < 0.12:
            return "roughly uniform across depth"
        if late < 0.7 * early:
            return "decays with depth"
        if peak <= 1 and late < early:
            return "concentrated early (blocks 0-1)"
        if peak >= 1 and late >= 0.9 * c.max():
            return "grows into mid/late depth (no decay)"
        return f"grows then plateaus near block {peak}"

    return {
        "n_clips": R["N"], "n_episodes": R["n_ep"], "readout_token": READOUT,
        "injection_ratio_attn_by_depth": inj_attn,
        "injection_ratio_mlp_by_depth": inj_mlp,
        "mean_abs_gate_msa_by_depth": m(B["absgate_msa"]),
        "mean_abs_gate_mlp_by_depth": m(B["absgate_mlp"]),
        "raw_update_attn_by_depth": m(B["upd_attn"]),
        "raw_update_mlp_by_depth": m(B["upd_mlp"]),
        "residual_xin_norm_by_depth": m(B["xin_norm"]),
        "residual_xmid_norm_by_depth": m(B["xmid_norm"]),
        "residual_xout_norm_by_depth": m(B["xout_norm"]),
        "branch_attn_norm_by_depth": m(B["branch_attn_norm"]),
        "branch_mlp_norm_by_depth": m(B["branch_mlp_norm"]),
        "mlp_over_attn_injection_ratio": mlp_attn_ratio,
        "D_l_full_swap": Dfull,
        "D_l_final_swap": Dfinal,
        "delta_gate_msa": m(Delta["gate_msa"]),
        "delta_gate_mlp": m(Delta["gate_mlp"]),
        "delta_shift_msa": m(Delta["shift_msa"]),
        "delta_scale_msa": m(Delta["scale_msa"]),
        "delta_shift_mlp": m(Delta["shift_mlp"]),
        "delta_scale_mlp": m(Delta["scale_mlp"]),
        "perturbation": {
            "a_minus_aprime_full": float(P["da_full"].mean()),
            "a_minus_aprime_final": float(P["da_final"].mean()),
            "c_minus_cprime_full_readout": float(P["dc_full_ro"].mean()),
            "c_minus_cprime_final_readout": float(P["dc_final_ro"].mean()),
            "c_minus_cprime_full_window": float(P["dc_full_win"].mean()),
        },
        "Dpos_full": R["Dpos"]["full"].tolist(),
        "Dpos_final": R["Dpos"]["final"].tolist(),
        "capacity_vs_use": cap_vs_use,
        "reads": {
            "D_l_full_swap": trend_read(Dfull),
            "D_l_final_swap": trend_read(Dfinal),
            "injection_ratio_mlp": trend_read(inj_mlp),
            "injection_ratio_attn": trend_read(inj_attn),
        },
    }


def make_plot(R, summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = R["nb"]
    depth = np.arange(nb)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))

    # (A) C: D_l curve, both swaps, with input anchor 0
    d0 = np.concatenate([[0.0], summary["D_l_full_swap"]])
    d1 = np.concatenate([[0.0], summary["D_l_final_swap"]])
    xx = np.arange(-1, nb)
    ax[0, 0].plot(xx, d0, "-o", color="#9467bd", lw=2.4, label="full-history swap")
    ax[0, 0].plot(xx, d1, "-s", color="#2ca02c", lw=2.4, label="final-action swap")
    ax[0, 0].axvline(-1, color="k", lw=0.6, alpha=0.3)
    ax[0, 0].set_xlabel("residual stream after block (−1 = input)")
    ax[0, 0].set_ylabel("D_l  (normalized divergence at readout)")
    ax[0, 0].set_title("(C) action-perturbation propagation D_l")
    ax[0, 0].set_xticks(xx)
    ax[0, 0].legend()

    # (B) injection ratio per site
    ax[0, 1].plot(depth, summary["injection_ratio_attn_by_depth"], "-o", color="#1f77b4", lw=2.4, label="attn site  inj")
    ax[0, 1].plot(depth, summary["injection_ratio_mlp_by_depth"], "-o", color="#d62728", lw=2.4, label="mlp site  inj")
    ax[0, 1].set_xlabel("predictor block (depth)")
    ax[0, 1].set_ylabel("||gate ⊙ branch|| / ||x||")
    ax[0, 1].set_title("(B) realized injection ratio (12 sites)")
    ax[0, 1].set_xticks(depth)
    ax[0, 1].legend()

    # (C) modulation response to action swap: delta_gate
    ax[1, 0].plot(depth, summary["delta_gate_msa"], "-o", color="#1f77b4", lw=2.2, label="Δgate_msa")
    ax[1, 0].plot(depth, summary["delta_gate_mlp"], "-o", color="#d62728", lw=2.2, label="Δgate_mlp")
    ax[1, 0].plot(depth, summary["delta_scale_mlp"], "--^", color="#bba98a", lw=1.6, label="Δscale_mlp")
    ax[1, 0].plot(depth, summary["delta_shift_mlp"], "--v", color="#ddccbb", lw=1.6, label="Δshift_mlp")
    ax[1, 0].set_xlabel("predictor block (depth)")
    ax[1, 0].set_ylabel("||chunk(a) − chunk(a')||  at readout")
    ax[1, 0].set_title("(B) modulation response to action swap")
    ax[1, 0].set_xticks(depth)
    ax[1, 0].legend(fontsize=8)

    # (D) capacity-vs-use disentanglement (MLP branch), normalized to block 0
    def norm0(key):
        v = np.asarray(summary[key]); return v / max(v[0], EPS)
    try:
        a_sum = json.loads((paths.RESULTS / "measurement_a" / "measurement_a_summary.json").read_text())
        cap = np.asarray(a_sum["gate_mlp_w_fro_by_depth"]); cap = cap / cap[0]
        ax[1, 1].plot(depth, cap, ":", color="#7f7f7f", lw=2.0, label="gate ‖W‖ param capacity (A)")
    except FileNotFoundError:
        pass
    ax[1, 1].plot(depth, norm0("mean_abs_gate_mlp_by_depth"), "-o", color="#d62728", lw=2.2, label="mean|gate_mlp| (realized)")
    ax[1, 1].plot(depth, norm0("raw_update_mlp_by_depth"), "-^", color="#ff7f0e", lw=2.0, label="||gate⊙mlp|| raw update")
    ax[1, 1].plot(depth, norm0("residual_xmid_norm_by_depth"), "-s", color="#8c564b", lw=2.0, label="||x|| residual norm")
    ax[1, 1].plot(depth, norm0("injection_ratio_mlp_by_depth"), "-D", color="#1f77b4", lw=2.2, label="injection ratio")
    ax[1, 1].plot(depth, norm0("D_l_final_swap"), "-x", color="#2ca02c", lw=2.2, label="D_l (causal effect)")
    ax[1, 1].set_xlabel("predictor block (depth)")
    ax[1, 1].set_ylabel("normalized to block 0")
    ax[1, 1].set_title("(D) MLP gate: capacity vs use vs effect")
    ax[1, 1].set_xticks(depth)
    ax[1, 1].legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(s):
    nb = len(s["D_l_full_swap"])
    print("\n================ MEASUREMENTS B & C (12-site, readout token) ================")
    print(f"clips={s['n_clips']} episodes={s['n_episodes']} readout=pos {s['readout_token']}")
    print("\nperturbation magnitudes (z-action / embedding):")
    p = s["perturbation"]
    print(f"  ||a-a'||  full={p['a_minus_aprime_full']:.3f}  final={p['a_minus_aprime_final']:.3f}")
    print(f"  ||c-c'|| @readout  full={p['c_minus_cprime_full_readout']:.3f}  final={p['c_minus_cprime_final_readout']:.3f}")
    print("\nper-block (0..5):")
    def row(name, vals, f="{:.4f}"):
        print(f"  {name:<22}", " ".join(f.format(v) for v in vals))
    row("inj_attn (B)", s["injection_ratio_attn_by_depth"])
    row("inj_mlp  (B)", s["injection_ratio_mlp_by_depth"])
    row("mlp/attn inj ratio", s["mlp_over_attn_injection_ratio"], "{:.2f}")
    row("mean|gate_mlp|", s["mean_abs_gate_mlp_by_depth"])
    row("raw upd_mlp", s["raw_update_mlp_by_depth"])
    row("||x_mid|| residual", s["residual_xmid_norm_by_depth"], "{:.3f}")
    row("Δgate_mlp", s["delta_gate_mlp"])
    row("D_l full-swap (C)", s["D_l_full_swap"])
    row("D_l final-swap (C)", s["D_l_final_swap"])
    print("\nreads:")
    for k, v in s["reads"].items():
        print(f"  {k:<22} {v}")
    cu = s["capacity_vs_use"]
    print("\ncapacity-vs-use (mlp branch), late/early ratios:")
    print(f"  gate param ‖W‖ (A) x{cu.get('gate_mlp_param_capacity_late_over_early', float('nan')):.2f}"
          f" | mean|gate| x{cu['mean_abs_gate_mlp_late_over_early']:.2f}"
          f" | raw update x{cu['raw_update_mlp_late_over_early']:.2f}"
          f" | residual x{cu['residual_xmid_norm_late_over_early']:.2f}"
          f" | inj ratio x{cu['injection_ratio_mlp_late_over_early']:.2f}"
          f" | D_l x{cu['D_l_final_late_over_early']:.2f}")
    print(f"  -> {cu['verdict']}")
    print("============================================================================\n")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--n-clips", type=int, default=512)
    pa.add_argument("--batch-size", type=int, default=128)
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
