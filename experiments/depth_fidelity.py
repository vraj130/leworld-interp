"""Phase 6 fidelity gate for the retrained depth-scaling checkpoints.

For each trained depth, load the checkpoint via build_lewm and run the same fidelity gate
as Phase 0/5 (teacher-forced readout next-emb MSE + open-loop rollout cost with true vs
shuffled actions) on the held-out val-cache episodes (disjoint from training). Reports a
table of parameter counts and fidelity per depth. Only depths that pass are compared.

    uv run python -m experiments.depth_fidelity
    uv run python -m experiments.depth_fidelity --from-cache
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
from experiments.measurement_e import rollout_final, tf_run, HISTORY_SIZE, READOUT, ROLL_TARGET

CKPT_ROOT = paths.CHECKPOINTS / "depth_scaling"
VAL_CACHE = paths.DATASETS / "pusht_train_cache" / "val"
RES_DIR = paths.RESULTS / "measurement_phase6_depthscaling"
ARR = paths.ACTIVATIONS / "measurement_phase6_depthscaling"
EPS = 1e-8
NUM_STEPS = 8


@torch.inference_mode()
def encode(model, px_uint8, action_raw, amean, astd, dev, bs=256):
    embs, acts = [], []
    for i in range(0, px_uint8.size(0), bs):
        sl = slice(i, i + bs)
        px = PC.gpu_normalize_pixels(px_uint8[sl], dev)
        a = torch.nan_to_num((action_raw[sl].to(dev).float() - amean) / astd, 0.0)
        embs.append(model.encode({"pixels": px, "action": a})["emb"].float())
        acts.append(model.action_encoder(a).float())
    return torch.cat(embs), torch.cat(acts)


def load_val_batch(n, seed):
    ds = PC.CachedWindows(VAL_CACHE, NUM_STEPS, 5)
    idx = np.sort(np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False))
    items = [ds[int(i)] for i in idx]
    return (torch.stack([it["pixels"] for it in items]),
            torch.stack([it["action"] for it in items]))


def gate_depth(depth, px, action_raw, amean, astd, dev, ckpt_root=CKPT_ROOT):
    cfg_p = ckpt_root / f"depth_{depth}" / "config.json"
    w_p = ckpt_root / f"depth_{depth}" / "weights.pt"
    model, cfg = build_lewm(cfg_p, w_p, device=dev, dtype=torch.float32)
    n_total = sum(p.numel() for p in model.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    emb, act_emb = encode(model, px, action_raw, amean, astd, dev)
    N = emb.size(0)
    tf_tgt = emb[:, HISTORY_SIZE]
    roll_tgt = emb[:, ROLL_TARGET]
    pred, _ = tf_run(model, emb, act_emb, None, capture=False)
    tf_mse = float((pred[:, READOUT] - tf_tgt).pow(2).mean())
    persist = float((emb[:, READOUT] - tf_tgt).pow(2).mean())
    energy = float((tf_tgt ** 2).mean())
    cost_true = float((rollout_final(model, emb, act_emb, None) - roll_tgt).pow(2).sum(-1).mean())
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    cost_shuf = float((rollout_final(model, emb, act_emb[perm], None) - roll_tgt).pow(2).sum(-1).mean())
    skill = persist / max(tf_mse, EPS)
    shuf = cost_shuf / max(cost_true, EPS)
    passed = bool(skill > 3.0 and shuf > 2.0 and tf_mse / energy < 0.2)
    return {
        "depth": depth, "params_total": n_total, "params_predictor": n_pred,
        "params_total_M": round(n_total / 1e6, 3), "params_predictor_M": round(n_pred / 1e6, 3),
        "tf_next_emb_mse": tf_mse, "relative_mse": tf_mse / energy,
        "skill_vs_persistence": skill, "rollout_cost_true": cost_true,
        "rollout_cost_shuffled": cost_shuf, "shuf_over_true": shuf, "passed": passed,
    }


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--depths", type=int, nargs="*", default=None)
    pa.add_argument("--n", type=int, default=1500)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--ckpt-root", default=None, help="override checkpoint root (e.g. released)")
    pa.add_argument("--from-cache", action="store_true")
    args = pa.parse_args()
    from pathlib import Path
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else CKPT_ROOT
    # released-scale runs write to the Phase 7 dir so the Phase 6 table is preserved
    res_dir = (paths.RESULTS / "measurement_phase7_released_d12") if args.ckpt_root else RES_DIR
    paths.ensure(res_dir, ARR)
    out = res_dir / "fidelity_table.json"
    if args.from_cache:
        print(out.read_text())
        return

    set_seed(args.seed)
    depths = args.depths or sorted(int(p.name.split("_")[1]) for p in ckpt_root.glob("depth_*")
                                   if (p / "weights.pt").exists())
    am2, as2 = D.compute_action_stats(paths.PUSHT_H5)
    amean, astd = PC.action_znorm(am2, as2, 5)
    amean, astd = amean.to(args.device), astd.to(args.device)
    px, action_raw = load_val_batch(args.n, args.seed)
    print(f"[phase6 fidelity] depths={depths} on {px.size(0)} held-out val clips (ckpt_root={ckpt_root.name})")

    rows = []
    for d in depths:
        r = gate_depth(d, px, action_raw, amean, astd, args.device, ckpt_root=ckpt_root)
        rows.append(r)
        # merge in training meta if present
        mp = ckpt_root / f"depth_{d}" / "train_meta.json"
        if mp.exists():
            r["train_wall_min"] = round(json.loads(mp.read_text()).get("wall_time_s", 0) / 60, 1)
        print(f"  depth {d:2d}: {r['params_total_M']:.2f}M (pred {r['params_predictor_M']:.2f}M)  "
              f"TF MSE={r['tf_next_emb_mse']:.5f} rel={r['relative_mse']:.3f} "
              f"({r['skill_vs_persistence']:.1f}x persist)  rollout shuf/true={r['shuf_over_true']:.1f}x  "
              f"-> {'PASS' if r['passed'] else 'FAIL'}")
    out.write_text(json.dumps({"n_val_clips": int(px.size(0)), "rows": rows}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
