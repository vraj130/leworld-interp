"""Phase 8 fidelity gate for the reacher depth sweep (released-quality bar).

For each trained depth and for the official released reacher checkpoint (depth 6, the reference),
load via build_lewm and run the same gate as Phase 0/5/7 (teacher-forced next-emb MSE + open-loop
rollout cost with true vs shuffled actions) on the held-out reacher val zstd cache. Reports a param
and fidelity table; only depths that pass at released-approaching quality are compared into the
fraction analysis. N defaults to 1500.

    uv run python -m experiments.sweep_fidelity --env reacher --depths 3 6 12 18 --device cuda:0
    uv run python -m experiments.sweep_fidelity --env reacher --from-cache
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
from experiments.depth_fidelity import encode
from experiments.train_sweep import env_h5, cache_dirs, ckpt_dir, FRAMESKIP

NUM_STEPS = 8
EPS = 1e-8


def load_val_batch(va_dir, n, seed):
    ds = PC.ZstdCachedWindows(va_dir, NUM_STEPS, FRAMESKIP)
    idx = np.sort(np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False))
    items = [ds[int(i)] for i in idx]
    return (torch.stack([it["pixels"] for it in items]),
            torch.stack([it["action"] for it in items]))


def gate(cfg_p, w_p, label, px, action_raw, amean, astd, dev):
    model, cfg = build_lewm(cfg_p, w_p, device=dev, dtype=torch.float32)
    nb = len(model.predictor.transformer.layers)
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
    return {"label": label, "depth": nb, "params_total_M": round(n_total / 1e6, 3),
            "params_predictor_M": round(n_pred / 1e6, 3), "params_per_block_M": round(n_pred / nb / 1e6, 3),
            "tf_next_emb_mse": tf_mse, "relative_mse": tf_mse / energy,
            "skill_vs_persistence": skill, "shuf_over_true": shuf, "passed": passed}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--env", default="reacher")
    pa.add_argument("--depths", type=int, nargs="+", default=[3, 6, 12, 18])
    pa.add_argument("--n", type=int, default=1500)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--from-cache", action="store_true")
    args = pa.parse_args()
    res_dir = paths.RESULTS / f"measurement_phase8_{args.env}_sweep"
    paths.ensure(res_dir)
    out = res_dir / "fidelity_table.json"
    if args.from_cache:
        print(out.read_text())
        return

    set_seed(args.seed)
    _, va_dir = cache_dirs(args.env)
    h5 = env_h5(args.env)
    am2, as2 = D.compute_action_stats(h5)
    amean, astd = PC.action_znorm(am2, as2, FRAMESKIP)
    amean, astd = amean.to(args.device), astd.to(args.device)
    px, action_raw = load_val_batch(va_dir, args.n, args.seed)
    print(f"[phase8 fidelity:{args.env}] {px.size(0)} held-out val clips")

    rows = []
    # official released reference (depth 6)
    rel_cfg = paths.CHECKPOINTS / f"lewm-{args.env}" / "config.json"
    rel_w = paths.CHECKPOINTS / f"lewm-{args.env}" / "weights.pt"
    if rel_cfg.exists():
        r = gate(rel_cfg, rel_w, f"released-{args.env}-ref", px, action_raw, amean, astd, args.device)
        rows.append(r)
        print(f"  released ref (d{r['depth']}): {r['params_total_M']:.2f}M rel={r['relative_mse']:.4f} "
              f"({r['skill_vs_persistence']:.1f}x persist) rollout shuf/true={r['shuf_over_true']:.1f}x -> "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    for d in args.depths:
        cfg_p = ckpt_dir(args.env, d) / "config.json"
        w_p = ckpt_dir(args.env, d) / "weights.pt"
        if not w_p.exists():
            print(f"  depth {d}: no checkpoint yet, skipping")
            continue
        r = gate(cfg_p, w_p, f"{args.env}-d{d}", px, action_raw, amean, astd, args.device)
        mp = ckpt_dir(args.env, d) / "train_meta.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            r["step"] = m.get("step"); r["train_wall_h"] = round(m.get("wall_time_s", 0) / 3600, 2)
        rows.append(r)
        print(f"  depth {d:2d}: {r['params_total_M']:.2f}M (pred {r['params_predictor_M']:.2f}M, "
              f"/blk {r['params_per_block_M']:.2f}M) rel={r['relative_mse']:.4f} "
              f"({r['skill_vs_persistence']:.1f}x persist) rollout shuf/true={r['shuf_over_true']:.1f}x -> "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    out.write_text(json.dumps({"env": args.env, "n_val_clips": int(px.size(0)), "rows": rows}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
