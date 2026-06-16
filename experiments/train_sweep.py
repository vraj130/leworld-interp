"""Phase 8 -- released-scale depth sweep on a chosen environment (reacher).

Generalizes the Phase 7 released-scale trainer to any env and any predictor depth, so the
reacher depth sweep {3,6,12,18} mirrors the PushT sweep exactly (identical config, only
predictor.depth varies, released-data scale via a lossless RAM-resident zstd cache). Phase 7's
train_depth_released.py is left frozen for reproducibility.

Data: a lossless per-frame zstd cache of the full train episodes (episode-disjoint val holdout,
200 episodes, seed 3072), RAM-resident so training is GPU-bound. Bit-exact vs the h5 frames.

Faithful to the released objective: next-emb MSE + 0.09 SIGReg, AdamW lr 5e-5 wd 1e-3, bf16,
grad-clip 1.0, history 3, num_preds 1, frameskip 5. Resumable (latest.pt), eval+checkpoint every
--eval-every, early-stop on a clean val rel-MSE plateau in the released band.

    uv run python -m experiments.train_sweep --env reacher --build-cache
    uv run python -m experiments.train_sweep --env reacher --depth 6 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from leworld_interp import data as D
from leworld_interp import paths
from leworld_interp import pixelcache as PC
from leworld_interp.lewm.module import SIGReg
from leworld_interp.model import set_seed
from experiments.train_depth import (build_fresh, config_dict, lejepa_loss, val_tf_mse,
                                     HISTORY_SIZE, NUM_PREDS, FRAMESKIP)

N_VAL_EPISODES = 200
SEED = 3072


def env_h5(env):
    ddir = paths.DATASETS / f"lewm-{env}"
    h5s = sorted(ddir.glob("*.h5"))
    return h5s[0] if h5s else ddir / f"{env}.h5"


def cache_dirs(env):
    root = paths.DATASETS / f"{env}_sweep_cache"
    return root / "train", root / "val"


def ckpt_dir(env, depth):
    return paths.CHECKPOINTS / f"depth_scaling_{env}" / f"depth_{depth}"


def maybe_build_caches(env):
    tr, va = cache_dirs(env)
    if (tr / "meta.json").exists() and (va / "meta.json").exists():
        m = json.loads((tr / "meta.json").read_text())
        print(f"[cache] using existing zstd caches for {env}: train {m['n_episodes']} episodes")
        return tr, va
    import h5py
    h5 = env_h5(env)
    with h5py.File(str(h5), "r", swmr=True) as f:
        n_ep = f["ep_len"].shape[0]
    perm = np.random.default_rng(SEED).permutation(n_ep)
    val_eps = perm[:N_VAL_EPISODES].tolist()
    train_eps = perm[N_VAL_EPISODES:].tolist()
    print(f"[cache] building {env} zstd caches: train={len(train_eps)} val={len(val_eps)} of {n_ep} episodes", flush=True)
    PC.build_cache_zstd(h5, val_eps, va, level=6)
    PC.build_cache_zstd(h5, train_eps, tr, level=6)
    return tr, va


def save_ckpt(model, action_input_dim, depth, extra, d):
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "weights.pt")
    (d / "config.json").write_text(json.dumps(config_dict(depth, action_input_dim), indent=2))
    (d / "train_meta.json").write_text(json.dumps(extra, indent=2))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--env", default="reacher")
    pa.add_argument("--depth", type=int, default=6)
    pa.add_argument("--max-steps", type=int, default=300000)
    pa.add_argument("--batch-size", type=int, default=128)
    pa.add_argument("--num-workers", type=int, default=18)
    pa.add_argument("--warmup-frac", type=float, default=0.02)
    pa.add_argument("--lr", type=float, default=5e-5)
    pa.add_argument("--wd", type=float, default=1e-3)
    pa.add_argument("--eval-every", type=int, default=5000)
    pa.add_argument("--ckpt-every", type=int, default=5000)
    pa.add_argument("--target-rel-mse", type=float, default=0.0072)
    pa.add_argument("--seed", type=int, default=SEED)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--build-cache", action="store_true")
    pa.add_argument("--no-resume", action="store_true")
    args = pa.parse_args()

    tr_dir, va_dir = maybe_build_caches(args.env)
    if args.build_cache:
        return

    set_seed(args.seed)
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    h5 = env_h5(args.env)
    probe = D.build_dataset(h5, num_steps=1, frameskip=1, normalize=False, keys_to_load=["pixels", "action"])
    action_dim = probe.get_dim("action")
    action_input_dim = FRAMESKIP * action_dim
    am2, as2 = D.compute_action_stats(h5)
    amean, astd = PC.action_znorm(am2, as2, FRAMESKIP)
    amean, astd = amean.to(dev), astd.to(dev)

    train_set = PC.ZstdCachedWindows(tr_dir, HISTORY_SIZE + NUM_PREDS, FRAMESKIP)
    val_set = PC.ZstdCachedWindows(va_dir, HISTORY_SIZE + NUM_PREDS, FRAMESKIP)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, persistent_workers=True, prefetch_factor=6,
        pin_memory=True, collate_fn=PC.collate, generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=3, collate_fn=PC.collate)

    model = build_fresh(args.depth, action_input_dim, dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    sigreg = SIGReg(knots=17, num_proj=1024).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup = max(1, int(args.warmup_frac * args.max_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, args.max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    outdir = ckpt_dir(args.env, args.depth)
    step, hist, best_rel = 0, [], float("inf")
    latest = outdir / "latest.pt"
    if latest.exists() and not args.no_resume:
        ck = torch.load(latest, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); step = ck["step"]; hist = ck["hist"]
        best_rel = ck.get("best_rel", float("inf"))
        print(f"[resume] {args.env} d{args.depth} from step {step} (best {best_rel:.5f})", flush=True)

    print(f"[sweep {args.env} d={args.depth}] params total={n_params/1e6:.2f}M pred={n_pred/1e6:.2f}M "
          f"params/block={n_pred/args.depth/1e6:.3f}M | train_clips={len(train_set)} val_clips={len(val_set)} "
          f"| max_steps={args.max_steps} action_dim={action_dim}", flush=True)

    model.train()
    t0 = time.time()
    done = step >= args.max_steps
    while not done:
        for b in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, pl, sl = lejepa_loss(model, sigreg, b, dev, amean, astd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1

            if step % 1000 == 0:
                print(f"  [{args.env} d{args.depth}] step {step}/{args.max_steps} loss={loss.item():.4f} "
                      f"pred={pl.item():.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)

            if step % args.eval_every == 0:
                mse, ratio, rel = val_tf_mse(model, val_loader, dev, amean, astd, n_batches=24)
                hist.append({"step": step, "val_tf_mse": mse, "skill_vs_persist": ratio, "rel_mse": rel})
                improved = rel < best_rel
                best_rel = min(best_rel, rel)
                print(f"  [eval {args.env} d{args.depth}] step {step} rel={rel:.4f} ({ratio:.1f}x persist) "
                      f"best={best_rel:.4f}", flush=True)
                meta = {"env": args.env, "depth": args.depth, "params_total": n_params,
                        "params_predictor": n_pred, "params_per_block": n_pred / args.depth,
                        "scale": "released", "max_steps": args.max_steps, "step": step,
                        "train_clips": len(train_set), "train_episodes": train_set.meta["n_episodes"],
                        "val_episodes": N_VAL_EPISODES, "batch_size": args.batch_size, "seed": args.seed,
                        "lr": args.lr, "wd": args.wd, "action_dim": action_dim,
                        "final_val_tf_mse": mse, "final_skill_vs_persist": ratio, "final_rel_mse": rel,
                        "best_rel_mse": best_rel, "wall_time_s": time.time() - t0, "history": hist}
                save_ckpt(model, action_input_dim, args.depth, meta, outdir)
                if improved:
                    save_ckpt(model, action_input_dim, args.depth, meta, outdir / "best")
                model.train()
                if rel <= args.target_rel_mse and len(hist) >= 3 and \
                        max(h["rel_mse"] for h in hist[-3:]) - min(h["rel_mse"] for h in hist[-3:]) < 0.0005:
                    print(f"[early-stop {args.env} d{args.depth}] rel {rel:.4f} <= {args.target_rel_mse} plateaued", flush=True)
                    done = True; break
                # convergence guard: stop if val has not improved over the last 5 evals (25k steps),
                # even above target, so a depth that plateaus high cannot waste the full step budget.
                if len(hist) >= 5 and \
                        max(h["rel_mse"] for h in hist[-5:]) - min(h["rel_mse"] for h in hist[-5:]) < 0.0004:
                    print(f"[converged {args.env} d{args.depth}] rel {rel:.4f} flat over 5 evals (above target)", flush=True)
                    done = True; break

            if step % args.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step, "hist": hist,
                            "best_rel": best_rel}, latest)
            if step >= args.max_steps:
                done = True; break

    mse, ratio, rel = val_tf_mse(model, val_loader, dev, amean, astd, n_batches=48)
    meta = {"env": args.env, "depth": args.depth, "params_total": n_params,
            "params_predictor": n_pred, "params_per_block": n_pred / args.depth, "scale": "released",
            "max_steps": args.max_steps, "step": step, "train_clips": len(train_set),
            "train_episodes": train_set.meta["n_episodes"], "val_episodes": N_VAL_EPISODES,
            "batch_size": args.batch_size, "seed": args.seed, "action_dim": action_dim,
            "final_val_tf_mse": mse, "final_skill_vs_persist": ratio, "final_rel_mse": rel,
            "best_rel_mse": min(best_rel, rel), "wall_time_s": time.time() - t0, "history": hist}
    save_ckpt(model, action_input_dim, args.depth, meta, outdir)
    print(f"\n[done {args.env} d={args.depth}] step {step} | final rel-MSE={rel:.4f} "
          f"({ratio:.1f}x persist) | best={min(best_rel, rel):.4f} | {(time.time()-t0)/3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
