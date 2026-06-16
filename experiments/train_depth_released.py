"""Phase 7 -- retrain LeWM predictor depth 12 at RELEASED-DATA scale.

The Phase 6 depth law was established in a reduced-data regime (1500 episodes, rel-MSE
about 0.025) and verified at released scale (rel-MSE about 0.007) only at depth 6. This
trains a single depth-12 model on the FULL released episode set so its commitment depth can
be measured at released quality, giving two released-scale points (d6, d12) spanning the
absolute-vs-relative divergence.

Data path: a raw uint8 cache of all 18k episodes is ~352 GB (far beyond this box's 62 GB
RAM, so random-access training on it is NFS-I/O-bound, and direct h5 reads are ~1.4 s/step).
A lossless per-frame zstd cache holds the full set in ~25-30 GB (RAM-resident, page-cached)
and decodes at ~9800 frames/s/core, so training is GPU-bound. The cache is faithful to the
released uint8 frames (lossless). Val is the SAME 200 held-out episodes as Phase 6 (seed
3072), so fidelity is directly comparable to d6 and reduced-d12.

Everything else matches the released config and the Phase 6 trainer (next-emb MSE + 0.09
SIGReg, AdamW lr 5e-5 wd 1e-3, bf16, grad-clip 1.0, history 3, num_preds 1). Only the data
scale and step budget change. Resumable: writes latest.pt (full state) every --ckpt-every and
auto-resumes; writes config.json + weights.pt + train_meta.json so the checkpoint loads via
build_lewm at any eval point.

    uv run python -m experiments.train_depth_released --build-cache       # one-time
    uv run python -m experiments.train_depth_released --device cuda:0      # train (auto-resume)
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

CKPT_DIR = paths.CHECKPOINTS / "depth_scaling_released" / "depth_12"
TRAIN_CACHE = paths.DATASETS / "pusht_released_cache" / "train"   # zstd, full train set
VAL_CACHE = paths.DATASETS / "pusht_train_cache" / "val"          # raw uint8, 200 episodes (Phase 6)
N_VAL_EPISODES = 200
SEED = 3072


def maybe_build_cache():
    if (TRAIN_CACHE / "meta.json").exists():
        m = json.loads((TRAIN_CACHE / "meta.json").read_text())
        print(f"[cache] using existing zstd train cache: {m['n_episodes']} episodes, "
              f"{m['n_frames']} frames at {TRAIN_CACHE}")
        return
    import h5py
    with h5py.File(str(paths.PUSHT_H5), "r", swmr=True) as f:
        n_ep = f["ep_len"].shape[0]
    perm = np.random.default_rng(SEED).permutation(n_ep)
    val_eps = set(perm[:N_VAL_EPISODES].tolist())
    train_eps = [e for e in range(n_ep) if e not in val_eps]    # all 18,485 non-val episodes
    print(f"[cache] building released zstd train cache: {len(train_eps)} episodes "
          f"(of {n_ep}, holding out the same {N_VAL_EPISODES} val episodes as Phase 6)", flush=True)
    PC.build_cache_zstd(paths.PUSHT_H5, train_eps, TRAIN_CACHE, level=6)


def save_checkpoint(model, action_input_dim, depth, extra, path_dir):
    path_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path_dir / "weights.pt")
    (path_dir / "config.json").write_text(json.dumps(config_dict(depth, action_input_dim), indent=2))
    (path_dir / "train_meta.json").write_text(json.dumps(extra, indent=2))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--depth", type=int, default=12)
    pa.add_argument("--max-steps", type=int, default=300000)
    pa.add_argument("--batch-size", type=int, default=128)
    pa.add_argument("--num-workers", type=int, default=20)
    pa.add_argument("--warmup-frac", type=float, default=0.02)
    pa.add_argument("--lr", type=float, default=5e-5)
    pa.add_argument("--wd", type=float, default=1e-3)
    pa.add_argument("--eval-every", type=int, default=5000)
    pa.add_argument("--ckpt-every", type=int, default=5000)
    pa.add_argument("--target-rel-mse", type=float, default=0.008,
                    help="early-stop if val rel-MSE reaches this and stops improving")
    pa.add_argument("--seed", type=int, default=SEED)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--build-cache", action="store_true")
    pa.add_argument("--no-resume", action="store_true")
    args = pa.parse_args()

    maybe_build_cache()
    if args.build_cache:
        return

    set_seed(args.seed)
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    probe = D.build_dataset(paths.PUSHT_H5, num_steps=1, frameskip=1, normalize=False)
    action_dim = probe.get_dim("action")
    action_input_dim = FRAMESKIP * action_dim
    am2, as2 = D.compute_action_stats(paths.PUSHT_H5)
    amean, astd = PC.action_znorm(am2, as2, FRAMESKIP)
    amean, astd = amean.to(dev), astd.to(dev)

    train_set = PC.ZstdCachedWindows(TRAIN_CACHE, HISTORY_SIZE + NUM_PREDS, FRAMESKIP)
    val_set = PC.CachedWindows(VAL_CACHE, HISTORY_SIZE + NUM_PREDS, FRAMESKIP)
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

    step, hist, best_rel = 0, [], float("inf")
    latest = CKPT_DIR / "latest.pt"
    if latest.exists() and not args.no_resume:
        ck = torch.load(latest, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); step = ck["step"]; hist = ck["hist"]
        best_rel = ck.get("best_rel", float("inf"))
        print(f"[resume] from step {step} (best rel-MSE {best_rel:.5f})", flush=True)

    print(f"[released d={args.depth}] params total={n_params/1e6:.2f}M pred={n_pred/1e6:.2f}M | "
          f"train_clips={len(train_set)} val_clips={len(val_set)} | max_steps={args.max_steps} "
          f"bs={args.batch_size} workers={args.num_workers}", flush=True)

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

            if step % 500 == 0:
                rate = step / max(time.time() - t0, 1e-9) if step > 0 else 0
                print(f"  step {step}/{args.max_steps} loss={loss.item():.4f} pred={pl.item():.4f} "
                      f"sig={sl.item():.3f} lr={sched.get_last_lr()[0]:.2e}", flush=True)

            if step % args.eval_every == 0:
                mse, ratio, rel = val_tf_mse(model, val_loader, dev, amean, astd, n_batches=24)
                hist.append({"step": step, "val_tf_mse": mse, "skill_vs_persist": ratio, "rel_mse": rel})
                improved = rel < best_rel
                best_rel = min(best_rel, rel)
                print(f"  [eval] step {step} val_tf_mse={mse:.5f} ({ratio:.1f}x persist, "
                      f"rel {rel:.4f}) best={best_rel:.4f}", flush=True)
                meta = {"depth": args.depth, "params_total": n_params, "params_predictor": n_pred,
                        "scale": "released", "max_steps": args.max_steps, "step": step,
                        "train_clips": len(train_set), "train_episodes": train_set.meta["n_episodes"],
                        "val_episodes": N_VAL_EPISODES, "batch_size": args.batch_size,
                        "seed": args.seed, "lr": args.lr, "wd": args.wd,
                        "final_val_tf_mse": mse, "final_skill_vs_persist": ratio, "final_rel_mse": rel,
                        "best_rel_mse": best_rel, "wall_time_s": time.time() - t0, "history": hist}
                save_checkpoint(model, action_input_dim, args.depth, meta, CKPT_DIR)
                if improved:
                    save_checkpoint(model, action_input_dim, args.depth, meta, CKPT_DIR / "best")
                model.train()
                # early stop: reached released territory and last 3 evals flat
                if rel <= args.target_rel_mse and len(hist) >= 3:
                    recent = [h["rel_mse"] for h in hist[-3:]]
                    if max(recent) - min(recent) < 0.0005:
                        print(f"[early-stop] rel-MSE {rel:.4f} <= target {args.target_rel_mse} and plateaued", flush=True)
                        done = True; break

            if step % args.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step, "hist": hist,
                            "best_rel": best_rel}, latest)

            if step >= args.max_steps:
                done = True; break

    # final eval + save
    mse, ratio, rel = val_tf_mse(model, val_loader, dev, amean, astd, n_batches=48)
    meta = {"depth": args.depth, "params_total": n_params, "params_predictor": n_pred,
            "scale": "released", "max_steps": args.max_steps, "step": step,
            "train_clips": len(train_set), "train_episodes": train_set.meta["n_episodes"],
            "val_episodes": N_VAL_EPISODES, "batch_size": args.batch_size, "seed": args.seed,
            "lr": args.lr, "wd": args.wd, "final_val_tf_mse": mse,
            "final_skill_vs_persist": ratio, "final_rel_mse": rel, "best_rel_mse": min(best_rel, rel),
            "wall_time_s": time.time() - t0, "history": hist}
    save_checkpoint(model, action_input_dim, args.depth, meta, CKPT_DIR)
    print(f"\n[done released d={args.depth}] step {step} | final val rel-MSE={rel:.4f} "
          f"({ratio:.1f}x persist) | best={min(best_rel, rel):.4f} | {(time.time()-t0)/3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
