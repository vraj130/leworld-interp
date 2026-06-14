"""Phase 6 -- retrain LeWM on PushT at a chosen predictor depth (depth-scaling study).

Faithful to the released training objective and hyperparameters (config/train/lewm.yaml):
next-embedding prediction MSE plus 0.09 * SIGReg, AdamW(lr 5e-5, wd 1e-3), bf16 autocast,
gradient clip 1.0, history_size 3, num_preds 1, SIGReg(knots 17, num_proj 1024). Model built
from the vendored classes (adaLN zero-init preserved) with ONLY predictor.depth varied. Saves
config.json + weights.pt so the checkpoint loads via leworld_interp.model.build_lewm.

For a fair cross-depth comparison every depth uses the SAME pixel cache (same train/val
EPISODES), SAME max-steps, and SAME seed; only predictor.depth differs. Because NFS random
reads bottleneck full-dataset training at ~1.6 s/step, training runs from a uint8 pixel cache
(a fixed episode subset, OS-page-cached) with GPU-side normalization. The reduced episode set
is identical across depths, so it does not confound the depth comparison; absolute fidelity may
sit below the released 18,685-episode checkpoint, which the fidelity gate accounts for.

    uv run python -m experiments.train_depth --build-cache         # one-time, ~min
    uv run python -m experiments.train_depth --depth 6 --smoke 60  # speed probe (cached)
    uv run python -m experiments.train_depth --depth 6 --max-steps 30000 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from leworld_interp import data as D
from leworld_interp import paths
from leworld_interp import pixelcache as PC
from leworld_interp.lewm.jepa import JEPA
from leworld_interp.lewm.module import ARPredictor, MLP, Embedder, SIGReg
from leworld_interp.model import set_seed

HISTORY_SIZE = 3
NUM_PREDS = 1
EMBED_DIM = 192
FRAMESKIP = 5
CKPT_ROOT = paths.CHECKPOINTS / "depth_scaling"
CACHE_ROOT = paths.DATASETS / "pusht_train_cache"


def build_fresh(depth, action_input_dim, device):
    import stable_pretraining as spt

    encoder = spt.backbone.utils.vit_hf("tiny", patch_size=14, image_size=224,
                                        pretrained=False, use_mask_token=False)
    predictor = ARPredictor(num_frames=HISTORY_SIZE, input_dim=EMBED_DIM, hidden_dim=EMBED_DIM,
                            output_dim=EMBED_DIM, depth=depth, heads=16, mlp_dim=2048,
                            dim_head=64, dropout=0.1, emb_dropout=0.0)
    action_encoder = Embedder(input_dim=action_input_dim, emb_dim=EMBED_DIM)

    def mlp():
        return MLP(input_dim=EMBED_DIM, hidden_dim=2048, output_dim=EMBED_DIM, norm_fn=nn.BatchNorm1d)

    return JEPA(encoder, predictor, action_encoder, mlp(), mlp()).to(device)


def config_dict(depth, action_input_dim):
    return {
        "encoder": {"size": "tiny", "patch_size": 14, "image_size": 224,
                    "pretrained": False, "use_mask_token": False},
        "predictor": {"num_frames": HISTORY_SIZE, "input_dim": EMBED_DIM, "hidden_dim": EMBED_DIM,
                      "output_dim": EMBED_DIM, "depth": depth, "heads": 16, "mlp_dim": 2048,
                      "dim_head": 64, "dropout": 0.1, "emb_dropout": 0.0},
        "action_encoder": {"input_dim": action_input_dim, "emb_dim": EMBED_DIM},
        "projector": {"input_dim": EMBED_DIM, "output_dim": EMBED_DIM, "hidden_dim": 2048},
        "pred_proj": {"input_dim": EMBED_DIM, "output_dim": EMBED_DIM, "hidden_dim": 2048},
    }


def maybe_build_cache(n_train_ep, n_val_ep, seed=3072):
    tr_dir, va_dir = CACHE_ROOT / "train", CACHE_ROOT / "val"
    if (tr_dir / "meta.json").exists() and (va_dir / "meta.json").exists():
        print(f"[cache] using existing cache at {CACHE_ROOT}")
        return tr_dir, va_dir
    import h5py
    with h5py.File(str(paths.PUSHT_H5), "r", swmr=True) as f:
        n_ep = f["ep_len"].shape[0]
    perm = np.random.default_rng(seed).permutation(n_ep)
    val_eps = perm[:n_val_ep]
    train_eps = perm[n_val_ep:n_val_ep + n_train_ep]
    print(f"[cache] building train={n_train_ep} val={n_val_ep} episodes (of {n_ep})")
    PC.build_cache(paths.PUSHT_H5, train_eps, tr_dir)
    PC.build_cache(paths.PUSHT_H5, val_eps, va_dir)
    return tr_dir, va_dir


def encode_batch(model, batch, dev, amean, astd):
    px = PC.gpu_normalize_pixels(batch["pixels"], dev)
    a = batch["action"].to(dev).float()
    a = torch.nan_to_num((a - amean) / astd, 0.0)
    return model.encode({"pixels": px, "action": a})


def lejepa_loss(model, sigreg, batch, dev, amean, astd):
    out = encode_batch(model, batch, dev, amean, astd)
    emb, act_emb = out["emb"], out["act_emb"]
    pred = model.predict(emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE])
    tgt = emb[:, NUM_PREDS:NUM_PREDS + HISTORY_SIZE]
    pred_loss = (pred - tgt).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    return pred_loss + 0.09 * sigreg_loss, pred_loss.detach(), sigreg_loss.detach()


@torch.inference_mode()
def val_tf_mse(model, val_loader, dev, amean, astd, n_batches=12):
    model.eval()
    se, persist, energy = [], [], []
    for i, b in enumerate(val_loader):
        if i >= n_batches:
            break
        out = encode_batch(model, b, dev, amean, astd)
        emb, act_emb = out["emb"], out["act_emb"]
        pred = model.predict(emb[:, :HISTORY_SIZE], act_emb[:, :HISTORY_SIZE]).float()
        tgt = emb[:, NUM_PREDS:NUM_PREDS + HISTORY_SIZE].float()
        se.append((pred - tgt).pow(2).mean().item())
        persist.append((emb[:, :HISTORY_SIZE].float() - tgt).pow(2).mean().item())
        energy.append((tgt ** 2).mean().item())
    model.train()
    mse = float(np.mean(se))
    return mse, float(np.mean(persist)) / max(mse, 1e-9), mse / float(np.mean(energy))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--depth", type=int, default=6)
    pa.add_argument("--max-steps", type=int, default=30000)
    pa.add_argument("--batch-size", type=int, default=128)
    pa.add_argument("--num-workers", type=int, default=8)
    pa.add_argument("--train-episodes", type=int, default=1500)
    pa.add_argument("--val-episodes", type=int, default=200)
    pa.add_argument("--warmup-frac", type=float, default=0.05)
    pa.add_argument("--lr", type=float, default=5e-5)
    pa.add_argument("--wd", type=float, default=1e-3)
    pa.add_argument("--eval-every", type=int, default=2000)
    pa.add_argument("--seed", type=int, default=3072)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--build-cache", action="store_true", help="build the pixel cache then exit")
    pa.add_argument("--smoke", type=int, default=0)
    args = pa.parse_args()

    tr_dir, va_dir = maybe_build_cache(args.train_episodes, args.val_episodes, seed=3072)
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

    train_set = PC.CachedWindows(tr_dir, HISTORY_SIZE + NUM_PREDS, FRAMESKIP)
    val_set = PC.CachedWindows(va_dir, HISTORY_SIZE + NUM_PREDS, FRAMESKIP)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None, pin_memory=True,
        collate_fn=PC.collate, generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=PC.collate)

    model = build_fresh(args.depth, action_input_dim, dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    sigreg = SIGReg(knots=17, num_proj=1024).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.smoke or args.max_steps
    warmup = max(1, int(args.warmup_frac * total))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"[train d={args.depth}] params total={n_params/1e6:.3f}M predictor={n_pred/1e6:.3f}M | "
          f"train_clips={len(train_set)} max_steps={total} bs={args.batch_size}", flush=True)

    model.train()
    step, t0, t_warm, hist, done = 0, time.time(), None, [], False
    while not done:
        for b in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, pl, sl = lejepa_loss(model, sigreg, b, dev, amean, astd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step == 10:
                t_warm = time.time()
            if args.smoke and step >= args.smoke:
                done = True
                break
            if step % 500 == 0:
                dt = (time.time() - t0) / step
                print(f"  step {step}/{total} loss={loss.item():.4f} pred={pl.item():.4f} "
                      f"sig={sl.item():.3f} lr={sched.get_last_lr()[0]:.2e} {dt*1000:.0f}ms/step", flush=True)
            if args.eval_every and step % args.eval_every == 0 and not args.smoke:
                mse, ratio, rel = val_tf_mse(model, val_loader, dev, amean, astd)
                hist.append({"step": step, "val_tf_mse": mse, "skill_vs_persist": ratio, "rel_mse": rel})
                print(f"  [eval] step {step} val_tf_mse={mse:.5f} ({ratio:.1f}x persist, rel {rel:.3f})", flush=True)
            if step >= total:
                done = True
                break

    if args.smoke:
        n = max(1, step - 10)
        per = (time.time() - (t_warm or t0)) / n
        print(f"\n[SMOKE d={args.depth}] {per*1000:.0f} ms/step over {n} steps | "
              f"{total} steps -> {per*30000/60:.1f} min for 30k steps", flush=True)
        return

    outdir = CKPT_ROOT / f"depth_{args.depth}"
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "weights.pt")
    (outdir / "config.json").write_text(json.dumps(config_dict(args.depth, action_input_dim), indent=2))
    fmse, fratio, frel = val_tf_mse(model, val_loader, dev, amean, astd, n_batches=24)
    meta = {"depth": args.depth, "params_total": n_params, "params_predictor": n_pred,
            "max_steps": total, "train_clips": len(train_set), "batch_size": args.batch_size,
            "seed": args.seed, "lr": args.lr, "wd": args.wd, "train_episodes": args.train_episodes,
            "val_episodes": args.val_episodes, "final_val_tf_mse": fmse,
            "final_skill_vs_persist": fratio, "final_rel_mse": frel,
            "wall_time_s": time.time() - t0, "history": hist}
    (outdir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[done d={args.depth}] saved {outdir} | final val_tf_mse={fmse:.5f} "
          f"({fratio:.1f}x persist, rel {frel:.3f}) | {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
