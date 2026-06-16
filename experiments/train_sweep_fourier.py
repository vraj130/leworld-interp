"""Fourier-host launcher for the reacher depth sweep -- identical to train_sweep,
output path namespaced by machine.

The shared NAS holds shannon's depth_scaling_reacher/depth_{3,6,12,18} outputs
(d12/d18 are shannon's live option-1 runs; d3/d6 are abandoned partials from an
earlier full-sweep attempt). To recover the missing {3,6} coverage from fourier
WITHOUT colliding with or resuming any of shannon's artifacts, this launcher
redirects only the checkpoint directory to

    checkpoints/depth_scaling_{env}_fourier/depth_{depth}

Everything else -- dataset, val cache (read-only, shared), normalization, gate
handling, optimizer, LR schedule, seed, SIGReg -- is imported unchanged from
experiments.train_sweep, so fourier's d3/d6 stay directly comparable to the rest
of the sweep. No config drift; only the write location differs.

    uv run python -m experiments.train_sweep_fourier --env reacher --depth 3 --max-steps 200000 --device cuda:0
    uv run python -m experiments.train_sweep_fourier --env reacher --depth 6 --max-steps 200000 --device cuda:1
"""

from __future__ import annotations

from leworld_interp import paths
import experiments.train_sweep as TS


def ckpt_dir_fourier(env, depth):
    return paths.CHECKPOINTS / f"depth_scaling_{env}_fourier" / f"depth_{depth}"


# Redirect output only; train_sweep.main() resolves ckpt_dir as a module global.
TS.ckpt_dir = ckpt_dir_fourier


if __name__ == "__main__":
    main = TS.main
    print(f"[fourier] checkpoints -> {ckpt_dir_fourier('<env>', '<depth>')}", flush=True)
    main()
