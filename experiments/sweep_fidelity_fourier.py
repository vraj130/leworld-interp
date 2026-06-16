"""Fourier-host fidelity gate -- identical to sweep_fidelity, reading fourier's
machine-namespaced checkpoints and writing a machine-namespaced results file.

Reads trained checkpoints from checkpoints/depth_scaling_{env}_fourier/depth_{d}
(see train_sweep_fourier) and writes the fidelity table under
results/fourier/measurement_phase8_{env}_sweep/ so it never overwrites shannon's
results/measurement_phase8_{env}_sweep/fidelity_table.json.

The official released-reacher reference (lewm-{env}) and all gate logic are
imported unchanged, so this also serves as the fourier pipeline sanity check:
gating the released ref must reproduce the known reacher reference numbers
(rel-MSE 0.0066, 74.7x persist, rollout shuf/true 86.3x) on fourier's val cache.

    # released-ref reproduction only (no fourier checkpoints yet -> missing depths skipped):
    uv run python -m experiments.sweep_fidelity_fourier --env reacher --depths 3 6 --device cuda:0
"""

from __future__ import annotations

from leworld_interp import paths
import experiments.sweep_fidelity as SF


def ckpt_dir_fourier(env, depth):
    return paths.CHECKPOINTS / f"depth_scaling_{env}_fourier" / f"depth_{depth}"


# Read fourier checkpoints; write results under a fourier-namespaced RESULTS root.
SF.ckpt_dir = ckpt_dir_fourier
SF.paths.RESULTS = paths.RESULTS / "fourier"


if __name__ == "__main__":
    SF.main()
