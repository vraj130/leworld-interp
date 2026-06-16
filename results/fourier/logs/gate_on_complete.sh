#!/usr/bin/env bash
# Detached watcher: wait until BOTH fourier reacher d3 and d6 training processes
# have exited, then run the fidelity gate on the trained checkpoints (n=1500) and
# also at shannon's n=384 sample for an apples-to-apples comparison. Idempotent-ish;
# writes results under results/fourier/ and a sentinel log here.
set -u
cd /mnt/NAS/home/vg2097/leworld-interp
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=".venv-$(hostname -s)"
export UV_LINK_MODE=copy
SENT="results/fourier/logs/gate_on_complete.out"
echo "[watcher] started $(date -Is); waiting for d3+d6 to finish" > "$SENT"

running() { pgrep -fa "train_sweep_fourier --env reacher --depth $1" | grep -qv pgrep; }
while running 3 || running 6; do sleep 120; done

echo "[watcher] both runs exited $(date -Is); running gate" >> "$SENT"
uv run --frozen python -m experiments.sweep_fidelity_fourier --env reacher --depths 3 6 --n 1500 --seed 0 --device cuda:0 >> "$SENT" 2>&1
echo "[watcher] --- gate at shannon's n=384 ---" >> "$SENT"
uv run --frozen python -m experiments.sweep_fidelity_fourier --env reacher --depths 3 6 --n 384 --seed 0 --device cuda:0 >> "$SENT" 2>&1
echo "[watcher] DONE $(date -Is)" >> "$SENT"
