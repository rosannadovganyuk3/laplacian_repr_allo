#!/bin/bash
# SLURM job array for the Fir cluster: one task per grid combination in
# sweep_laprepr.py's GRID, run in parallel instead of the 256 sequential
# subprocess calls sweep_laprepr.py does on its own.
#
# The %20 in --array throttles this to at most 20 concurrent tasks -- Fir's
# def-mbowling GPU access is "Opportunistic Use" (likely a small, possibly
# preemptible pool of slots), so trying to claim all 256 at once would just
# flood the queue rather than actually run faster. Raise/lower the %20 based
# on how many opportunistic slots you actually see available (squeue/sinfo).
#
# --array's upper bound (255 below) must match
# `python sweep_laprepr.py --print_count` minus 1. If you edit GRID in
# sweep_laprepr.py, re-run --print_count and update the line below to match --
# SLURM won't do this for you, and a mismatch silently skips or errors on the
# extra/missing combinations.
#
# Same unfilled-in placeholders as train_laprepr_fir.slurm: partition,
# module names, and the venv all need verifying against Fir's own docs
# (see slurm/README.md for the discovery commands). Don't guess at these.
#
# Submit from the repo root with:  sbatch slurm/sweep_laprepr_fir.slurm

#SBATCH --job-name=laprepr_sweep
#SBATCH --account=def-mbowling
# No --partition: see train_laprepr_fir.slurm's note -- Fir auto-routes based
# on the GRES request below when --partition is omitted.
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1   # smallest H100 slice Fir offers -- plenty for this small model
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:59:00                  # sweep runs use total_train_steps=30000 (~1/7 of a full 200k-step run); bump up if that's not enough headroom
#SBATCH --array=0-255%20
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

ENV_ID="${SWEEP_ENV_ID:-OneRoom}"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: sweep combo ${SLURM_ARRAY_TASK_ID} for ${ENV_ID} on $(hostname)"

# ---- environment setup ----
# Adjust module names to whatever Fir actually provides (check `module avail`).
module load python cuda cudnn 2>/dev/null || true

# Same venv as train_laprepr_fir.slurm -- see slurm/README.md if not yet built.
source "${LAPREPR_VENV:-$HOME/envs/laprepr}/bin/activate"

cd "${SLURM_SUBMIT_DIR}"

# ---- run ----
# sweep_laprepr.py itself shells out to train_laprepr.py for the one combo
# selected by --task_id, so device selection is whatever the config picks
# (cuda, since torch.cuda.is_available() is true on a GPU node -- no need to
# force it like the single-run scripts do).
python -u -B sweep_laprepr.py \
    --task_id="${SLURM_ARRAY_TASK_ID}" \
    --env_id="${ENV_ID}" \
    --exp_name=laprepr_sweep

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: done."
