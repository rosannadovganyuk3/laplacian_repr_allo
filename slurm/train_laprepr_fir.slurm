#!/bin/bash
# SLURM job array for the Fir cluster: one task per environment, matching
# run_full_experiments.sh's laprepr loop
# (./run_laprepr.sh OneRoom 5.0 / TwoRoom 5.0 / HardMaze 1.0).
#
# This is adapted from slurm/train_laprepr.slurm (written for Vulcan). Fir's
# account name, module names, and GPU-type syntax are almost certainly
# different -- the placeholders below need filling in from Fir's own docs or
# discovery commands (see slurm/README.md for the general pattern we used on
# Vulcan: sshare -U $USER for account, sinfo for partitions, module avail for
# module names). Don't guess at these -- an invalid --account or --partition
# will make sbatch reject the job outright.
#
# Submit from the repo root with:  sbatch slurm/train_laprepr_fir.slurm

#SBATCH --job-name=laprepr_train
# def-mbowling is the only Fir account with GPU access (rrg-mbowling-ad is
# CPU-only). Its GPU access is listed as "Opportunistic Use" -- likely lower
# scheduling priority and possibly preemptible mid-run. train_laprepr.py has
# no resume-from-checkpoint support, so a preempted task would need restarting
# from scratch, not just resumed.
#SBATCH --account=def-mbowling
# No --partition: Fir rejected "gpu" as invalid/unfit for this GRES request,
# and its own error message says to omit --partition and let the scheduler
# auto-route based on the GPU type requested below.
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1   # smallest H100 slice Fir offers (10GB) -- plenty for this small model
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G                       # bumped from 4G: n_samples=200000/replay_buffer_size=200000 OOM'd at 4G (job 53256086), likely the per-step one-hot `index` field (sized to n_states, stored twice per step) dominating memory
#SBATCH --time=3:59:00                  # full 200000-step run per the updated config; no checkpoint-resume, so leave real buffer
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

ENVS=(OneRoom TwoRoom HardMaze)
ENV_ID=${ENVS[$SLURM_ARRAY_TASK_ID]}

# w_neg (and every other GDO hyperparameter) now comes straight from
# laprepr_config_gridworld.py's own defaults -- the sweep found w_neg=15.0
# (among others) wins consistently across all three environments, so there's
# no need for the old per-environment override this script used to pass.

# Unique per submission (not per task), with a full timestamp -- same approach
# as the Vulcan script: SLURM_ARRAY_JOB_ID's SubmitTime is shared by all 3
# tasks in this array, so they land in the same run folder even though each
# task may start running at a different real moment.
SUBMIT_TIME=$(scontrol show job "${SLURM_ARRAY_JOB_ID}" \
    | grep -o 'SubmitTime=[^ ]*' | head -1 | cut -d= -f2)
RUN_ID=$(echo "${SUBMIT_TIME}" | tr -d ':-' | tr 'T' '_')

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: training ${ENV_ID} on $(hostname), run ${RUN_ID}"

# ---- environment setup ----
# Adjust module names to whatever Fir actually provides (check `module avail`
# on Fir -- these are very likely not the same names as Vulcan's).
module load python cuda cudnn 2>/dev/null || true

# Point this at a venv that has torch (CUDA build), scipy, matplotlib, pyyaml
# installed. Create one on Fir the same way we did on Vulcan:
#   python -m venv ~/envs/laprepr && source ~/envs/laprepr/bin/activate
#   pip install torch scipy matplotlib pyyaml
# A venv built on Vulcan will NOT necessarily work on Fir -- different cluster,
# possibly different CPU architecture/CUDA version, so build a fresh one here.
source "${LAPREPR_VENV:-$HOME/envs/laprepr}/bin/activate"

cd "${SLURM_SUBMIT_DIR}"

# ---- run ----
python -u -B train_laprepr.py \
    --env_id="${ENV_ID}" \
    --log_sub_dir="${RUN_ID}" \
    --args="device='cuda'"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: done."
