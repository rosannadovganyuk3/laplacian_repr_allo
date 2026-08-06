#!/bin/bash
# SLURM job array: one task per environment, matching run_full_experiments.sh's
# laprepr loop (./run_laprepr.sh OneRoom 5.0 / TwoRoom 5.0 / HardMaze 1.0).
#
# Submit from the repo root with:  sbatch slurm/train_laprepr.slurm
# See slurm/README.md for full instructions.

#SBATCH --job-name=laprepr_train
#SBATCH --account=aip-mbowling
# No --partition set: Vulcan didn't show one in the allocation info. If sbatch
# rejects the job asking for a partition, run `sinfo` on Vulcan to see valid
# options and add `#SBATCH --partition=<name>` back in.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3:59:00                 # ~300k steps takes ~2h48m at observed throughput; this gives ~70min buffer
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

ENVS=(OneRoom TwoRoom HardMaze)
W_NEGS=(5.0 5.0 1.0)
ENV_ID=${ENVS[$SLURM_ARRAY_TASK_ID]}
W_NEG=${W_NEGS[$SLURM_ARRAY_TASK_ID]}

# Sweep coefficient -- set via `sbatch --export=ALL,LAGRANGE_MULT=0.1 ...`;
# falls back to the config's own default (0.01) if not passed.
LAGRANGE_MULT="${LAGRANGE_MULT:-0.01}"

# Unique per submission (not per task), with a full timestamp. The 3 tasks in
# this array can genuinely start running at different real moments (whichever
# gets a free GPU shard first), so we can't just call `date` independently in
# each task -- they could disagree and land in different run folders. Instead
# we ask SLURM for this job's SubmitTime, which is one property shared by the
# whole array, guaranteed identical across all three tasks regardless of when
# each actually starts. The lagrange_mult value is embedded directly in the
# folder name too, so runs are self-documenting -- no more needing to grep
# flags.yaml to remember which folder used which coefficient.
SUBMIT_TIME=$(scontrol show job "${SLURM_ARRAY_JOB_ID}" \
    | grep -o 'SubmitTime=[^ ]*' | head -1 | cut -d= -f2)
RUN_ID="$(echo "${SUBMIT_TIME}" | tr -d ':-' | tr 'T' '_')_lm${LAGRANGE_MULT}"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: training ${ENV_ID} (w_neg=${W_NEG}, lagrange_mult=${LAGRANGE_MULT}) on $(hostname), run ${RUN_ID}"

# ---- environment setup ----
# Adjust module names to whatever your cluster provides (check `module avail`).
module load python cuda cudnn 2>/dev/null || true

# Point this at a venv that has torch (CUDA build), scipy, matplotlib, pyyaml
# installed. Create it once per cluster with, e.g.:
#   python -m venv ~/envs/laprepr && source ~/envs/laprepr/bin/activate
#   pip install torch scipy matplotlib pyyaml
source "${LAPREPR_VENV:-$HOME/envs/laprepr}/bin/activate"

cd "${SLURM_SUBMIT_DIR}"

# ---- run ----
python -u -B train_laprepr.py \
    --env_id="${ENV_ID}" \
    --log_sub_dir="${RUN_ID}" \
    --args="device='cuda'" \
    --args="w_neg=${W_NEG}" \
    --args="lagrange_mult=${LAGRANGE_MULT}"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: done."
