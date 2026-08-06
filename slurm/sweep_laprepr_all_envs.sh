#!/bin/bash
# Submits sweep_laprepr_fir.slurm's full 256-combo grid array once per
# environment, so OneRoom/TwoRoom/HardMaze each get their own independent
# sweep (the GRID itself doesn't vary by room -- env_id is a separate axis,
# not part of GRID).
#
# Run from the repo root on Fir:  bash slurm/sweep_laprepr_all_envs.sh

ENVS=(OneRoom TwoRoom HardMaze)

for ENV_ID in "${ENVS[@]}"; do
    echo "Submitting sweep for ${ENV_ID}..."
    sbatch --export=ALL,SWEEP_ENV_ID="${ENV_ID}" slurm/sweep_laprepr_fir.slurm
done
