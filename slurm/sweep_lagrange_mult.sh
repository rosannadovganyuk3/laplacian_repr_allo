#!/bin/bash
# Submits one job array (all 3 environments) per lagrange_mult value being
# swept. Each submission gets its own unique RUN_ID automatically (embedding
# the coefficient value too), so all runs land in separate, self-documenting
# folders with no manual coordination needed.
#
# Run from the repo root on Vulcan:  bash slurm/sweep_lagrange_mult.sh

VALUES=(0.001 0.003 0.03 0.1 0.3 0.5 0.9)

for LM in "${VALUES[@]}"; do
    echo "Submitting lagrange_mult=${LM}..."
    sbatch --export=ALL,LAGRANGE_MULT="${LM}" slurm/train_laprepr.slurm
done
