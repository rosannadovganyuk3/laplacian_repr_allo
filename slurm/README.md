# Running the laprepr retrain on a SLURM cluster

Throughout, `<cluster_user>@<cluster_login_node>` needs Vulcan's actual SSH
login hostname and your username there — I don't have that (only your
account name, `aip-mbowling`, from the allocation screenshot), so fill it in
with whatever your institution's docs / the portal you saw that allocation
info in tells you to use.

## 1. Get the code onto the cluster

From your local machine, sync the repo (skip the local virtualenvs and any
large log artifacts you don't need on the cluster copy):

```bash
rsync -avz --exclude='.venv' --exclude='myenv' --exclude='__pycache__' \
    --exclude='.git' \
    /Users/rosannadovganyuk/Documents/laplacian_repr_allo/ \
    rdovgany@vulcan.alliancecan.ca:~/laplacian_repr_allo/
```

## 2. One-time environment setup on the cluster

SSH in, then create a venv with the packages this repo needs (torch build
should match whatever CUDA module version the cluster provides):

```bash
ssh rdovgany@vulcan.alliancecan.ca
module load python cuda cudnn      # adjust to the module names Vulcan provides
python -m venv ~/envs/laprepr
source ~/envs/laprepr/bin/activate
pip install torch scipy matplotlib pyyaml
```

If your cluster's login node has no internet access for `pip install`, check
its docs for an offline/wheelhouse install method, or install from a compute
node with `srun --pty ...` if outbound access is only blocked on the login
node.

## 3. Account/partition (already set) and the venv path

`slurm/train_laprepr.slurm` is already set to `--account=aip-mbowling`, from
your Vulcan allocation (Group: aip-mbowling, Resource Type: GPU). No
`--partition` is set since Vulcan's allocation info didn't list one — if
`sbatch` rejects the job asking for a partition, run `sinfo` on Vulcan to see
the valid options and add `#SBATCH --partition=<name>` back into the script.

Set `LAPREPR_VENV` if your venv isn't at `~/envs/laprepr`:

```bash
export LAPREPR_VENV=~/envs/laprepr
```

(or just edit the `source` line directly in the script).

## 4. Submit

```bash
cd ~/laplacian_repr_allo
mkdir -p slurm/logs
sbatch slurm/train_laprepr.slurm
```

This submits a 3-task array (`--array=0-2`), one task per environment
(`OneRoom`, `TwoRoom`, `HardMaze`), each requesting its own GPU.

## 5. Monitor

```bash
squeue -u $USER                          # job status
tail -f slurm/logs/laprepr_train-<jobid>_0.out   # task 0 = OneRoom, _1 = TwoRoom, _2 = HardMaze
```

`sacct -j <jobid> --format=JobID,JobName,Elapsed,State,ExitCode` after
completion shows how long each task actually took and whether it succeeded —
useful for judging whether `--time=12:00:00` was enough headroom, since the
`window_len=8` history windowing is expected to run slower per step than the
original (memoryless) training.

## 6. Get the trained checkpoints back

Each task writes to `log/laprepr/<Env>/test/` on the cluster, exactly like a
local run would. Sync that back:

```bash
rsync -avz <cluster_user>@<cluster_login_node>:~/laplacian_repr_allo/log/laprepr/ \
    /Users/rosannadovganyuk/Documents/laplacian_repr_allo/log/laprepr/
```

Then `evaluate_reprs.py` can be run locally against the retrained checkpoints
as usual.
