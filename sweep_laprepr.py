"""Full grid-search sweep over LapReprLearner hyperparameters.

Launches one `train_laprepr.py` subprocess per combination in GRID, each
writing to its own log_sub_dir under log/<exp_name>/<env_id>/. Edit GRID
below to control which values (and therefore how many runs) are swept.

Each run overrides total_train_steps/n_samples down to a smaller sweep
budget by default -- comparing 128 runs at the full 200000-step config
would be prohibitively expensive. Re-run the winning config at full length
separately once you've picked it.

Usage:
    python sweep_laprepr.py --dry_run          # see the plan, run nothing
    python sweep_laprepr.py                    # launch the full grid, sequentially
    python sweep_laprepr.py --env_id TwoRoom
    python sweep_laprepr.py --print_count      # print len(GRID combos), e.g. for --array=0-N
    python sweep_laprepr.py --task_id 42       # run only combo #42 (one SLURM array task)
"""
import argparse
import itertools
import os
import subprocess
import sys

# --- Grid definition: edit these lists to control sweep size/scope. ---
# Total runs = product of the list lengths below (currently 256; check with
# `python sweep_laprepr.py --print_count`).
GRID = {
    'window_len':   [1, 4, 8, 16],
    'w_neg':        [5.0, 15.0],
    'c_neg':        [1.0, 2.0],
    'reg_neg':      [0.0, 0.1],
    'discount':     [[0.1, 0.9], [0.3, 0.7]],
    'opt_args.lr':  [0.0001, 0.001],
    'rnn_type':     ['LSTM', 'GRU'],
}


def format_arg(key, val):
    # String-valued flags (e.g. rnn_type) need to round-trip through
    # ast.literal_eval on the receiving end, so they must stay quoted.
    val_str = f"'{val}'" if isinstance(val, str) else str(val)
    return f'{key}={val_str}'


def run_tag(index, combo):
    def short(k):
        return k.split('.')[-1]
    parts = [f'{short(k)}={v}'.replace(' ', '') for k, v in combo.items()]
    return f'{index:04d}_' + '_'.join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='OneRoom')
    p.add_argument('--exp_name', type=str, default='laprepr_sweep')
    p.add_argument('--log_base_dir', type=str,
                    default=os.path.join(os.getcwd(), 'log'))
    p.add_argument('--config_file', type=str,
                    default='laprepr_config_gridworld')
    p.add_argument('--sweep_train_steps', type=int, default=30000,
                    help='total_train_steps applied to every sweep run')
    p.add_argument('--sweep_n_samples', type=int, default=30000,
                    help='n_samples applied to every sweep run')
    p.add_argument('--dry_run', action='store_true',
                    help='print the run plan without launching anything')
    p.add_argument('--start_at', type=int, default=0,
                    help='resume a partially-completed sweep from this index')
    p.add_argument('--task_id', type=int, default=None,
                    help='run only this one combination index (e.g. from '
                         '$SLURM_ARRAY_TASK_ID), instead of looping over the '
                         'whole grid in this process')
    p.add_argument('--print_count', action='store_true',
                    help='print the number of combinations in GRID and exit '
                         '(use to size an --array=0-N SBATCH line)')
    args = p.parse_args()

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))

    if args.print_count:
        print(len(combos))
        return

    print(f'{len(combos)} combinations in the grid.')

    if args.task_id is not None:
        indices = [args.task_id]
    else:
        indices = range(args.start_at, len(combos))

    for i in indices:
        values = combos[i]
        combo = dict(zip(keys, values))
        tag = run_tag(i, combo)

        cmd = [
            sys.executable, 'train_laprepr.py',
            '--env_id', args.env_id,
            '--exp_name', args.exp_name,
            '--log_base_dir', args.log_base_dir,
            '--config_file', args.config_file,
            '--log_sub_dir', tag,
        ]
        for k, v in combo.items():
            cmd += ['--args', format_arg(k, v)]
        cmd += ['--args', format_arg('total_train_steps', args.sweep_train_steps)]
        cmd += ['--args', format_arg('n_samples', args.sweep_n_samples)]
        # save_freq = total_train_steps -> only the final checkpoint gets
        # written; evaluate_reprs.py never reads the intermediate ones anyway.
        cmd += ['--args', format_arg('save_freq', args.sweep_train_steps)]

        print(f'\n[{i + 1}/{len(combos)}] {tag}')
        print(' '.join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
