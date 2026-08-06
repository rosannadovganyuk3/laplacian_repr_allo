"""Batch ground-truth evaluation for every run in a sweep_laprepr.py sweep.

Meant to run on the cluster where the checkpoints already live, not after
copying them locally -- it reuses evaluate_reprs.evaluate() (no_figures=True,
so no plots) on every combo folder under log/<exp_name>/<env>/, and writes
ONE combined CSV. That CSV is the only thing worth syncing back locally --
a few hundred KB instead of the gigabytes of checkpoints it was computed
from.

Usage (run from the repo root, e.g. on a Fir login node):
    python evaluate_sweep.py
    python evaluate_sweep.py --envs OneRoom --exp_name laprepr_sweep
"""
import argparse
import csv
import os

from rl_lap.tools import flag_tools
import evaluate_reprs

# Scalar-only metrics worth a column (mirrors evaluate_reprs.py main()'s own
# `cols` list -- array-valued fields like sv_short/recov_short are skipped).
METRIC_COLS = [
    'n_states', 'n_obs', 'max_alias', 'ceiling', 'overlap_alias_eig',
    'overlap_short', 'overlap_long', 'overlap_cross',
    'window_len', 'overlap_short_windowed', 'overlap_long_windowed',
    'raw_cos', 'aligned_cos', 'erank_short', 'erank_long',
    'rho_gt', 'rho_short', 'rho_long', 'rho_concat', 'rho_pixels',
]

# Swept hyperparameters (from sweep_laprepr.py's GRID) pulled out of each
# run's own flags.yaml, so the CSV is self-documenting even if GRID changes
# later.
HPARAM_FIELDS = ['w_neg', 'c_neg', 'reg_neg', 'rnn_type']


def load_hparams(log_dir):
    flags = flag_tools.load_flags(log_dir)
    h = {f: getattr(flags, f, None) for f in HPARAM_FIELDS}
    discount = getattr(flags, 'discount', [None, None])
    h['discount_short'], h['discount_long'] = discount[0], discount[1]
    h['lr'] = getattr(flags.opt_args, 'lr', None)
    return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_base_dir', type=str,
                    default=os.path.join(os.getcwd(), 'log'))
    p.add_argument('--exp_name', type=str, default='laprepr_sweep')
    p.add_argument('--config_dir', type=str, default='rl_lap.configs')
    p.add_argument('--config_file', type=str,
                    default='laprepr_config_gridworld')
    p.add_argument('--envs', type=str, nargs='+',
                    default=['OneRoom', 'TwoRoom', 'HardMaze'])
    p.add_argument('--device', type=str, default='cpu',
                    help='defaults to cpu -- this is cheap enough to not '
                         'need a GPU allocation just to evaluate checkpoints')
    p.add_argument('--output_csv', type=str, default=None)
    p.add_argument('--rollout_steps', type=int, default=5000,
                    help='real random-walk steps collected per environment '
                         'for windowed evaluation (build_eval_context)')
    args = p.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(
            args.log_base_dir, f'{args.exp_name}_summary.csv')

    rows = []
    n_ok, n_skipped = 0, 0
    for env_id in args.envs:
        env_dir = os.path.join(args.log_base_dir, args.exp_name, env_id)
        if not os.path.isdir(env_dir):
            print(f'[skip] no directory for {env_id}: {env_dir}')
            continue
        tags = sorted(os.listdir(env_dir))

        # Ground truth (maze graph, true eigenvectors, ceiling) and the real
        # random-walk rollout used for windowed evaluation are identical for
        # every one of this env's ~256 runs -- build once here and reuse,
        # instead of paying for it (rollout collection especially) on every
        # single run.
        ctx_args = flag_tools.Flags(
            config_dir=args.config_dir, config_file=args.config_file,
            device=args.device)
        print(f'Building eval context for {env_id} '
              f'(ground truth + real-trajectory rollout)...')
        ctx = evaluate_reprs.build_eval_context(
            env_id, ctx_args, rollout_steps=args.rollout_steps)

        for tag in tags:
            run_dir = os.path.join(env_dir, tag)
            if not os.path.isfile(os.path.join(run_dir, 'flags.yaml')):
                continue  # not a run directory (or still being written)

            eval_args = flag_tools.Flags()
            eval_args.log_base_dir = args.log_base_dir
            eval_args.exp_name = args.exp_name
            eval_args.log_sub_dir = tag
            eval_args.config_dir = args.config_dir
            eval_args.config_file = args.config_file
            eval_args.device = args.device
            eval_args.no_figures = True
            eval_args.output_sub_dir = None

            try:
                hparams = load_hparams(run_dir)
                res, _, _ = evaluate_reprs.evaluate(env_id, eval_args, ctx)
            except Exception as e:
                print(f'[skip] {env_id}/{tag}: {type(e).__name__}: {e}')
                n_skipped += 1
                continue

            row = {'env': env_id, 'tag': tag}
            row.update(hparams)
            row.update({k: res[k] for k in METRIC_COLS})
            rows.append(row)
            n_ok += 1
            if n_ok % 20 == 0:
                print(f'...evaluated {n_ok} runs so far')

    if not rows:
        print('No runs found/evaluated -- nothing written.')
        return

    fieldnames = ['env', 'tag'] + HPARAM_FIELDS + \
        ['discount_short', 'discount_long', 'lr'] + METRIC_COLS
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nDone: {n_ok} runs evaluated, {n_skipped} skipped.')
    print(f'Written to {args.output_csv}')


if __name__ == '__main__':
    main()
