"""Rank sweep results from evaluate_sweep.py's summary CSV.

Usage:
    python analyze_sweep.py                              # log/laprepr_sweep_summary.csv
    python analyze_sweep.py path/to/other_summary.csv
    python analyze_sweep.py --env OneRoom --top 20
"""
import argparse
import sys

import pandas as pd

HPARAMS = ['window_len', 'w_neg', 'c_neg', 'reg_neg',
           'discount_short', 'discount_long', 'lr', 'rnn_type']


def load(csv_path):
    df = pd.read_csv(csv_path)
    # Use the *_windowed columns, not the plain overlap_short/overlap_long:
    # evaluate_reprs.py's embed() defaults to window_len=1, so the plain
    # columns test every model with a single frame regardless of what
    # window_len it was actually trained with -- unfairly out-of-distribution
    # for anything but window_len=1. overlap_*_windowed feeds each model the
    # number of frames it actually trained with (identical to the plain
    # column when window_len==1, so nothing changes for those rows).
    # overlap_cross has no windowed variant, so it's left as-is.
    df['overlap_ratio_short'] = df['overlap_short_windowed'] / df['ceiling']
    df['overlap_ratio_long'] = df['overlap_long_windowed'] / df['ceiling']
    df['overlap_ratio_cross'] = df['overlap_cross'] / df['ceiling']
    return df


def show_top_runs(df, metric, top):
    print(f"\n{'='*90}\nTop {top} runs by {metric}\n{'='*90}")
    # ceiling and erank ride along with every leaderboard so a high ratio
    # can be sanity-checked at a glance: was the ceiling itself high (a
    # meaningful win) or low (acing an easy target)? is the branch's
    # effective rank healthy, or is it hitting this score through a
    # collapsed handful of dimensions?
    branch = 'long' if 'long' in metric else (
        'short' if 'short' in metric else None)
    erank_col = f'erank_{branch}' if branch else 'erank_short'
    cols = ['env', 'tag'] + HPARAMS + [metric, 'ceiling', erank_col]
    print(df.sort_values(metric, ascending=False)[cols]
            .head(top).to_string(index=False))


def show_hyperparam_effect(df, metric):
    print(f"\n{'='*90}\nMean {metric} by hyperparameter value (per env)\n{'='*90}")
    for h in HPARAMS:
        pivot = (df.groupby(['env', h])[metric]
                   .mean()
                   .unstack('env')
                   .round(4))
        print(f"\n-- {h} --")
        print(pivot.to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv_path', nargs='?',
                    default='log/laprepr_sweep_summary.csv')
    p.add_argument('--env', type=str, default=None,
                    help='restrict to one environment')
    p.add_argument('--metric', type=str, default=None,
                    choices=['overlap_ratio_short', 'overlap_ratio_long',
                             'overlap_ratio_cross'],
                    help='defaults to showing both short and long; pass '
                         'this to see just one (or cross, which is not '
                         'shown by default)')
    p.add_argument('--top', type=int, default=15)
    args = p.parse_args()

    df = load(args.csv_path)
    if args.env:
        df = df[df['env'] == args.env]
        if df.empty:
            sys.exit(f"No rows for env={args.env!r} in {args.csv_path}")

    metrics = [args.metric] if args.metric else \
        ['overlap_ratio_short', 'overlap_ratio_long']
    for metric in metrics:
        show_top_runs(df, metric, args.top)
        show_hyperparam_effect(df, metric)


if __name__ == '__main__':
    main()
