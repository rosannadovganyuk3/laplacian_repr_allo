import argparse
import os
import re

import numpy as np

ENVS = ["OneRoom", "TwoRoom", "HardMaze"]
LAGRANGE_MULTS = [0.001, 0.003, 0.03, 0.1, 0.3, 0.5, 0.9]

LOSS_TOTAL_RE = re.compile(r"Step (\d+);.*loss_total ([0-9.eE+-]+)")


def parse_loss_total(log_path):
    """Extract (step, loss_total) pairs from a laprepr log.txt file."""
    steps, losses = [], []
    with open(log_path) as f:
        for line in f:
            m = LOSS_TOTAL_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)


def windowed_table(steps, losses, window):
    total_steps = int(steps.max())
    rows = []
    for start in range(0, total_steps, window):
        mask = (steps > start) & (steps <= start + window)
        if mask.sum() > 0:
            rows.append((start, start + window, losses[mask].mean(), int(mask.sum())))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_prefix", type=str,
                    help="Timestamp prefix identifying the sweep, e.g. 20260802_232246")
    p.add_argument("--log_root", type=str, default="log/laprepr")
    p.add_argument("--window", type=int, default=5000)
    args = p.parse_args()

    for env in ENVS:
        for lm in LAGRANGE_MULTS:
            run_dir = f"{args.log_root}/{env}/{args.run_prefix}_lm{lm}"
            log_path = os.path.join(run_dir, "log.txt")
            if not os.path.isfile(log_path):
                print(f"\n=== {env}  lagrange_mult={lm}  ({args.run_prefix}) ===")
                print(f"[skip] {log_path} not found")
                continue

            steps, losses = parse_loss_total(log_path)
            if len(steps) == 0:
                print(f"\n=== {env}  lagrange_mult={lm}  ({args.run_prefix}) ===")
                print(f"[skip] no loss_total entries found in {log_path}")
                continue

            rows = windowed_table(steps, losses, args.window)

            print(f"\n=== {env}  lagrange_mult={lm}  ({args.run_prefix}) ===")
            print(f"{'steps':<16}{'avg loss_total':<16}{'n':<5}")
            for start, end, avg, n in rows:
                print(f"{start:>6}-{end:<8} {avg:<16.4f}{n:<5}")


if __name__ == "__main__":
    main()
