import argparse
import glob
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

LM_RE = re.compile(r"_lm([0-9.]+)\.csv$")

REF_STYLE = dict(memoryless_ceiling=dict(color="#555555", ls="--", marker="s",
                                          label="memoryless ceiling"),
                  overlap_alias_eig=dict(color="#555555", ls=":", marker="^",
                                          label="eigvecs of transition matrix under aliasing"))
SERIES_STYLE = dict(overlap_short=dict(color="#1f77b4", ls="-", marker="o",
                                        label="short (gamma=0.1)"),
                     overlap_long=dict(color="#ff7f0e", ls="-", marker="o",
                                        label="long (gamma=0.9)"))


def load_sweep(csv_glob):
    """One row per (env, lagrange_mult) from a set of ground_truth_metrics_*_lm<value>.csv
    files -- each file is a single evaluate_reprs.py run at a fixed lagrange_mult."""
    rows = []
    for path in sorted(glob.glob(csv_glob)):
        m = LM_RE.search(path)
        if not m:
            continue
        lm = float(m.group(1))
        df = pd.read_csv(path)
        df["lagrange_mult"] = lm
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no files matched {csv_glob}")
    return pd.concat(rows, ignore_index=True)


def plot_lambda_sweep(csv_glob, output_path=None):
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})

    df = load_sweep(csv_glob)
    df = df.rename(columns={"ceiling": "memoryless_ceiling"})
    envs = sorted(df["env"].unique(), key=lambda e: df[df.env == e]["n_states"].iloc[0])

    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4.5), sharey=True)
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        d = df[df.env == env].sort_values("lagrange_mult")
        for col, style in {**REF_STYLE, **SERIES_STYLE}.items():
            ax.plot(d["lagrange_mult"], d[col], markersize=6, linewidth=2, **style)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(sorted(d["lagrange_mult"].unique()))
        ax.set_title(env, fontsize=12)
        ax.set_xlabel("lagrange_mult", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=.3)

    axes[0].set_ylabel("Subspace overlap with true eigenspace", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=4, fontsize=9, frameon=False)
    plt.tight_layout()

    if output_path is None:
        output_path = "overlap_lambda_sweep.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_glob", type=str,
                   help="Glob matching ground_truth_metrics_*_lm<value>.csv files, "
                        "e.g. 'log/gt_eval/ground_truth_metrics_20260802_232246_lm*.csv'")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    plot_lambda_sweep(args.csv_glob, args.output)
