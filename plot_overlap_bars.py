import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_overlap_bars(csv_path, output_path=None):
    """Simple grouped bar chart of subspace overlap with the true eigenspace:
    ceiling / short / long, one group per environment. Reads the metrics CSV
    evaluate_reprs.py already writes (log/gt_eval/ground_truth_metrics_<run>.csv)."""
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})

    df = pd.read_csv(csv_path)
    envs = df["env"].tolist()
    x = np.arange(len(envs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - 1.5 * width, df["ceiling"], width, label="memoryless ceiling", color="#333333")
    ax.bar(x - 0.5 * width, df["overlap_alias_eig"], width,
           label="eigvecs of transition matrix under aliasing", color="#2ca02c")
    ax.bar(x + 0.5 * width, df["overlap_short"], width, label="short (gamma=0.1)", color="#1f77b4")
    ax.bar(x + 1.5 * width, df["overlap_long"], width, label="long (gamma=0.9)", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=12)
    ax.set_ylabel("Subspace overlap with true eigenspace", fontsize=12)
    ax.set_ylim(0, 1.05)
    run_name = os.path.basename(csv_path).replace("ground_truth_metrics_", "").replace(".csv", "")
    ax.set_title(f"Overlap with true eigenspace  ({run_name})", fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    if output_path is None:
        output_path = f"overlap_bars_{run_name}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=str,
                   help="Path to a ground_truth_metrics_<run>.csv file from evaluate_reprs.py")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    plot_overlap_bars(args.csv_path, args.output)
