import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_mse_and_cos_sim():
    """Walks ./log/laprepr for repr_convergence.csv files and plots mse and
    cos_sim side by side over training steps, one figure per run folder found
    (matches check_repr.py's file-discovery style, extended for multiple runs
    per environment now that each SLURM submission gets its own folder)."""
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
    log_root = "./log/laprepr"

    found_files = []
    for root, _, files in os.walk(log_root):
        if "repr_convergence.csv" in files:
            found_files.append(os.path.join(root, "repr_convergence.csv"))

    if not found_files:
        print(f"No 'repr_convergence.csv' files found under {log_root}!")
        return

    for csv_path in found_files:
        parts = csv_path.split(os.sep)
        env_name = parts[-3] if len(parts) >= 3 else "Unknown"
        run_name = parts[-2] if len(parts) >= 2 else "unknown_run"

        df = pd.read_csv(csv_path)
        if "mse" not in df.columns:
            print(f"[skip] {csv_path}: no 'mse' column (run predates this diagnostic)")
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        ax1.plot(df["step"], df["cos_sim"], color="#0000FF", linewidth=2)
        ax1.set_title("Cosine Similarity (Short vs Long)", fontsize=14)
        ax1.set_xlabel("Steps", fontsize=12)
        ax1.set_ylabel("cos_sim", fontsize=12)
        ax1.set_ylim(-1.05, 1.05)

        ax2.plot(df["step"], df["mse"], color="#D62728", linewidth=2)
        ax2.set_title("MSE (Short vs Long)", fontsize=14)
        ax2.set_xlabel("Steps", fontsize=12)
        ax2.set_ylabel("mse", fontsize=12)

        fig.suptitle(f"{env_name}  ({run_name})", fontsize=16)
        plt.tight_layout()

        output_name = f"mse_plot_{env_name}_{run_name}.png"
        plt.savefig(output_name, dpi=200)
        plt.close()
        print(f"Saved: {output_name}")


if __name__ == "__main__":
    plot_mse_and_cos_sim()
