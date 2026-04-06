import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_convergence(log_dir, window_size=5):
    sns.set_theme(style="whitegrid")
    csv_path = os.path.join(log_dir, 'repr_convergence.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("File is empty. Wait for the first few training steps to finish!")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # 1. Plot the raw data (faded)
    plt.plot(df['step'], df['cos_sim'], alpha=0.3, color='royalblue', label='Raw Batch Sim')
    
    # 2. Plot the smoothed trend
    df['smoothed'] = df['cos_sim'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(df['step'], df['smoothed'], color='blue', linewidth=2, label='Trend (Moving Avg)')

    # Add visual guides for interpretation
    plt.axhspan(0.4, 0.7, color='green', alpha=0.1, label='Expected Specialization Zone')
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Identical (No Specialization)')

    # Formatting
    plt.title(f"Representation Convergence\n{log_dir}", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Cosine Similarity (Short vs Long)", fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    
    # Save the result
    output_path = os.path.join(log_dir, 'convergence_plot.png')
    plt.savefig(output_path)
    plt.show()
    print(f"Success! Plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Change default to the root log folder
    parser.add_argument('--log_root', type=str, default='./log/laprepr',
                        help='Root directory containing all environment folders')
    args = parser.parse_args()
    
    # Automatically find all 'repr_convergence.csv' files in subfolders
    for root, dirs, files in os.walk(args.log_root):
        if 'repr_convergence.csv' in files:
            print(f"\n--- Plotting for Environment: {os.path.basename(root)} ---")
            plot_convergence(root)