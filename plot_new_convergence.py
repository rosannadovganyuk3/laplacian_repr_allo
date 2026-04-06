import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import math

def plot_paper_reproduction(log_root):
    sns.set_theme(style="white", rc={"axes.grid": True, "grid.linestyle": '--'})
    
    # 1. Dynamically find all valid data folders
    valid_data = []
    for root, dirs, files in os.walk(log_root):
        if 'repr_convergence.csv' in files:
            csv_path = os.path.join(root, 'repr_convergence.csv')
            env_name = os.path.basename(os.path.dirname(root)) # Gets 'OneRoom', etc.
            valid_data.append((env_name, csv_path))
    
    if not valid_data:
        print(f"No data found in {log_root}")
        return

    # 2. Calculate Grid Dimensions
    num_plots = len(valid_data)
    cols = min(3, num_plots)
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharey=True)
    # Ensure axes is an array even for 1 plot
    axes = axes.flatten() if num_plots > 1 else [axes]

    # 3. Use a large palette to avoid IndexError
    colors = sns.color_palette("rocket_r", n_colors=max(10, num_plots * 2))

    for i, (env_name, csv_path) in enumerate(valid_data):
        ax = axes[i]
        df = pd.read_csv(csv_path)
        
        # Smooth for the "Paper Look"
        window = 25
        df['smoothed'] = df['cos_sim'].rolling(window=window, min_periods=1).mean()
        df['std'] = df['cos_sim'].rolling(window=window, min_periods=1).std()
        
        # Plotting
        color = colors[i % len(colors)]
        ax.plot(df['step'], df['smoothed'], color=color, linewidth=2)
        ax.fill_between(df['step'], 
                        df['smoothed'] - df['std'], 
                        df['smoothed'] + df['std'], 
                        color=color, alpha=0.15)
        
        # Formatting to match Figure 3
        ax.set_title(env_name, fontsize=14, fontweight='bold')
        ax.set_xlabel("Gradient steps", fontsize=12)
        ax.set_ylim(0, 1.05)
        
        # Academic X-axis
        ax.set_xticks([0, 25000, 50000])
        ax.set_xticklabels(['0', '2.5', '$10^4$']) 

    # Only set Y-label on the leftmost plots
    for j in range(0, num_plots, cols):
        axes[j].set_ylabel("Average cosine similarity", fontsize=12)

    # Hide unused subplots
    for k in range(num_plots, len(axes)):
        axes[k].axis('off')

    sns.despine()
    plt.tight_layout()
    
    output_path = os.path.join(log_root, 'dynamic_paper_grid.png')
    plt.savefig(output_path, dpi=300)
    print(f"Success! Saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', type=str, default='./log/laprepr')
    args = parser.parse_args()
    plot_paper_reproduction(args.log_root)