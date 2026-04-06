import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_exact_box_style(window_size=30):
    envs = ['OneRoom', 'TwoRoom', 'HardMaze']
    
    # EXACT colors from your second image
    method_configs = {
        'mix':    {'label': 'mix',    'color': '#0000FF', 'ls': '-',  'lw': 3}, # Bold Blue
        'l2':     {'label': 'l2',     'color': '#00CED1', 'ls': '--', 'lw': 3}, # Cyan
        'rawmix': {'label': 'rawmix', 'color': '#BCBD22', 'ls': '--', 'lw': 3}, # Olive
        'sparse': {'label': 'sparse', 'color': '#FF0000', 'ls': '--', 'lw': 3}, # Red
    }

    # 1. THE BOX SHADING: Use 'darkgrid' for the blue-gray background + white lines
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#EAEAF2", # This is the exact Seaborn lavender/gray shade
        "grid.color": "white",       # White grid lines inside the gray box
        "grid.linestyle": "-",
        "axes.edgecolor": "white",
        "font.family": "sans-serif", # NO Times New Roman
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    })

    all_csv_paths = []
    for root, _, files in os.walk("./log"):
        for file in files:
            if file == "results.csv":
                all_csv_paths.append(os.path.join(root, file))

    for env in envs:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        found_any = False
        
        for folder, cfg in method_configs.items():
            files = [f for f in all_csv_paths if f"/{folder}/" in f and env in f]
            if not files: continue
                
            all_runs = []
            for f in files:
                df = pd.read_csv(f, header=None)
                success = (df.iloc[:, 1] > -35.0).astype(float) # changed from -20.0 to -35.0
                s_rate = success.rolling(window=window_size, min_periods=1).mean()
                all_runs.append(pd.DataFrame({'steps': df.iloc[:, 0], 'success': s_rate}))
                found_any = True

            if all_runs:
                combined = pd.concat(all_runs)
                # Shaded error bars with the same color as the line
                sns.lineplot(
                    data=combined, x='steps', y='success',
                    label=cfg['label'], color=cfg['color'],
                    linestyle=cfg['ls'], linewidth=cfg['lw'],
                    errorbar=('ci', 95), ax=ax,
                    alpha=1.0 # Keep lines bold
                )

        if found_any:
            # 2. LABELS & SIZES
            ax.set_title(f"{env} environment", fontsize=26, pad=10)
            ax.set_ylabel("test success rate", fontsize=24)
            ax.set_xlabel("training steps", fontsize=24)
            
            # Tick parameters for large, readable numbers
            ax.tick_params(labelsize=20)
            
            # 3. AXIS LIMITS matching the box image
            ax.set_ylim(-0.6, 1.2) 
            ax.set_xlim(-1000, 31000)
            
            # 4. LEGEND: Bottom Right, Large, No Frame
            ax.legend(loc='lower right', fontsize=22, frameon=False)

            plt.tight_layout()
            plt.savefig(f"box_style_{env}.png", dpi=300)
            print(f"✅ Box-style plot generated for {env}")
        plt.close()

if __name__ == "__main__":
    plot_exact_box_style()