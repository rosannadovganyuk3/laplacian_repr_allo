import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_exact_box_style(window_size=15):
    envs = ['OneRoom', 'TwoRoom', 'HardMaze']
    
    # Matching your folder structure: 'laprepr' is the folder you are using
    method_configs = {
        'laprepr': {'label': 'mix',    'color': '#0000FF', 'ls': '-',  'lw': 3}, # Bold Blue
        'l2':      {'label': 'l2',     'color': '#00CED1', 'ls': '--', 'lw': 3}, # Cyan
        'rawmix':  {'label': 'rawmix', 'color': '#BCBD22', 'ls': '--', 'lw': 3}, # Olive
        'sparse':  {'label': 'sparse', 'color': '#FF0000', 'ls': '--', 'lw': 3}, # Red
    }

    # 1. THE EXACT BOX SHADING (Lavender/Gray background)
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#EAEAF2", 
        "grid.color": "white",       
        "grid.linestyle": "-",
        "axes.edgecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    })

    all_csv_paths = []
    for root, _, files in os.walk("./log"):
        for file in files:
            if file == "repr_convergence.csv":
                all_csv_paths.append(os.path.join(root, file))

    for env in envs:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        found_any = False
        
        for folder, cfg in method_configs.items():
            # Filters for the method folder and the room name
            files = [f for f in all_csv_paths if f"/{folder}/" in f and env in f]
            if not files: continue
                
            all_runs = []
            for f in files:
                # Read with headers because your CSV has them
                df = pd.read_csv(f)
                
                # TRANSFORMATION: (1 - similarity) makes the curve climb as it improves
                # This makes your specialization drop look like the Paper's success rate.
                specialization = 1 - df['cos_sim']
                s_rate = specialization.rolling(window=window_size, min_periods=1).mean()
                
                all_runs.append(pd.DataFrame({'steps': df['step'], 'success': s_rate}))
                found_any = True

            if all_runs:
                combined = pd.concat(all_runs)
                # Shaded error bars and bold lines exactly like your target image
                sns.lineplot(
                    data=combined, x='steps', y='success',
                    label=cfg['label'], color=cfg['color'],
                    linestyle=cfg['ls'], linewidth=cfg['lw'],
                    errorbar=('ci', 95), ax=ax,
                    alpha=1.0 
                )

        if found_any:
            # 2. EXACT LABELS & SIZES
            ax.set_title(f"{env} environment", fontsize=26, pad=10)
            ax.set_ylabel("specialization rate", fontsize=24) # Changed label to reflect metric
            ax.set_xlabel("training steps", fontsize=24)
            
            ax.tick_params(labelsize=20)
            
            # 3. AXIS LIMITS (Expanded to 50k to show your full success)
            ax.set_ylim(-0.1, 1.1) 
            ax.set_xlim(-1000, 51000)
            
            # 4. LEGEND: Bottom Right, Large, No Frame
            ax.legend(loc='lower right', fontsize=22, frameon=False)

            plt.tight_layout()
            # Save with a clear name
            save_name = f"final_success_style_{env}.png"
            plt.savefig(save_name, dpi=300)
            print(f"✅ Exact style plot generated: {save_name}")
        plt.close()

if __name__ == "__main__":
    plot_exact_box_style()