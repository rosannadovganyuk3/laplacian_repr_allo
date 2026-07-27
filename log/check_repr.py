import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_representation_fix():
    # 1. Use the style from the paper
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#EAEAF2"})
    
    # 2. Find ALL representation files
    # Change '.' to the actual path if your logs are elsewhere
    log_root = "./log" 
    
    found_files = []
    for root, _, files in os.walk(log_root):
        if "repr_convergence.csv" in files:
            found_files.append(os.path.join(root, "repr_convergence.csv"))

    if not found_files:
        print(f"❌ No 'repr_convergence.csv' files found in {log_root}!")
        return

    for csv_path in found_files:
        # Extract environment name from path for the title
        env_name = csv_path.split('/')[-3] if '/' in csv_path else "Unknown"
        
        df = pd.read_csv(csv_path)
        
        plt.figure(figsize=(10, 6))
        
        # Plot the drop from 0.9 to 0.25
        plt.plot(df['step'], df['cos_sim'], color='#0000FF', linewidth=2, label='Cosine Similarity')
        
        # Formatting to match the "Box Style"
        plt.title(f"Representation: {env_name}", fontsize=20)
        plt.ylabel("Similarity (Short vs Long)", fontsize=18)
        plt.xlabel("Steps", fontsize=18)
        plt.ylim(-0.1, 1.1) 
        plt.xlim(0, 50000)
        
        output_name = f"repr_plot_{env_name}.png"
        plt.savefig(output_name, dpi=300)
        plt.show()
        print(f"✅ Successfully plotted: {output_name}")

if __name__ == "__main__":
    plot_representation_fix()