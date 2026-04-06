"""Plot training losses to check convergence."""
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def parse_log_file(log_path):
    """Extract loss values from log file."""
    steps = []
    loss_pos_short = []
    loss_pos_long = []
    loss_neg_short = []
    loss_neg_long = []
    loss_total = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Look for lines with loss information
            # Adjust this regex based on your actual log format
            if 'Step' in line and 'loss' in line:
                # Extract step number
                step_match = re.search(r'Step (\d+)', line)
                if step_match:
                    step = int(step_match.group(1))
                    steps.append(step)
                    
                    # Extract loss values (adjust regex to match your format)
                    pos_short = re.search(r'loss_pos_short[:\s=]+([0-9.e-]+)', line)
                    pos_long = re.search(r'loss_pos_long[:\s=]+([0-9.e-]+)', line)
                    neg_short = re.search(r'loss_neg_short[:\s=]+([0-9.e-]+)', line)
                    neg_long = re.search(r'loss_neg_long[:\s=]+([0-9.e-]+)', line)
                    total = re.search(r'loss_total[:\s=]+([0-9.e-]+)', line)
                    
                    loss_pos_short.append(float(pos_short.group(1)) if pos_short else np.nan)
                    loss_pos_long.append(float(pos_long.group(1)) if pos_long else np.nan)
                    loss_neg_short.append(float(neg_short.group(1)) if neg_short else np.nan)
                    loss_neg_long.append(float(neg_long.group(1)) if neg_long else np.nan)
                    loss_total.append(float(total.group(1)) if total else np.nan)
    
    return {
        'steps': np.array(steps),
        'loss_pos_short': np.array(loss_pos_short),
        'loss_pos_long': np.array(loss_pos_long),
        'loss_neg_short': np.array(loss_neg_short),
        'loss_neg_long': np.array(loss_neg_long),
        'loss_total': np.array(loss_total)
    }

def plot_training_curves(env_id, output_dir='log/training_analysis'):
    """Plot training losses for an environment."""
    log_path = f'log/laprepr/{env_id}/test/log.txt'
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    
    print(f"Parsing log file: {log_path}")
    losses = parse_log_file(log_path)
    
    if len(losses['steps']) == 0:
        print(f"No loss data found in {log_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Positive losses
    axes[0].plot(losses['steps'], losses['loss_pos_short'], label='Short-term', alpha=0.7)
    axes[0].plot(losses['steps'], losses['loss_pos_long'], label='Long-term', alpha=0.7)
    axes[0].set(xlabel='Training Steps', ylabel='Positive Loss',
                title=f'{env_id} - Positive Loss (Temporal Pairs)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Negative losses
    axes[1].plot(losses['steps'], losses['loss_neg_short'], label='Short-term', alpha=0.7)
    axes[1].plot(losses['steps'], losses['loss_neg_long'], label='Long-term', alpha=0.7)
    axes[1].set(xlabel='Training Steps', ylabel='Negative Loss',
                title=f'{env_id} - Negative Loss (Random Samples)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Total loss
    axes[2].plot(losses['steps'], losses['loss_total'], label='Total', color='black', alpha=0.7)
    axes[2].set(xlabel='Training Steps', ylabel='Total Loss',
                title=f'{env_id} - Total Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{env_id}_training_losses.png'), 
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/{env_id}_training_losses.png")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY - {env_id}")
    print(f"{'='*60}")
    print(f"Total steps: {losses['steps'][-1]}")
    print(f"\nFinal losses (last 5 steps average):")
    print(f"  Pos short: {np.nanmean(losses['loss_pos_short'][-5:]):.6f}")
    print(f"  Pos long:  {np.nanmean(losses['loss_pos_long'][-5:]):.6f}")
    print(f"  Neg short: {np.nanmean(losses['loss_neg_short'][-5:]):.6f}")
    print(f"  Neg long:  {np.nanmean(losses['loss_neg_long'][-5:]):.6f}")
    print(f"  Total:     {np.nanmean(losses['loss_total'][-5:]):.6f}")
    
    print(f"\nInitial losses (first 5 steps average):")
    print(f"  Pos short: {np.nanmean(losses['loss_pos_short'][:5]):.6f}")
    print(f"  Pos long:  {np.nanmean(losses['loss_pos_long'][:5]):.6f}")
    print(f"  Total:     {np.nanmean(losses['loss_total'][:5]):.6f}")
    
    print(f"\nConvergence check:")
    # Check if loss is still decreasing significantly
    last_1000 = losses['loss_total'][losses['steps'] > losses['steps'][-1] - 1000]
    if len(last_1000) > 10:
        recent_std = np.nanstd(last_1000)
        recent_mean = np.nanmean(last_1000)
        if recent_std / recent_mean < 0.05:
            print(f"  ✅ Converged (std/mean = {recent_std/recent_mean:.3f} < 0.05)")
        else:
            print(f"  ⚠️  Still changing (std/mean = {recent_std/recent_mean:.3f} >= 0.05)")
    print(f"{'='*60}\n")

def main():
    """Plot training curves for all environments."""
    envs = ['OneRoom', 'TwoRoom', 'HardMaze']
    
    for env_id in envs:
        print(f"\nProcessing {env_id}...")
        plot_training_curves(env_id)
    
    print("\nAll training curves saved to log/training_analysis/")

if __name__ == '__main__':
    main()