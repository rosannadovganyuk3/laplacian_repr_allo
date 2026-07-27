"""Visualize learned representation and compare with Laplacian Ground Truth."""
import os
import argparse
import importlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.stats import spearmanr
from scipy.linalg import eigh 

from rl_lap.agent import laprepr
from rl_lap.tools import flag_tools
from rl_lap.tools import torch_tools

# Plotting parameters
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

parser = argparse.ArgumentParser()
parser.add_argument('--log_base_dir', type=str, 
        default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_sub_dir', type=str, 
        default='laprepr/HardMaze/test')
parser.add_argument('--output_sub_dir', type=str, 
        default='visualize_reprs')
parser.add_argument('--config_dir', type=str, default='rl_lap.configs')
parser.add_argument('--config_file', 
        type=str, default='laprepr_config_gridworld')
FLAGS = parser.parse_args()

def get_config_cls():
    config_module = importlib.import_module(FLAGS.config_dir+'.'+FLAGS.config_file)
    return config_module.Config

def get_robust_maze_layout(env, pos_batch):
    """Finds the maze grid regardless of attribute name and converts it to numeric."""
    if hasattr(env.task.maze, 'grid'):
        layout = env.task.maze.grid
    elif hasattr(env.task.maze, '_maze'):
        layout = env.task.maze._maze
    elif hasattr(env.task.maze, 'matrix'):
        layout = env.task.maze.matrix
    else:
        # Fallback: create a grid from the empty positions we already have
        max_r = pos_batch[:, 0].max() + 1
        max_c = pos_batch[:, 1].max() + 1
        layout = np.ones((max_r + 1, max_c + 1)) 
        layout[pos_batch[:, 0], pos_batch[:, 1]] = 0 

    layout = np.array(layout)
    # Handle string-based mazes (like HardMaze with '+', '-', '|')
    if layout.dtype.kind in {'U', 'S'}:
        numeric_layout = np.where((layout == ' ') | (layout == '.'), 0.0, 1.0)
    else:
        numeric_layout = layout.astype(float)
    return numeric_layout

def compute_laplacian_eigenvectors(env, n_eigenvectors):
    """Build transition matrix and compute Laplacian eigendecomposition"""
    pos_batch = env.task.maze.all_empty_grids()
    n_states = len(pos_batch)
    pos_to_idx = {tuple(pos): i for i, pos in enumerate(pos_batch)}

    # Build Adjacency Matrix (Graph skeleton)
    A = np.zeros((n_states, n_states))
    for i, pos in enumerate(pos_batch):
        for move in [(0,1), (0,-1), (1,0), (-1,0)]:
            next_p = tuple(pos + move)
            if next_p in pos_to_idx:
                A[i, pos_to_idx[next_p]] = 1

    # Laplacian: L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Solve for smallest eigenvectors (skip constant 0th)
    eigenvalues, eigenvectors = eigh(L, subset_by_index=[1, n_eigenvectors])
    return eigenvectors, eigenvalues, A

def compute_correlation(learned_reprs, true_eigenvectors):
    """Compute alignment between learned and true eigenvectors."""
    n_dims = min(learned_reprs.shape[1], true_eigenvectors.shape[1])
    correlations = []
    for i in range(n_dims):
        # Handle collapse: check if dimension has variance
        if np.std(learned_reprs[:, i]) > 1e-6:
            best_corr = max(abs(spearmanr(learned_reprs[:, i],
                                         true_eigenvectors[:, j])[0])
                            for j in range(n_dims))
        else:
            best_corr = 0.0 # Dimension is constant/collapsed
        correlations.append(best_corr)
    return np.array(correlations)

def visualize_representation(model, states_batch, goal_state, pos_batch, goal_pos, 
                            env, device, output_dir, filename, title):
    """Visualize L2 distances from all states to the goal state."""
    states_torch = torch_tools.to_tensor(states_batch, device)
    goal_torch = torch_tools.to_tensor(goal_state, device)
    
    with torch.no_grad():
        states_reprs = model(states_torch).detach().cpu().numpy()
        goal_repr = model(goal_torch).detach().cpu().numpy()
    
    l2_dists = np.sqrt(np.sum(np.square(states_reprs - goal_repr), axis=-1))
    numeric_layout = get_robust_maze_layout(env, pos_batch)
    
    grid_h, grid_w = numeric_layout.shape[:2]
    map_ = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    map_[pos_batch[:, 0], pos_batch[:, 1]] = l2_dists
    
    plt.figure(figsize=(6, 5))
    im = plt.imshow(map_, interpolation='none', cmap='Blues_r')
    plt.colorbar(label='L2 Distance to Goal')
    plt.contour(numeric_layout, levels=[0.5], colors='black', linewidths=1.5)
    plt.scatter(goal_pos[1], goal_pos[0], color='red', marker='*', s=150, label='Goal')
    
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), bbox_inches='tight', dpi=150)
    plt.close()
    return states_reprs

def visualize_representation_dimensions(states_reprs, pos_batch, env, output_dir, filename_prefix, n_dims_to_show=5):
    """Visualize individual embedding dimensions as heatmaps (Eigenmaps)."""
    numeric_layout = get_robust_maze_layout(env, pos_batch)
    grid_h, grid_w = numeric_layout.shape[:2]
    n_dims = min(n_dims_to_show, states_reprs.shape[1])
    
    fig, axes = plt.subplots(1, n_dims, figsize=(4*n_dims, 4))
    if n_dims == 1: axes = [axes]
    
    for i in range(n_dims):
        dim_values = states_reprs[:, i]
        map_ = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
        map_[pos_batch[:, 0], pos_batch[:, 1]] = dim_values
        
        im = axes[i].imshow(map_, cmap='RdBu', interpolation='none')
        axes[i].contour(numeric_layout, levels=[0.5], colors='black', linewidths=1)
        axes[i].set_title(f'Dim {i}')
        axes[i].set_xticks([]); axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}_dimensions.png'), bbox_inches='tight', dpi=150)
    plt.close()

def plot_comparison(corr_short, corr_long, eigenvalues, env_id, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    x = np.arange(len(corr_short))
    axes[0].bar(x - 0.2, corr_short, width=0.4, label='Short-term', alpha=0.7)
    axes[0].bar(x + 0.2, corr_long,  width=0.4, label='Long-term',  alpha=0.7)
    axes[0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target threshold')
    axes[0].set(xlabel='Dimension', ylabel='Correlation with GT Laplacian', title=f'{env_id} - Spectral Alignment')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(eigenvalues, 'o-', markersize=4)
    axes[1].set(xlabel='Eigenvector Index', ylabel='Eigenvalue', title='GT Laplacian Spectrum', yscale='log')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{env_id}_gt_comparison.png'), bbox_inches='tight', dpi=150)
    plt.close()

def main():
    log_dir = os.path.join(FLAGS.log_base_dir, FLAGS.log_sub_dir)
    output_dir = os.path.join(FLAGS.log_base_dir, FLAGS.output_sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    flags = flag_tools.load_flags(log_dir)
    cfg = get_config_cls()(flags)
    learner_args = cfg.args_as_flags
    device = learner_args.device

    model_short = learner_args.model_cfg.model_factory().to(device)
    model_long = learner_args.model_cfg.model_factory().to(device)
    model_short.load_state_dict(torch.load(os.path.join(log_dir, 'model_short.ckpt')))
    model_long.load_state_dict(torch.load(os.path.join(log_dir, 'model_long.ckpt')))

    env = learner_args.env_factory()
    obs_prepro = learner_args.obs_prepro
    pos_batch = env.task.maze.all_empty_grids()
    obs_batch = [env.task.pos_to_obs(p) for p in pos_batch]
    states_batch = np.array([obs_prepro(o) for o in obs_batch])
    
    goal_pos = env.task.goal_pos
    goal_state = obs_prepro(env.task.pos_to_obs(goal_pos))[None]

    print(f"\nComputing GT Laplacian for {flags.env_id}...")
    n_dims = flags.d
    true_evecs, evals, _ = compute_laplacian_eigenvectors(env, n_dims)

    reprs_short = visualize_representation(model_short, states_batch, goal_state, pos_batch, goal_pos, env, device, output_dir, f'{flags.env_id}_short', "Short-term Representation")
    corr_short = compute_correlation(reprs_short, true_evecs)

    reprs_long = visualize_representation(model_long, states_batch, goal_state, pos_batch, goal_pos, env, device, output_dir, f'{flags.env_id}_long', "Long-term Representation")
    corr_long = compute_correlation(reprs_long, true_evecs)

    plot_comparison(corr_short, corr_long, evals, flags.env_id, output_dir)
    visualize_representation_dimensions(reprs_short, pos_batch, env, output_dir, f'{flags.env_id}_short_dims')
    
    print(f"\nSummary - {flags.env_id}")
    print(f"Short-term Alignment: {np.mean(corr_short):.3f}")
    print(f"Long-term Alignment:  {np.mean(corr_long):.3f}")

if __name__ == '__main__':
    main()