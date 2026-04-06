"""Visualize learned representation."""
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
        default='laprepr/OneRoom/test')
parser.add_argument('--output_sub_dir', type=str, 
        default='visualize_reprs')
parser.add_argument('--config_dir', type=str, default='rl_lap.configs')
parser.add_argument('--config_file', 
        type=str, default='laprepr_config_gridworld')
FLAGS = parser.parse_args()


def get_config_cls():
    config_module = importlib.import_module(
            FLAGS.config_dir+'.'+FLAGS.config_file)
    config_cls = config_module.Config
    return config_cls


def compute_laplacian_eigenvectors(env, n_eigenvectors):
    """Build transition matrix and compute Laplacian eigendecomposition"""
    pos_batch = env.task.maze.all_empty_grids() # gets all valid grid positions in the maze as an array of (row, col) coord.
    n_states = len(pos_batch) # total num of valid states in maze
    pos_to_idx = {(r,c): i for i, (r,c) in enumerate(pos_batch)} # reverse lookup dictionary, given a pos what is its index

    # build uniform random policy transition matrix
    P = np.zeros((n_states, n_states)) # initialize the transition matrix as all zeros
    
    # loop over every state/action to f
    num_actions = env.action_spec.n #len(env.task._action_map) 
    for i, pos in enumerate(pos_batch):
        for action in range(num_actions):
            # 1. Get the direction (delta) for this action
            delta = env.task._action_map[action]
            
            # 2. Calculate the potential next position
            next_pos = tuple(pos + delta)
            
            # 3. Check if the next square is a wall (1) or a path (0)
            # If it's a wall, the agent stays in the current 'pos'
            if env.task._maze[next_pos] == 1:
                next_pos = tuple(pos)

            # 4. Look up the index and update transition matrix
            j = pos_to_idx.get(next_pos)
            # returns none if its a wall or out of bounds
            # adds equal probability for each action if next pos is valid
            if j is not None:
                P[i, j] += 1.0 / env.action_spec.n

    # Graph Laplacian and eigendecomposition
    # P.sum(axis=1) gives row sums (total prob. per state)
    D = np.diag(P.sum(axis=1)) # degree matrix
    L = D - P # Laplacian
    eigenvalues, eigenvectors = eigh(L)

    return eigenvectors[:, :n_eigenvectors], eigenvalues[:n_eigenvectors], P 


def compute_correlation(learned_reprs, true_eigenvectors):
    """Compute alignment between learned and true eigenvectors."""
    n_dims = min(learned_reprs.shape[1], true_eigenvectors.shape[1])
    correlations = []
    for i in range(n_dims):
        best_corr = max(abs(spearmanr(learned_reprs[:, i],
                                      true_eigenvectors[:, j])[0])
                        for j in range(n_dims))
        correlations.append(best_corr)
    return np.array(correlations)


def visualize_representation(model, states_batch, goal_state, pos_batch, goal_pos, 
                            env, obs_prepro, device, output_dir, filename, title):
    """Helper function to visualize one model's representations"""
    # get representations from loaded model
    states_torch = torch_tools.to_tensor(states_batch, device)
    goal_torch = torch_tools.to_tensor(goal_state, device)
    states_reprs = model(states_torch).detach().cpu().numpy()
    goal_repr = model(goal_torch).detach().cpu().numpy()
    
    # compute l2 distances from states to goal
    l2_dists = np.sqrt(np.sum(np.square(states_reprs - goal_repr), axis=-1))
    
    # -- visualize state representations --
    # plot raw distances with the walls
    goal_obs = env.task.pos_to_obs(goal_pos)
    image_shape = goal_obs.agent.image.shape
    map_ = np.zeros(image_shape[:2], dtype=np.float32)
    map_[pos_batch[:, 0], pos_batch[:, 1]] = l2_dists
    im_ = plt.imshow(map_, interpolation='none', cmap='Blues')
    plt.colorbar()
    
    # add the walls to the normalized distance plot
    walls = np.expand_dims(env.task.maze.render(), axis=-1)
    map_2 = im_.cmap(im_.norm(map_))
    map_2[:, :, :-1] = map_2[:, :, :-1] * (1 - walls) + 0.5 * walls
    map_2[:, :, -1:] = map_2[:, :, -1:] * (1 - walls) + 1.0 * walls
    map_2[goal_pos[0], goal_pos[1]] = [1, 0, 0, 1]
    
    plt.cla()
    plt.imshow(map_2, interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    
    figfile = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(figfile, bbox_inches='tight')
    plt.clf()
    print(f"Saved visualization: {figfile}")

    return states_reprs  # return for ground truth comparison


def plot_comparison(corr_short, corr_long, eigenvalues, env_id, output_dir):
    """Plot alignment bar chart and eigenvalue spectrum."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    x = np.arange(len(corr_short))
    axes[0].bar(x - 0.2, corr_short, width=0.4, label='Short-term', alpha=0.7)
    axes[0].bar(x + 0.2, corr_long,  width=0.4, label='Long-term',  alpha=0.7)
    axes[0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Good threshold')
    axes[0].set(xlabel='Dimension', ylabel='Correlation with Ground Truth',
                title=f'{env_id} - Alignment with Laplacian Eigenvectors')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(eigenvalues, 'o-', markersize=4)
    axes[1].set(xlabel='Eigenvector Index', ylabel='Eigenvalue',
                title='Ground Truth Eigenvalue Spectrum', yscale='log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{env_id}_ground_truth_comparison.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {env_id}_ground_truth_comparison.png")

def save_results(corr_short, corr_long, env_id, n_dims, output_dir):
    """Write alignment scores to a text file."""
    def section(f, label, corrs):
        f.write(f"\n{'='*60}\n{label}\n{'='*60}\n")
        f.write(f"Mean: {np.mean(corrs):.4f} | Std: {np.std(corrs):.4f} | "
                f"Min: {np.min(corrs):.4f} | Max: {np.max(corrs):.4f}\n")
        f.write("Per-dimension:\n")
        for i, c in enumerate(corrs):
            f.write(f"  Dim {i:2d}: {c:.4f}\n")
    
    with open(os.path.join(output_dir, f'{env_id}_alignment_scores.txt'), 'w') as f:
        f.write(f"Environment: {env_id} | Dimensions: {n_dims}\n")
        section(f, "SHORT-TERM MODEL (discount=0.9)", corr_short)
        section(f, "LONG-TERM MODEL  (discount=0.1)", corr_long)
        
        f.write(f"\n{'='*60}\nCOMPARISON\n{'='*60}\n")
        diff = np.mean(corr_short) - np.mean(corr_long)
        f.write(f"Difference: {diff:+.4f} | Better: "
                f"{'Short-term' if diff > 0 else 'Long-term'}\n")
    
    print(f"Saved: {env_id}_alignment_scores.txt")

def quality_label(score):
    """Return quality descriptor for correlation score."""
    return ('Excellent' if score > 0.8 else 'Good' if score > 0.6 else 
            'Moderate' if score > 0.4 else 'Poor')

def print_summary(env_id, corr_short, corr_long):
    """Print concise summary to console."""
    print(f"\n{'='*60}")
    print(f"SUMMARY - {env_id}")
    print(f"{'='*60}")
    print(f"Short-term: {np.mean(corr_short):.3f} ({quality_label(np.mean(corr_short))})")
    print(f"Long-term:  {np.mean(corr_long):.3f} ({quality_label(np.mean(corr_long))})")
    print(f"{'='*60}\n")

# rep
def visualize_representation_dimensions(states_reprs, pos_batch, env, output_dir, filename_prefix, n_dims_to_show=5):
    """Visualize individual representation dimensions as heatmaps."""
    image_shape = env.task.maze.render().shape
    n_dims = min(n_dims_to_show, states_reprs.shape[1])
    
    fig, axes = plt.subplots(1, n_dims, figsize=(4*n_dims, 4))
    if n_dims == 1:
        axes = [axes]
    
    for i in range(n_dims):
        # Get values for dimension i across all states
        dim_values = states_reprs[:, i]
        
        # Create heatmap
        map_ = np.zeros(image_shape[:2], dtype=np.float32)
        map_[pos_batch[:, 0], pos_batch[:, 1]] = dim_values
        
        # Add walls
        walls = np.expand_dims(env.task.maze.render(), axis=-1)
        im = axes[i].imshow(map_, cmap='RdBu', interpolation='none')
        
        # Overlay walls
        axes[i].contour(walls[:, :, 0], levels=[0.5], colors='gray', linewidths=2)
        
        axes[i].set_title(f'Dimension {i}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}_dimensions.png'), 
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {filename_prefix}_dimensions.png")


def main():
    # setup log directories
    log_dir = os.path.join(FLAGS.log_base_dir, FLAGS.log_sub_dir)
    output_dir = os.path.join(FLAGS.log_base_dir, FLAGS.output_sub_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load config
    flags = flag_tools.load_flags(log_dir)
    cfg_cls = get_config_cls()
    cfg = cfg_cls(flags)
    learner_args = cfg.args_as_flags
    device = learner_args.device
    #n_dims = flags.d

    # load both models from checkpoints
    model_short = learner_args.model_cfg.model_factory()
    model_long = learner_args.model_cfg.model_factory()

    model_short.to(device=device)
    model_long.to(device=device)

    ckpt_path_short = os.path.join(log_dir, 'model_short.ckpt')
    ckpt_path_long = os.path.join(log_dir, 'model_long.ckpt')

    model_short.load_state_dict(torch.load(ckpt_path_short))
    model_long.load_state_dict(torch.load(ckpt_path_long))

    print(f"Loaded short-term model from: {ckpt_path_short}")
    print(f"Loaded long-term model from: {ckpt_path_long}")

    # -- use loaded models to get state representations --
    # get the full batch of states from env
    env = learner_args.env_factory()
    obs_prepro = learner_args.obs_prepro
    n_states = env.task.maze.n_states
    pos_batch = env.task.maze.all_empty_grids()
    obs_batch = [env.task.pos_to_obs(pos_batch[i]) for i in range(n_states)]
    states_batch = np.array([obs_prepro(obs) for obs in obs_batch])

    # get goal state representation
    goal_pos = env.task.goal_pos
    goal_obs = env.task.pos_to_obs(goal_pos)
    goal_state = obs_prepro(goal_obs)[None]

    # Compute ground truth
    print("\nComputing ground truth Laplacian eigenvectors...")
    n_dims = flags.d #learner_args.model_cfg.d
    true_eigenvectors, eigenvalues, transition_matrix = compute_laplacian_eigenvectors(env, n_dims)
    print(f"Eigenvalue range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")

    # -- get representations from loaded models --
    # visualize short-term model
    reprs_short = visualize_representation(
        model_short, states_batch, goal_state, pos_batch, goal_pos,
        env, obs_prepro, device, output_dir,
        filename=f'{flags.env_id}_short_term',
        title=f'{flags.env_id} - Short-term (discount=0.9)'
    )
    corr_short = compute_correlation(reprs_short, true_eigenvectors)
    
    # visualize long-term model
    reprs_long = visualize_representation(
        model_long, states_batch, goal_state, pos_batch, goal_pos,
        env, obs_prepro, device, output_dir,
        filename=f'{flags.env_id}_long_term',
        title=f'{flags.env_id} - Long-term (discount=0.1)'
    )
    corr_long = compute_correlation(reprs_long, true_eigenvectors)

    print("\nVisualization complete!")
    print(f"Output directory: {output_dir}")

    # Create comparison plots and save results
    print("\nCreating comparison plots...")
    plot_comparison(corr_short, corr_long, eigenvalues, flags.env_id, output_dir)
    save_results(corr_short, corr_long, flags.env_id, n_dims, output_dir)
    
    # Print summary
    print_summary(flags.env_id, corr_short, corr_long)
    print(f"All outputs saved to: {output_dir}")

        # After getting representations
    print("\nVisualizing individual dimensions...")
    visualize_representation_dimensions(
        reprs_short, pos_batch, env, output_dir,
        f'{flags.env_id}_short_term', n_dims_to_show=5
    )
    visualize_representation_dimensions(
        reprs_long, pos_batch, env, output_dir,
        f'{flags.env_id}_long_term', n_dims_to_show=5
    )

if __name__ == '__main__':
    main()
