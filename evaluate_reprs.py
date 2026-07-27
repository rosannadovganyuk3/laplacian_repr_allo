"""Ground-truth evaluation of learned Laplacian representations.

The ground truth is the graph Laplacian of the underlying MDP:  L = D - A over the
true state graph.  There is exactly one ground truth.  Partial observability does not
introduce a second one -- it only limits what a function of observations can express,
which is reported here as a *ceiling* (a reference line on the score), not as a target.

Metrics are rotation-invariant, because the graph-drawing objective is invariant under
phi -> R phi for orthogonal R.  Raw cosine similarity between two encoders is therefore
meaningless as a measure of agreement; subspace overlap is the correct analogue.

Usage:
    python -u -B evaluate_reprs.py                       # all three envs
    python -u -B evaluate_reprs.py --envs HardMaze
    python -u -B evaluate_reprs.py --device cpu --no_figures
"""
import os
import argparse
import importlib
import collections

import numpy as np
import torch
from scipy.linalg import eigh, svd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rl_lap.tools import flag_tools, torch_tools

MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]


# ----------------------------------------------------------------------------------
# Ground truth: the MDP graph Laplacian
# ----------------------------------------------------------------------------------

def build_state_graph(env):
    """Adjacency + combinatorial Laplacian over the true (fully observable) states."""
    pos = env.task.maze.all_empty_grids()
    n = len(pos)
    index = {tuple(p): i for i, p in enumerate(pos)}
    A = np.zeros((n, n))
    for i, p in enumerate(pos):
        for mv in MOVES:
            q = tuple(p + np.array(mv))
            if q in index:
                A[i, index[q]] = 1.0
    L = np.diag(A.sum(1)) - A
    return pos, index, A, L


def verify_discount_invariance(env, pos, index, L):
    """The random-policy transition matrix is exactly P = I - L/5 for this action set
    (4 moves + stay, blocked moves become stay).  P is symmetric, so the chain is
    reversible with a uniform stationary distribution, and every gamma-discounted
    Laplacian  I - (1-g)(I-gP)^-1  is a resolvent in P: same eigenvectors, same order.
    Returns the max deviation, which should be ~1e-16.
    """
    amap = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
    n = len(pos)
    P = np.zeros((n, n))
    for i, p in enumerate(pos):
        for a in amap:
            q = p + a
            j = index[tuple(q)] if env.task.maze.is_empty(q) else i
            P[i, j] += 1.0 / 5.0
    return np.abs(P - (np.eye(n) - L / 5.0)).max(), np.allclose(P, P.T)


def shortest_path_from(A, source):
    """BFS hop distance from one node to all others."""
    n = A.shape[0]
    dist = np.full(n, np.inf)
    dist[source] = 0
    frontier = collections.deque([source])
    while frontier:
        u = frontier.popleft()
        for v in np.flatnonzero(A[u]):
            if np.isinf(dist[v]):
                dist[v] = dist[u] + 1
                frontier.append(v)
    return dist


# ----------------------------------------------------------------------------------
# Observation aliasing and the memoryless ceiling
# ----------------------------------------------------------------------------------

def aliasing_classes(states):
    """Group state indices by identical observation.  A memoryless encoder phi = f(o)
    is necessarily constant within each group."""
    buckets = collections.OrderedDict()
    for i, s in enumerate(states):
        buckets.setdefault(np.ascontiguousarray(s).tobytes(), []).append(i)
    return list(buckets.values())


def block_constant_basis(groups, n):
    """Orthonormal basis for functions that are constant on each aliasing class."""
    B = np.zeros((n, len(groups)))
    for g, members in enumerate(groups):
        B[members, g] = 1.0 / np.sqrt(len(members))
    return B


def memoryless_ceiling(V, B):
    """Upper bound on subspace overlap achievable by any function of the observation.

    For orthonormal U with d columns constrained to span(B), max (1/d)||U^T V||_F^2
    equals the sum of the d largest eigenvalues of (B^T V)(B^T V)^T.  That matrix has
    rank <= d, so when #classes >= d the maximum is simply ||B^T V||_F^2 / d.
    """
    BtV = B.T @ V
    per_vec = (BtV ** 2).sum(0)              # ||P_B v_i||^2 for unit v_i
    d = V.shape[1]
    if B.shape[1] >= d:
        ceiling = per_vec.sum() / d
    else:
        s = svd(BtV, compute_uv=False)
        ceiling = (s ** 2).sum() / d
    return ceiling, per_vec


# ----------------------------------------------------------------------------------
# Learned representation
# ----------------------------------------------------------------------------------

def embed(model, states, device, batch=256):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(states), batch):
            x = torch_tools.to_tensor(states[i:i + batch], device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out, 0)


def orthonormal_basis(Phi, tol=1e-3):
    """Centred column space of the embedding.  Centring removes the constant direction,
    matching the exclusion of the trivial eigenvector v_0 from the ground truth."""
    Phi_c = Phi - Phi.mean(0, keepdims=True)
    U, s, _ = svd(Phi_c, full_matrices=False)
    rank = int((s > tol * s[0]).sum()) if s[0] > 0 else 0
    return U, s, rank


def subspace_overlap(U, V):
    """(1/d) * sum of cos^2 of principal angles between span(U) and span(V), in [0, 1]."""
    M = U.T @ V
    cos2 = svd(M, compute_uv=False) ** 2
    return cos2.sum() / V.shape[1], np.sqrt(np.clip(cos2, 0, 1))


def procrustes_rotation(X, Y):
    """Orthogonal R minimising ||X R - Y||_F."""
    W, _, Zt = svd(X.T @ Y, full_matrices=False)
    return W @ Zt


def aligned_cosine(Phi_a, Phi_b):
    """Row-wise cosine AFTER removing the basis ambiguity the objective leaves free.
    This is the meaningful version of the raw cos_sim logged during training."""
    A = Phi_a - Phi_a.mean(0, keepdims=True)
    Bm = Phi_b - Phi_b.mean(0, keepdims=True)
    A = A / np.linalg.norm(A)
    Bm = Bm / np.linalg.norm(Bm)
    Ar = A @ procrustes_rotation(A, Bm)
    num = (Ar * Bm).sum(1)
    den = np.linalg.norm(Ar, axis=1) * np.linalg.norm(Bm, axis=1) + 1e-12
    return float(np.mean(num / den))


def raw_cosine(Phi_a, Phi_b):
    """The metric currently logged to repr_convergence.csv, for comparison."""
    num = (Phi_a * Phi_b).sum(1)
    den = np.linalg.norm(Phi_a, axis=1) * np.linalg.norm(Phi_b, axis=1) + 1e-12
    return float(np.mean(num / den))


def effective_rank(s):
    """Participation ratio of the singular value spectrum: collapse diagnostic."""
    p = s ** 2
    if p.sum() <= 0:
        return 0.0
    return float((p.sum() ** 2) / (p ** 2).sum())


# ----------------------------------------------------------------------------------
# Reward-shaping diagnostic
# ----------------------------------------------------------------------------------

def distance_quality(Phi, goal_idx, true_dist):
    """How well does ||phi(s) - phi(g)|| track true shortest-path distance?
    This is exactly the quantity the 'mix' shaped reward depends on."""
    emb_dist = np.linalg.norm(Phi - Phi[goal_idx], axis=1)
    finite = np.isfinite(true_dist)
    rho = spearmanr(emb_dist[finite], true_dist[finite])[0]
    # fraction of states whose shaped reward is indistinguishable from another state's
    _, counts = np.unique(np.round(emb_dist, 6), return_counts=True)
    degenerate = (counts[counts > 1].sum()) / len(emb_dist)
    return float(rho), float(degenerate)


# ----------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------

def plot_env(env_id, out_dir, res, pos, layout_shape, V, U_s, U_l):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    # principal angles
    ax[0, 0].plot(res['cos_short'], 'o-', label='short (g=0.1)')
    ax[0, 0].plot(res['cos_long'], 's-', label='long (g=0.9)')
    ax[0, 0].axhline(np.sqrt(res['ceiling']), color='k', ls='--',
                     label=f"memoryless ceiling (rms {np.sqrt(res['ceiling']):.2f})")
    ax[0, 0].set(xlabel='principal angle index', ylabel='cos(theta)',
                 title=f'{env_id}: alignment with true eigenspace', ylim=(0, 1.05))
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)

    # per-eigenvector recovery vs ceiling
    x = np.arange(len(res['per_vec_ceiling']))
    ax[0, 1].bar(x - .2, res['recov_short'], .4, label='short', alpha=.8)
    ax[0, 1].bar(x + .2, res['recov_long'], .4, label='long', alpha=.8)
    ax[0, 1].plot(x, res['per_vec_ceiling'], 'k^--', ms=5, label='ceiling')
    ax[0, 1].set(xlabel='true eigenvector index', ylabel='captured fraction',
                 title='per-eigenvector recovery', ylim=(0, 1.05))
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)

    # singular spectrum (collapse)
    ax[0, 2].semilogy(res['sv_short'] / res['sv_short'][0], 'o-', label='short')
    ax[0, 2].semilogy(res['sv_long'] / res['sv_long'][0], 's-', label='long')
    ax[0, 2].set(xlabel='component', ylabel='normalised singular value',
                 title=f"collapse check (eff. rank {res['erank_short']:.1f} / "
                       f"{res['erank_long']:.1f} of {res['d']})")
    ax[0, 2].legend(fontsize=8); ax[0, 2].grid(alpha=.3)

    # eigenmaps: true vs learned (rotated onto the GT basis for comparability)
    def draw(a, vals, title):
        m = np.full(layout_shape, np.nan)
        m[pos[:, 0], pos[:, 1]] = vals
        a.imshow(m, cmap='RdBu', interpolation='none')
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])

    R_s = procrustes_rotation(U_s, V)
    aligned = U_s @ R_s
    draw(ax[1, 0], V[:, 0], 'true eigenvector 1')
    draw(ax[1, 1], aligned[:, 0], 'learned (short), aligned')
    draw(ax[1, 2], V[:, 1], 'true eigenvector 2')

    plt.tight_layout()
    path = os.path.join(out_dir, f'{env_id}_ground_truth_eval.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ----------------------------------------------------------------------------------

def evaluate(env_id, args):
    log_dir = os.path.join(args.log_base_dir, 'laprepr', env_id, args.log_sub_dir)
    flags = flag_tools.load_flags(log_dir)
    if args.device:
        flags.device = args.device
    cfg = importlib.import_module(args.config_dir + '.' + args.config_file).Config(flags)
    la = cfg.args_as_flags
    device = la.device
    d = flags.d

    env = la.env_factory()
    pos, index, A, L = build_state_graph(env)
    n = len(pos)

    dev, sym = verify_discount_invariance(env, pos, index, L)
    evals, V = eigh(L, subset_by_index=[1, d])          # ground truth, n x d

    states = np.array([la.obs_prepro(env.task.pos_to_obs(p)) for p in pos])

    groups = aliasing_classes(states)
    B = block_constant_basis(groups, n)
    ceiling, per_vec_ceiling = memoryless_ceiling(V, B)

    models = {}
    for tag in ['short', 'long']:
        m = la.model_cfg.model_factory().to(device)
        m.load_state_dict(torch.load(os.path.join(log_dir, f'model_{tag}.ckpt'),
                                     map_location=device))
        models[tag] = m

    Phi_s = embed(models['short'], states, device)
    Phi_l = embed(models['long'], states, device)
    U_s, sv_s, rank_s = orthonormal_basis(Phi_s)
    U_l, sv_l, rank_l = orthonormal_basis(Phi_l)

    ov_s, cos_s = subspace_overlap(U_s, V)
    ov_l, cos_l = subspace_overlap(U_l, V)
    ov_cross, _ = subspace_overlap(U_s, U_l)

    P_s = U_s @ U_s.T
    P_l = U_l @ U_l.T
    recov_s = np.array([V[:, i] @ P_s @ V[:, i] for i in range(d)])
    recov_l = np.array([V[:, i] @ P_l @ V[:, i] for i in range(d)])

    goal_idx = index[tuple(env.task.goal_pos)]
    true_dist = shortest_path_from(A, goal_idx)
    rho_s, deg_s = distance_quality(Phi_s, goal_idx, true_dist)
    rho_l, deg_l = distance_quality(Phi_l, goal_idx, true_dist)
    rho_c, deg_c = distance_quality(np.concatenate([Phi_s, Phi_l], 1), goal_idx, true_dist)
    rho_gt, _ = distance_quality(V, goal_idx, true_dist)
    rho_px, _ = distance_quality(states.reshape(n, -1), goal_idx, true_dist)

    sizes = np.array([len(g) for g in groups])
    res = dict(
        env=env_id, d=d, n_states=n, n_obs=len(groups),
        max_alias=int(sizes.max()), goal_twins=int(sizes[[goal_idx in g for g in groups]][0]),
        P_deviation=dev, P_symmetric=sym,
        ceiling=ceiling, per_vec_ceiling=per_vec_ceiling,
        overlap_short=ov_s, overlap_long=ov_l, overlap_cross=ov_cross,
        cos_short=cos_s, cos_long=cos_l,
        recov_short=recov_s, recov_long=recov_l,
        sv_short=sv_s, sv_long=sv_l,
        erank_short=effective_rank(sv_s), erank_long=effective_rank(sv_l),
        rank_short=rank_s, rank_long=rank_l,
        raw_cos=raw_cosine(Phi_s, Phi_l), aligned_cos=aligned_cosine(Phi_s, Phi_l),
        rho_short=rho_s, rho_long=rho_l, rho_concat=rho_c,
        rho_gt=rho_gt, rho_pixels=rho_px,
        degenerate_short=deg_s, degenerate_concat=deg_c,
    )

    fig_path = None
    if not args.no_figures:
        out_dir = os.path.join(args.log_base_dir, args.output_sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        fig_path = plot_env(env_id, out_dir, res, pos,
                            env.task.maze.maze_array.shape, V, U_s, U_l)
    return res, fig_path


def report(r):
    print(f"\n{'='*74}\n{r['env']}   (d={r['d']}, {r['n_states']} states)\n{'='*74}")
    print(f"  ground truth is discount-invariant: P == I - L/5 to {r['P_deviation']:.1e}, "
          f"symmetric={r['P_symmetric']}")
    print(f"  observations: {r['n_obs']} distinct for {r['n_states']} states "
          f"({100*(1-r['n_obs']/r['n_states']):.0f}% collapsed); "
          f"largest aliasing class {r['max_alias']}")
    print(f"\n  SUBSPACE OVERLAP WITH TRUE TOP-{r['d']} EIGENSPACE   (1.0 = perfect)")
    print(f"    memoryless ceiling      {r['ceiling']:.3f}")
    print(f"    short  (gamma=0.1)      {r['overlap_short']:.3f}"
          f"   ({100*r['overlap_short']/r['ceiling']:.0f}% of ceiling)")
    print(f"    long   (gamma=0.9)      {r['overlap_long']:.3f}"
          f"   ({100*r['overlap_long']/r['ceiling']:.0f}% of ceiling)")
    print(f"\n  DO THE TWO DISCOUNTS AGREE?")
    print(f"    subspace overlap short vs long   {r['overlap_cross']:.3f}   <- the valid test")
    print(f"    raw cosine (as logged in CSV)    {r['raw_cos']:.3f}   <- basis-dependent, ignore")
    print(f"    Procrustes-aligned cosine        {r['aligned_cos']:.3f}")
    print(f"\n  COLLAPSE CHECK")
    print(f"    effective rank   short {r['erank_short']:.2f} / {r['d']}"
          f"    long {r['erank_long']:.2f} / {r['d']}")
    print(f"\n  SHAPED-REWARD QUALITY  (Spearman of ||phi(s)-phi(goal)|| vs true hop distance)")
    print(f"    true eigenvectors (ceiling) {r['rho_gt']:+.3f}")
    print(f"    short {r['rho_short']:+.3f}   long {r['rho_long']:+.3f}   "
          f"concat ('mix') {r['rho_concat']:+.3f}   raw pixels ('l2') {r['rho_pixels']:+.3f}")
    print(f"    states with a duplicated shaped reward: {100*r['degenerate_concat']:.0f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_base_dir', type=str, default=os.path.join(os.getcwd(), 'log'))
    p.add_argument('--log_sub_dir', type=str, default='test')
    p.add_argument('--output_sub_dir', type=str, default='gt_eval')
    p.add_argument('--config_dir', type=str, default='rl_lap.configs')
    p.add_argument('--config_file', type=str, default='laprepr_config_gridworld')
    p.add_argument('--envs', type=str, nargs='+',
                   default=['OneRoom', 'TwoRoom', 'HardMaze'])
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--no_figures', action='store_true')
    args = p.parse_args()

    rows = []
    for env_id in args.envs:
        try:
            r, fig = evaluate(env_id, args)
        except FileNotFoundError as e:
            print(f"\n[skip] {env_id}: {e}")
            continue
        report(r)
        if fig:
            print(f"\n  figure: {fig}")
        rows.append(r)

    if rows:
        out_dir = os.path.join(args.log_base_dir, args.output_sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        cols = ['env', 'n_states', 'n_obs', 'max_alias', 'ceiling',
                'overlap_short', 'overlap_long', 'overlap_cross',
                'raw_cos', 'aligned_cos', 'erank_short', 'erank_long',
                'rho_gt', 'rho_short', 'rho_long', 'rho_concat', 'rho_pixels']
        csv_path = os.path.join(out_dir, 'ground_truth_metrics.csv')
        with open(csv_path, 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in rows:
                f.write(','.join(
                    f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                    for c in cols) + '\n')
        print(f"\nMetrics written to {csv_path}")


if __name__ == '__main__':
    main()
