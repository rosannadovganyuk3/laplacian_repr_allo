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
    """Adjacency over the true (fully observable) states, the random-policy
    transition matrix P (4 moves + stay, blocked moves become self-loops), and
    the policy Laplacian L = I - P -- the object whose eigenvectors GDO
    actually recovers (a resolvent of P), not just a combinatorial D - A."""
    pos = env.task.maze.all_empty_grids()
    n = len(pos)
    index = {tuple(p): i for i, p in enumerate(pos)}
    A = np.zeros((n, n))
    for i, p in enumerate(pos):
        for mv in MOVES:
            q = tuple(p + np.array(mv))
            if q in index:
                A[i, index[q]] = 1.0

    amap = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
    P = np.zeros((n, n))
    for i, p in enumerate(pos):
        for a in amap:
            q = p + a
            j = index[tuple(q)] if env.task.maze.is_empty(q) else i
            P[i, j] += 1.0 / 5.0

    L = np.eye(n) - P
    return pos, index, A, P, L


def verify_discount_invariance(P):
    """P must be symmetric for every gamma-discounted Laplacian to share
    eigenvectors with L = I - P (a resolvent-in-P argument) -- this is what
    makes the short/long-discount ground truth the same eigenspace regardless
    of gamma. Returns the max asymmetry, which should be ~0."""
    return np.abs(P - P.T).max(), np.allclose(P, P.T)


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


def aliased_transition_eigvecs(P, B, d):
    """Eigenvectors of the transition matrix under aliasing: instead of
    projecting the TRUE eigenvectors onto span(B) (memoryless_ceiling),
    aggregate the dynamics themselves by the aliasing partition and
    diagonalize that.

    P_alias = B^T P B is the (n_groups x n_groups) matrix of transition mass
    between aliasing classes; exact (not an approximation) because P is
    symmetric, hence has uniform stationary distribution, so plain
    block-averaging is also the stationary-weighted aggregation.
    L_alias = I - P_alias mirrors L = I - P from the ground-truth section,
    so the same eigh convention applies: ascending eigenvalues of L_alias,
    index 0 is the trivial/constant eigenvector.

    The lifted eigenvectors (B @ eigvecs_alias) are a specific candidate
    memoryless representation -- but they still live in span(B) by
    construction, so by the same Ky Fan argument as the ceiling above, their
    overlap with V is bounded above by it, and only matches it if the
    aliasing partition happens to be an exact (Kemeny-Snell) lumping.
    """
    n_groups = B.shape[1]
    P_alias = B.T @ P @ B
    L_alias = np.eye(n_groups) - P_alias
    n_take = min(d, n_groups - 1)
    _, eigvecs_alias = eigh(L_alias, subset_by_index=[0, n_take])
    return B @ eigvecs_alias[:, 1:]          # drop trivial eigvec, lift to state space


# ----------------------------------------------------------------------------------
# Learned representation
# ----------------------------------------------------------------------------------

def embed(model, states, device, batch=256, window_len=1):
    """window_len=1 embeds each state as a single frame (the memoryless case).
    window_len>1 tiles each state's frame that many times along a new time
    axis before embedding -- there's no real trajectory behind a synthetic
    grid-sweep position, so this approximates "the agent has been stationary
    here" rather than real history. It's the best available proxy for
    comparing a windowed encoder against the memoryless ceiling without
    sweeping real recorded episodes instead of synthetic positions."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(states), batch):
            chunk = states[i:i + batch]
            if window_len > 1:
                chunk = np.repeat(chunk[:, None, ...], window_len, axis=1)
            x = torch_tools.to_tensor(chunk, device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out, 0)


def collect_raw_rollout(env, obs_prepro, index, n_steps=20000):
    """Random-policy rollout, split into episodes, for building REAL (not
    synthetic-repeated) windowed observations.

    This depends only on the environment and its observation function --
    not on window_len, not on any specific run's hyperparameters or
    checkpoint -- so it should be collected once per environment and reused
    for every run and every window_len via windows_from_rollout(), rather
    than re-collected per run (768 rollouts of this size would be far too
    slow)."""
    episodes = []
    total = 0
    action_spec = env.task.action_spec
    while total < n_steps:
        ts = env.reset()
        frames, pos_idxs = [], []
        while True:
            frames.append(obs_prepro(ts.observation))
            pos_idxs.append(index.get(tuple(env.task._agent_pos)))
            total += 1
            if ts.is_last or total >= n_steps:
                break
            ts = env.step(action_spec.sample())
        episodes.append((frames, pos_idxs))
    return episodes


def windows_from_rollout(episodes, window_len):
    """One real trailing window_len-frame history per visited position (the
    first occurrence in the rollout), left-padded by repeating that episode's
    own first frame -- the same indexing convention as
    episodic_replay_buffer.py's _get_window used during training, so this
    matches what the model actually saw.

    Only the first visit per position is kept (not every occurrence): a
    single real window already fixes the "repeated fake frame" problem, and
    keeping every visit would mean embedding thousands of sequences per run
    (once per rollout step) instead of one per position -- correct either
    way, but the single-occurrence version is 5-30x cheaper to embed and the
    extra averaging wasn't buying much."""
    seqs, pos_idxs = [], []
    seen = set()
    for frames, pidxs in episodes:
        for t, pidx in enumerate(pidxs):
            if pidx is None or pidx in seen:
                continue
            seen.add(pidx)
            window = [frames[max(t - window_len + 1 + i, 0)]
                      for i in range(window_len)]
            seqs.append(np.stack(window, axis=0))
            pos_idxs.append(pidx)
    return np.array(seqs), np.array(pos_idxs)


def embed_averaged(model, seqs, pos_idxs, n, d, device, batch=256):
    """Embed real windowed sequences and average per true position (a state
    visited multiple times by the rollout gets one embedding per visit,
    averaged). Returns a coverage mask -- positions the rollout never
    reached need the synthetic-repeat fallback in embed())."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            x = torch_tools.to_tensor(seqs[i:i + batch], device)
            out.append(model(x).cpu().numpy())
    Phi_all = np.concatenate(out, 0) if out else np.zeros((0, d))
    sums = np.zeros((n, d))
    counts = np.zeros(n)
    if len(pos_idxs) > 0:
        np.add.at(sums, pos_idxs, Phi_all)
        np.add.at(counts, pos_idxs, 1)
    covered = counts > 0
    Phi = np.zeros((n, d))
    Phi[covered] = sums[covered] / counts[covered, None]
    return Phi, covered


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

def plot_env(env_id, out_dir, res, pos, layout_shape, V, U_s, U_l, run_tag=None):
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
    suffix = f'_{run_tag}' if run_tag else ''
    path = os.path.join(out_dir, f'{env_id}{suffix}_ground_truth_eval.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_eigenmaps(env_id, out_dir, pos, layout_shape, V, U_s, U_l,
                    V0=None, s0=None, l0=None, n_dims=5, run_tag=None):
    """True vs. learned (Procrustes-aligned) eigenmaps, side by side, for the
    first n_dims dimensions -- same clean heatmap style as plot_env's bottom
    row (NaN-masked walls, no contour outlines), just extended past dim 1.
    Each real panel auto-scales to its own data for maximum contrast.

    V0, if given, is the trivial/constant eigenvector (excluded from V and
    every other comparison in this file); it's pinned to a fixed scale
    instead of auto-scaling, since its own data only varies at the ~1e-15
    floating-point level -- auto-scaling would stretch that noise across the
    full colorbar and make it look like a dramatic pattern instead of the
    flat field it actually is. s0/l0, if given, are the leading singular
    vectors of the RAW (uncentered) short/long embeddings -- the closest
    principled analog to "dim 0" on the learned side, since it's exactly
    what orthonormal_basis()'s centering step removes from U_s/U_l."""
    n_dims = min(n_dims, V.shape[1])
    aligned_s = U_s @ procrustes_rotation(U_s, V)
    aligned_l = U_l @ procrustes_rotation(U_l, V)

    has_trivial_row = V0 is not None
    n_rows = n_dims + (1 if has_trivial_row else 0)
    fig, ax = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        ax = ax[None, :]

    def draw(a, vals, title, vmax=None, colorbar=False):
        m = np.full(layout_shape, np.nan)
        m[pos[:, 0], pos[:, 1]] = vals
        if vmax is not None:
            im = a.imshow(m, cmap='RdBu', interpolation='none', vmin=-vmax, vmax=vmax)
        else:
            im = a.imshow(m, cmap='RdBu', interpolation='none')
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
        if colorbar:
            plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    row = 0
    if has_trivial_row:
        vmax0 = max(np.abs(V0).max(), 1e-3)  # wide enough to swallow its own fp noise
        draw(ax[row, 0], V0[:, 0], 'trivial eigenvector (excluded)', vmax=vmax0, colorbar=True)
        if s0 is not None:
            draw(ax[row, 1], s0[:, 0], 'short, leading raw component', colorbar=True)
        else:
            ax[row, 1].axis('off')
        if l0 is not None:
            draw(ax[row, 2], l0[:, 0], 'long, leading raw component', colorbar=True)
        else:
            ax[row, 2].axis('off')
        row += 1

    for i in range(n_dims):
        draw(ax[row, 0], V[:, i], f'true eigenvector {i+1}', colorbar=True)
        draw(ax[row, 1], aligned_s[:, i], f'short, aligned (dim {i+1})', colorbar=True)
        draw(ax[row, 2], aligned_l[:, i], f'long, aligned (dim {i+1})', colorbar=True)
        row += 1

    plt.tight_layout()
    suffix = f'_{run_tag}' if run_tag else ''
    path = os.path.join(out_dir, f'{env_id}{suffix}_eigenmaps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ----------------------------------------------------------------------------------

def build_eval_context(env_id, args, rollout_steps=20000):
    """Everything about an environment's ground truth and real-trajectory
    rollout that's identical across every run/hyperparameter combo for that
    environment. Build once per env_id and pass to evaluate() for every run,
    rather than recomputing per run -- both wasteful (the graph/eigh work is
    pure overhead 256x over) and, for the rollout specifically, far too slow
    to redo per run."""
    init_kwargs = dict(env_id=env_id)
    if args.device:
        init_kwargs['device'] = args.device
    cfg = importlib.import_module(args.config_dir + '.' + args.config_file) \
        .Config(flag_tools.Flags(**init_kwargs))
    la = cfg.args_as_flags
    d = cfg.flags.d

    env = la.env_factory()
    pos, index, A, P, L = build_state_graph(env)
    n = len(pos)

    dev, sym = verify_discount_invariance(P)
    evals_full, V_full = eigh(L, subset_by_index=[0, d])
    V0 = V_full[:, [0]]
    V = V_full[:, 1:]

    states = np.array([la.obs_prepro(env.task.pos_to_obs(p)) for p in pos])
    groups = aliasing_classes(states)
    B = block_constant_basis(groups, n)
    ceiling, per_vec_ceiling = memoryless_ceiling(V, B)
    W_alias = aliased_transition_eigvecs(P, B, d)
    ov_alias, _ = subspace_overlap(W_alias, V)

    goal_idx = index[tuple(env.task.goal_pos)]
    true_dist = shortest_path_from(A, goal_idx)
    rho_gt, _ = distance_quality(V, goal_idx, true_dist)
    rho_px, _ = distance_quality(states.reshape(n, -1), goal_idx, true_dist)

    sizes = np.array([len(g) for g in groups])
    rollout = collect_raw_rollout(env, la.obs_prepro, index, n_steps=rollout_steps)

    return flag_tools.Flags(
        env=env, pos=pos, index=index, A=A, P=P, L=L, n=n, d=d,
        P_deviation=dev, P_symmetric=sym,
        V0=V0, V=V, states=states, groups=groups, sizes=sizes,
        ceiling=ceiling, per_vec_ceiling=per_vec_ceiling, ov_alias=ov_alias,
        goal_idx=goal_idx, true_dist=true_dist, rho_gt=rho_gt, rho_px=rho_px,
        rollout=rollout, window_cache={},
    )


def evaluate(env_id, args, ctx):
    exp_name = getattr(args, 'exp_name', 'laprepr')
    log_dir = os.path.join(args.log_base_dir, exp_name, env_id, args.log_sub_dir)
    flags = flag_tools.load_flags(log_dir)
    if args.device:
        flags.device = args.device
    cfg = importlib.import_module(args.config_dir + '.' + args.config_file).Config(flags)
    la = cfg.args_as_flags
    device = la.device
    d = flags.d
    assert d == ctx.d, f"ctx was built for d={ctx.d}, this run uses d={d}"

    models = {}
    for tag in ['short', 'long']:
        m = la.model_cfg.model_factory().to(device)
        m.load_state_dict(torch.load(os.path.join(log_dir, f'model_{tag}.ckpt'),
                                     map_location=device))
        models[tag] = m

    Phi_s = embed(models['short'], ctx.states, device)
    Phi_l = embed(models['long'], ctx.states, device)
    U_s, sv_s, rank_s = orthonormal_basis(Phi_s)
    U_l, sv_l, rank_l = orthonormal_basis(Phi_l)

    # "dim 0" analog for the learned side: orthonormal_basis centers Phi before
    # its SVD, discarding any consistent per-state offset the same way V0 is
    # excluded from V. Running SVD on the RAW (uncentered) embedding instead,
    # its leading component is the closest principled stand-in for what
    # centering throws away -- sign-aligned against V0 just for a consistent
    # red/blue orientation in the plot (SVD signs are otherwise arbitrary).
    s0_raw, _, _ = svd(Phi_s, full_matrices=False)
    l0_raw, _, _ = svd(Phi_l, full_matrices=False)
    s0 = s0_raw[:, [0]] * np.sign(s0_raw[:, 0] @ ctx.V0[:, 0])
    l0 = l0_raw[:, [0]] * np.sign(l0_raw[:, 0] @ ctx.V0[:, 0])

    ov_s, cos_s = subspace_overlap(U_s, ctx.V)
    ov_l, cos_l = subspace_overlap(U_l, ctx.V)
    ov_cross, _ = subspace_overlap(U_s, U_l)

    # Windowed comparison: does giving the encoder REAL history (as it was
    # trained with, sliced from ctx.rollout -- not a repeated static frame)
    # recover more of the true eigenspace than the memoryless case above?
    # getattr default handles checkpoints from before window_len existed.
    window_len = getattr(flags, 'window_len', 1)
    if window_len > 1:
        if window_len not in ctx.window_cache:
            ctx.window_cache[window_len] = windows_from_rollout(ctx.rollout, window_len)
        seqs, pos_idxs = ctx.window_cache[window_len]

        Phi_s_win, cov_s = embed_averaged(models['short'], seqs, pos_idxs, ctx.n, d, device)
        Phi_l_win, cov_l = embed_averaged(models['long'], seqs, pos_idxs, ctx.n, d, device)
        if not cov_s.all():
            Phi_s_win[~cov_s] = embed(models['short'], ctx.states[~cov_s], device,
                                       window_len=window_len)
        if not cov_l.all():
            Phi_l_win[~cov_l] = embed(models['long'], ctx.states[~cov_l], device,
                                       window_len=window_len)

        U_s_win, _, _ = orthonormal_basis(Phi_s_win)
        U_l_win, _, _ = orthonormal_basis(Phi_l_win)
        ov_s_win, _ = subspace_overlap(U_s_win, ctx.V)
        ov_l_win, _ = subspace_overlap(U_l_win, ctx.V)
    else:
        ov_s_win, ov_l_win = ov_s, ov_l

    P_s = U_s @ U_s.T
    P_l = U_l @ U_l.T
    recov_s = np.array([ctx.V[:, i] @ P_s @ ctx.V[:, i] for i in range(d)])
    recov_l = np.array([ctx.V[:, i] @ P_l @ ctx.V[:, i] for i in range(d)])

    rho_s, deg_s = distance_quality(Phi_s, ctx.goal_idx, ctx.true_dist)
    rho_l, deg_l = distance_quality(Phi_l, ctx.goal_idx, ctx.true_dist)
    rho_c, deg_c = distance_quality(np.concatenate([Phi_s, Phi_l], 1),
                                     ctx.goal_idx, ctx.true_dist)

    res = dict(
        env=env_id, d=d, n_states=ctx.n, n_obs=len(ctx.groups),
        max_alias=int(ctx.sizes.max()),
        goal_twins=int(ctx.sizes[[ctx.goal_idx in g for g in ctx.groups]][0]),
        P_deviation=ctx.P_deviation, P_symmetric=ctx.P_symmetric,
        ceiling=ctx.ceiling, per_vec_ceiling=ctx.per_vec_ceiling,
        overlap_alias_eig=ctx.ov_alias,
        overlap_short=ov_s, overlap_long=ov_l, overlap_cross=ov_cross,
        window_len=window_len,
        overlap_short_windowed=ov_s_win, overlap_long_windowed=ov_l_win,
        cos_short=cos_s, cos_long=cos_l,
        recov_short=recov_s, recov_long=recov_l,
        sv_short=sv_s, sv_long=sv_l,
        erank_short=effective_rank(sv_s), erank_long=effective_rank(sv_l),
        rank_short=rank_s, rank_long=rank_l,
        raw_cos=raw_cosine(Phi_s, Phi_l), aligned_cos=aligned_cosine(Phi_s, Phi_l),
        rho_short=rho_s, rho_long=rho_l, rho_concat=rho_c,
        rho_gt=ctx.rho_gt, rho_pixels=ctx.rho_px,
        degenerate_short=deg_s, degenerate_concat=deg_c,
    )

    fig_path = None
    eigenmaps_path = None
    if not args.no_figures:
        out_dir = os.path.join(args.log_base_dir, args.output_sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        fig_path = plot_env(env_id, out_dir, res, ctx.pos,
                            ctx.env.task.maze.maze_array.shape, ctx.V, U_s, U_l,
                            run_tag=args.log_sub_dir)
        eigenmaps_path = plot_eigenmaps(env_id, out_dir, ctx.pos,
                            ctx.env.task.maze.maze_array.shape, ctx.V, U_s, U_l,
                            V0=ctx.V0, s0=s0, l0=l0, run_tag=args.log_sub_dir)
    return res, fig_path, eigenmaps_path


def report(r):
    print(f"\n{'='*74}\n{r['env']}   (d={r['d']}, {r['n_states']} states)\n{'='*74}")
    print(f"  ground truth L = I - P is discount-invariant: P symmetric to "
          f"{r['P_deviation']:.1e} (max asymmetry), symmetric={r['P_symmetric']}")
    print(f"  observations: {r['n_obs']} distinct for {r['n_states']} states "
          f"({100*(1-r['n_obs']/r['n_states']):.0f}% collapsed); "
          f"largest aliasing class {r['max_alias']}")
    print(f"\n  SUBSPACE OVERLAP WITH TRUE TOP-{r['d']} EIGENSPACE   (1.0 = perfect)")
    print(f"    memoryless ceiling      {r['ceiling']:.3f}")
    print(f"    eigvecs of transition matrix under aliasing   {r['overlap_alias_eig']:.3f}"
          f"   ({100*r['overlap_alias_eig']/r['ceiling']:.0f}% of ceiling)")
    print(f"    short  (gamma=0.1)      {r['overlap_short']:.3f}"
          f"   ({100*r['overlap_short']/r['ceiling']:.0f}% of ceiling)")
    print(f"    long   (gamma=0.9)      {r['overlap_long']:.3f}"
          f"   ({100*r['overlap_long']/r['ceiling']:.0f}% of ceiling)")
    if r['window_len'] > 1:
        print(f"\n  SAME, BUT WITH {r['window_len']}-FRAME HISTORY (does memory beat the ceiling?)")
        print(f"    short, windowed         {r['overlap_short_windowed']:.3f}"
              f"   ({100*r['overlap_short_windowed']/r['ceiling']:.0f}% of ceiling)")
        print(f"    long,  windowed         {r['overlap_long_windowed']:.3f}"
              f"   ({100*r['overlap_long_windowed']/r['ceiling']:.0f}% of ceiling)")
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
    p.add_argument('--exp_name', type=str, default='laprepr')
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
            ctx = build_eval_context(env_id, args)
            r, fig, eigenmaps_fig = evaluate(env_id, args, ctx)
        except FileNotFoundError as e:
            print(f"\n[skip] {env_id}: {e}")
            continue
        report(r)
        if fig:
            print(f"\n  figure: {fig}")
        if eigenmaps_fig:
            print(f"  eigenmaps figure: {eigenmaps_fig}")
        rows.append(r)

    if rows:
        out_dir = os.path.join(args.log_base_dir, args.output_sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        cols = ['env', 'n_states', 'n_obs', 'max_alias', 'ceiling', 'overlap_alias_eig',
                'overlap_short', 'overlap_long', 'overlap_cross',
                'window_len', 'overlap_short_windowed', 'overlap_long_windowed',
                'raw_cos', 'aligned_cos', 'erank_short', 'erank_long',
                'rho_gt', 'rho_short', 'rho_long', 'rho_concat', 'rho_pixels']
        csv_path = os.path.join(out_dir, f'ground_truth_metrics_{args.log_sub_dir}.csv')
        with open(csv_path, 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in rows:
                f.write(','.join(
                    f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                    for c in cols) + '\n')
        print(f"\nMetrics written to {csv_path}")


if __name__ == '__main__':
    main()
