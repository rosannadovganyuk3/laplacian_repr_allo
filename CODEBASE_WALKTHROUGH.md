# Codebase walkthrough: `laplacian_repr_allo`

Origin: a re-implementation of Wu, Tucker & Nachum (2019), *The Laplacian in RL: Learning
Representations with Efficient Approximations*. You have modified it to (a) make the gridworld
partially observable and (b) train **two** representation networks with **two different discount
factors**, in order to test whether the singular Laplacian representation is still recoverable
under partial observability.

---

## 1. The 30-second map

```
train_laprepr.py          ← MAIN ENTRY POINT for your experiment
  └─ rl_lap/configs/laprepr_config_gridworld.py   (hyperparameters, obs preprocessing)
       └─ rl_lap/agent/laprepr.py                 (the two-model training loop + losses)
            ├─ rl_lap/agent/episodic_replay_buffer.py  (dual-discount pair sampling)
            ├─ rl_lap/configs/networks.py              (the LSTM encoder)
            └─ rl_lap/envs/gridworld/*                 (the partially-observable maze)

visualize_reprs.py        ← post-hoc analysis vs. ground-truth Laplacian eigenvectors
plot_*.py / check_repr.py ← plotting the cosine-similarity CSV

train_dqn.py / train_dqn_repr.py  ← downstream reward-shaping experiment (currently broken, see §7)
```

Everything under `log/laprepr/<Env>/test/` is output: checkpoints, `repr_convergence.csv`,
`log.txt`, `flags.yaml`, and `.npz` representation snapshots.

---

## 2. The environment — where partial observability lives

### `rl_lap/envs/gridworld/maze.py`
Pure maze geometry, no RL. Parses ASCII maps into a char array (`' '` = floor, everything else =
wall). Factories: `SquareRoomFactory` (OneRoom), `TwoRoomsFactory`, `MazeFactoryBase(HARD_MAZE)`.
The `Maze` class indexes empty cells, answers `is_empty(pos)`, and `render()`s a binary
wall/floor array. **`all_empty_grids()` is the canonical list of true underlying states** — this is
what `visualize_reprs.py` uses to sweep the whole state space.

### `rl_lap/envs/gridworld/maze2d_base.py` ← **your partial-observability edit is here**
`Maze2DBase` is the task: 5 actions (up/down/left/right/stay), 50-step episodes, random start.

The change is entirely inside `pos_to_obs()` (lines 88–125):

1. Render the full maze to RGB (walls black, floor white).
2. Paint the agent red at its own cell.
3. Pad the whole map by 2 cells of black (so out-of-bounds reads as wall).
4. **Crop a 5×5×3 window centred on the agent.**
5. Return `ObservationType(image=<5×5×3 local view>, position=<normalised (x,y)>, index=<one-hot>)`.

So yes — you did make it partially observable, and this is the right place for it. Note two
properties of this specific design:

- **The view is egocentric and always centred.** The agent's red pixel is always at (2,2). The
  observation therefore encodes *only the local wall pattern*, and carries no absolute position
  information at all.
- **The aliasing is severe.** In OneRoom (15×15 interior, walls on the border), any cell with
  row and column both in `[3, 13]` sees an all-white window with a red centre — literally the same
  array. That's **121 of 225 states collapsed into one observation** (~54%). TwoRoom adds one
  interior wall, HardMaze is the least aliased of the three.

`position` and `index` are still computed and returned but are **not used by the learner** —
`_obs_prepro` in the config throws them away and keeps only `obs.agent.image`. They are only there
for logging and for the ground-truth analysis in `visualize_reprs.py`.

### `rl_lap/envs/gridworld/maze2d_single_goal.py`
Adds a goal cell and reward (`-1` everywhere, `0` at goal, `end_at_goal=False`). Wraps the
observation as `ObservationType(agent=<local view>, goal=<local view around the goal>)`. For
Laplacian representation learning the reward is irrelevant (data is collected under a random
policy); this only matters for the DQN experiments.

### `rl_lap/envs/gridworld/gridworld_envs.py`
Instantiates the three named environments (`OneRoom`, `TwoRoom`, `HardMaze`) with their goal
positions, and exposes `make(env_id)`.

### `rl_lap/envs/env_base.py`
Framework plumbing: `TimeStep(observation, reward, is_final, is_last, info)`,
`DiscreteActionSpec`, the abstract `Task`, and `Environment` which drives `Task` and handles
episode termination / auto-reset.

### `rl_lap/envs/actors.py`
`StepActor.get_steps(n, policy_fn)` rolls the environment forward `n` steps under a policy and
returns a flat list of `Step(time_step, action, context)`. The `context` slot is the hook that a
recurrent policy *would* use to carry hidden state — it is currently always `None`.

### `rl_lap/envs/evaluator.py`, `rl_lap/envs/gym_wrapper.py`
Test-time episode runner (DQN only) and an unused OpenAI-Gym adapter.

---

## 3. The replay buffer — where the two discount factors are implemented

### `rl_lap/agent/episodic_replay_buffer.py`

Stores **whole episodes**, not loose transitions, because the Laplacian objective needs to sample
*pairs of states from the same trajectory at a discounted time offset*.

`discounted_sampling(ranges, discount)` draws a jump length `k` with
`p(k) ∝ (1-γ)γ^k`, truncated to the remaining length of the episode, by inverse-CDF sampling.
You generalised it to accept a **vector** of discounts (one per batch element) rather than a scalar.

`sample_pairs(batch_size, discount=[γ_short, γ_long])` is the heart of your modification:

1. Sample `batch_size` episodes and a uniformly-random start index `t` in each.
2. **Duplicate** those indices, so you have `2 × batch_size` entries.
3. Assign `γ_short` to the first half and `γ_long` to the second half.
4. Draw a jump `k` per entry with the appropriate γ, take `s' = s_{t+k}`.
5. Split back into `s = (s_short, s_long)`, `ns = (ns_short, ns_long)`.

The crucial point: **both halves start from the *same* `t` indices**. So `s1_short` and `s1_long`
are the same states; only the *second* elements of the pairs differ. Small γ (0.1) → `s'` is almost
always the immediate successor; large γ (0.9) → `s'` is drawn from far down the trajectory.

`sample_steps` returns i.i.d. states for the negative (repulsion) term.

> ⚠️ The default argument is `discount=[0.0,]`, a one-element list, which would `IndexError` on
> `discount[1]`. It only works because every caller passes two values.

---

## 4. The encoder

### `rl_lap/configs/networks.py`

`RNN`: flattens the 5×5×3 image (75 values) → `Linear(75, 256)` → ReLU → 2-layer
`LSTM(256, 256, batch_first=True)` → returns the **last timestep's** output.

`ReprNetRNN` = `RNN` + `Linear(256, d=20)`. This is the representation network φ.
`DiscreteQNetRNN` = same encoder + `Linear(256, n_actions)`, for DQN.

The LSTM is the intended answer to partial observability: integrate a history of local views to
disambiguate aliased observations.

> ⚠️ **The recurrence is currently inert.** Every call site feeds a sequence of length 1: the
> buffer returns single states, `forward()` does `if x.dim() == 4: x = x.unsqueeze(1)`, and
> `hidden` is never threaded from one call to the next (it defaults to zeros every time). The LSTM
> therefore sees one frame with no history and the network is functionally a 4-layer MLP.
> To actually get memory you would need `sample_pairs`/`sample_steps` to return **sequences**
> (`[B, T, C, H, W]` with `T > 1`), and to carry `hidden` across steps in the actor.
> This matters a lot for your research question — right now you are testing "can a memoryless
> encoder recover the Laplacian from aliased observations?", for which the answer is provably no
> for the aliased states.

---

## 5. The learner — `rl_lap/agent/laprepr.py`

This is the file that most encodes your experiment.

**Losses (module level).**
- `pos_loss(x1, x2)` = `E‖φ(s) − φ(s')‖²` — attraction between temporally close states.
- `neg_loss(x, c, reg)` — a sample-based estimator of `‖E[φφᵀ] − (c/d)I‖²`, pushing the embedding
  toward being isotropic/orthonormal so it can't collapse. `reg` adds an extra penalty pinning
  `‖φ‖² → c`.

Together these are the "graph drawing" relaxation of the Laplacian eigenvector problem.

**`LapReprLearner`.**

| Method | What it does |
|---|---|
| `_build_model` | Creates **two independent `ReprNetRNN` instances**: `_repr_fn_short`, `_repr_fn_long`. Different random init, no weight sharing. |
| `_build_optimizer` | One Adam per network. |
| `_get_train_batch` | Calls `sample_pairs(..., discount=[0.1, 0.9])` and packs `s1_short, s2_short, s1_long, s2_long, s_neg` into a `Flags` container. **γ = [0.1, 0.9] is hard-coded here** — `flags.discount = 0.9` in the config is saved to `flags.yaml` but never actually used. |
| `_build_loss` | `loss_short = pos_loss(φ_s(s1_s), φ_s(s2_s)) + w_neg · neg_loss(φ_s(s_neg))`, and symmetrically for long. Then, if `use_reg`, adds a **coupling term** `λ · MSE(φ_short(s1_short), φ_long(s1_short))` to *both* losses. |
| `_train_step` | Computes cosine similarity between `φ_short(s1_short)` and `φ_long(s1_short)` (the headline metric), then backprops `loss_short + loss_long` once and steps both optimizers. |
| `train` | Collects `n_samples=30000` random-policy steps into the buffer, then runs `total_train_steps=50000` gradient steps, logging to `log.txt` and appending `step,loss_s,loss_l,cos_sim` to `repr_convergence.csv` every `print_freq`. |
| `save_ckpt` | Writes `model_short.ckpt` and `model_long.ckpt` (the base path's `.ckpt` is rewritten with a `_short`/`_long` suffix). |

`LapReprConfig` at the bottom is the base config class: builds env/model/optimizer factories and
marshals flags into the learner's kwargs.

**Notes on the coupling regularizer.** Because `reg_loss` is added to *both* `total_loss_short` and
`total_loss_long`, and then `_train_step` optimizes their **sum**, the coupling term is counted
twice — the effective multiplier is `2λ = 0.02`, not `0.01`. Also, since a single
`total_loss.backward()` populates gradients for both networks and both optimizers then step, the
two "separate" optimizers behave exactly like one joint optimizer.

---

## 6. Config, entry point, and analysis

### `rl_lap/configs/laprepr_config_gridworld.py`
Your tuned hyperparameters, with your edits annotated in comments:
`d=20`, `n_samples=30000`, `batch_size=256`, `w_neg=15.0`, `reg_neg=0.1`, `lagrange_mult=0.01`,
`lr=3e-4`, `total_train_steps=50000`, device `mps`. `_obs_prepro` extracts `obs.agent.image`,
transposes HWC→CHW, and divides by 255.

> ⚠️ `pos_to_obs` builds the image from float colour constants already in `[0, 1]`, then
> `_obs_prepro` divides by 255 again. Inputs are therefore in `[0, 0.0039]` — a ~255× attenuation
> before the first `Linear`. Not fatal (the layer can compensate) but it slows early learning and
> is almost certainly unintended.

### `train_laprepr.py`
CLI → build config → `LapReprLearner(**cfg.args)` → `train()` → `save_dual_discount_representations()`,
which dumps one batch of embeddings to `dual_discount_representations.npz` and a
`repr_snapshot_step_N.npz`, and prints the final cosine similarity.

> ⚠️ Two things in that function: the RNN-unsqueeze block computes `s1_short`, `s2_short`, … and
> then never uses them (the calls below pass `batch.s1_short` etc. directly). And the saved key
> `'s1_long'` actually holds `φ_long(s1_short)`, not `φ_long(s1_long)` — deliberate if you want a
> common input, but the name is misleading.

### `visualize_reprs.py` — **the most scientifically informative script here**
1. Loads `model_short.ckpt` / `model_long.ckpt`.
2. Sweeps **every empty grid cell**, builds its observation, and embeds it.
3. Independently builds the **ground-truth graph Laplacian** `L = D − A` over the maze's 4-connected
   adjacency and eigendecomposes it (`compute_laplacian_eigenvectors`).
4. `compute_correlation` scores each learned dimension by its best absolute Spearman correlation
   against any true eigenvector.
5. Plots: L2-distance-to-goal heatmaps per model, per-dimension eigenmaps, and a bar chart of
   short vs. long spectral alignment (`<Env>_gt_comparison.png`).

### The plotting scripts
- `plot_convergence.py` — walks `./log/laprepr`, plots raw + smoothed `cos_sim` per env with an
  annotated "specialization zone", saves `convergence_plot.png` next to each CSV.
- `plot_new_convergence.py` — the same data as a paper-style multi-panel grid
  (`log/laprepr/dynamic_paper_grid.png`).
- `check_repr.py` — a simpler single-line cosine-similarity plot per env.
- `plot_training_losses.py` — regexes `log.txt` for loss fields. **Its regexes are stale**: it looks
  for `loss_neg_short` / `loss_neg_long`, but `laprepr.py` writes `loss_negative_short` /
  `loss_negative_long`, and for a literal `Step N` which the summary formatter does not emit. It
  will find nothing.
- `plot_curves.py`, `plot_success_rate.py`, `plot_success_rate2.py` — for the DQN reward-shaping
  results (`results.csv`), not the representation experiment.

### `rl_lap/tools/`
`flag_tools.py` (the `Flags` bag, YAML save/load, `--args="k=v"` parsing), `torch_tools.py`
(numpy→tensor), `py_tools.py` (the `@store_args` decorator that turns every `__init__` kwarg into
`self._kwarg`), `logging_tools.py`, `summary_tools.py`, `timer_tools.py`.

---

## 7. The DQN branch (downstream, currently broken)

`agent.py` (generic train loop) → `dqn_agent.py` (standard DQN with a soft-updated target net) →
`dqn_repr_agent.py` (your dual-discount version). `DqnReprAgent` trains the Q-function *and* both
representation networks online, and uses `reward_mode` to shape rewards:
`sparse` (unshaped), `l2` (raw pixel distance to goal), `rawmix` (blend of raw distance + reward),
`mix` (blend using `concat[φ_short, φ_long]` distance).

Four blockers if you try to run `./run_dqn_repr.sh`:

1. `train_dqn_repr.py:48` — `flags.env_id = FLAGS.env_idf` (typo) → immediate `AttributeError`.
2. `run_dqn_repr.sh` passes `--repr_ckpt_sub_path`, which `train_dqn_repr.py` no longer defines →
   argparse error.
3. `dqn_repr_config_gridworld.py:70` references `networks.ReprNetCNN`, which does not exist
   (`networks.py` only has `ReprNetRNN`).
4. Neither DQN config sets `flags.n_layers` / `flags.n_units`, which both `_q_model_factory`
   implementations read.

Also in `agent.py`, `self._replay_buffer.add_steps(steps)` sits *outside* the
`if (step+1) % replay_update_freq == 0` block, so the same steps are re-added on every iteration.

---

## 8. What the current results say

From `log/laprepr/*/test/repr_convergence.csv`:

| Env | cos_sim @1k | cos_sim @50k | loss_short @50k | loss_long @50k |
|---|---|---|---|---|
| OneRoom  | ~1.00 | **0.74** | 0.478 | 0.634 |
| HardMaze | ~1.00 | **0.27** | 0.163 | 0.415 |

The shape is consistent across envs: both networks first collapse to a near-constant embedding
(losses pinned at ~0.72, cosine ≈ 1.0 because everything maps to the same vector), then escape
collapse as the negative term bites and the two embeddings diverge.

**One caveat you should weigh before concluding "different representations".** The Laplacian
objective used here is **invariant to orthogonal rotation** of the embedding: replacing φ with `Rφ`
for any orthogonal `R` leaves `pos_loss` unchanged and leaves `E[φφᵀ] = (c/d)I` unchanged. Two
networks can therefore learn *exactly the same d-dimensional subspace* — the same representation in
every sense that matters — and still have near-zero cosine similarity, because they landed on
different bases for it. In 20 dimensions, random vectors have expected |cos| ≈ `1/√20` ≈ 0.22, which
is very close to HardMaze's 0.27. So the cosine number alone cannot distinguish "the two discounts
learned different things" from "the two discounts learned the same thing in different coordinates."

The rotation-invariant tests are the ones in `visualize_reprs.py` (correlation against the true
eigenvectors), plus subspace measures like CCA / principal angles / Procrustes-aligned distance
between the two embeddings. If you want to make the "singular representation is not learnable under
partial observability" claim rigorous, those are the numbers to report — the cosine curve is best
used as a training diagnostic, not as the result.

---

## 9. Summary of issues found

| # | File | Issue |
|---|---|---|
| 1 | `networks.py` + all call sites | LSTM always receives sequence length 1 and never carries `hidden`; the recurrence does nothing. |
| 2 | `laprepr_config_gridworld.py` | Double normalization — colours already in `[0,1]` divided by 255 again. |
| 3 | `laprepr.py` | γ = `[0.1, 0.9]` hard-coded in `_get_train_batch`; `flags.discount` unused. |
| 4 | `laprepr.py` | Coupling term added to both losses then summed → effective λ is 2×. |
| 5 | `train_laprepr.py` | Dead unsqueeze block; `'s1_long'` key actually stores `φ_long(s1_short)`. |
| 6 | `episodic_replay_buffer.py` | `sample_pairs` default `discount=[0.0,]` would `IndexError`. |
| 7 | `agent.py` | `add_steps` outside its `if`, duplicating buffer inserts every step. |
| 8 | `train_dqn_repr.py` | `FLAGS.env_idf` typo. |
| 9 | `run_dqn_repr.sh` | Passes a `--repr_ckpt_sub_path` flag that no longer exists. |
| 10 | `dqn_repr_config_gridworld.py` | References nonexistent `networks.ReprNetCNN`; `n_layers`/`n_units` never set. |
| 11 | `plot_training_losses.py` | Regexes don't match the actual log format. |

Items 1 and 2 are the ones most likely to be affecting your scientific conclusion; 8–10 only block
the downstream reward-shaping experiment.
