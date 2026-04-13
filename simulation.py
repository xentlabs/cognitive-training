import os
import random
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import torch

# ============================================================
# Simulation hyper-parameters
# ============================================================

n_skills = 2**6
n_games = 2**8

num_steps = 2048

norm_skill_multiplier = 0.01 # \eta in the paper
power_size = 0.1 # \mu_skill in the paper
strong_scale_sg = 1.0 # \mu_high^-1 in the paper
weak_scale_sg = 0.01 # \mu_low^-1 in the paper
p_strong = (n_games * n_skills) ** (-0.5) # p_high in the paper

eps = 0.01 # to prevent diversity from blowing up

# ============================================================
# Experiment configuration
# ============================================================

OUT_DIR = "plots"
DELTA_GRID = np.linspace(0.0, 1.0, 11).tolist()
N_WORLDS = 64
RECORD_EVERY = max(1, num_steps // 80)
BASE_SEED = 1234

# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

print(f"Using device: {DEVICE}")

# ============================================================
# Seeds
# ============================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Core simulation code
# ============================================================

@dataclass(frozen=True)
class World:
    game_skills: torch.Tensor
    skill_powers: torch.Tensor
    skill_limits: torch.Tensor


class Run:
    def __init__(self, world: World | None = None):
        if world is None:
            self.game_skills = self.init_game_skills()  # [n_games, n_skills]

            self.skill_powers = (
                power_size
                * (torch.randn(n_skills, device=DEVICE, dtype=DTYPE) ** 2)
            )  # [n_skills]

            self.skill_limits = torch.ones(n_skills, device=DEVICE, dtype=DTYPE)  # [n_skills]
        else:
            self.game_skills = world.game_skills.clone()
            self.skill_powers = world.skill_powers.clone()
            self.skill_limits = world.skill_limits.clone()

        self.init_model()

    def init_model(self):
        self.game_i_count = torch.zeros(n_games, device=DEVICE, dtype=DTYPE)  # [n_games]
        self.free_model_skills = torch.zeros(n_skills, device=DEVICE, dtype=DTYPE)  # [n_skills]

        # Useful cache: depends only on the latent world and the empty model
        init_norm_skills = self.comp_norm_skills(self.free_model_skills)
        self.init_scores = self.score_games(self.game_skills, init_norm_skills)[0]

    def init_game_skills(self):
        # Sample directly on the GPU
        weak_game_skills = torch.empty(
            (n_games, n_skills), device=DEVICE, dtype=DTYPE
        ).exponential_(1.0 / weak_scale_sg)

        mask = (torch.rand((n_games, n_skills), device=DEVICE) < p_strong).to(DTYPE)

        strong_game_skills = torch.empty(
            (n_games, n_skills), device=DEVICE, dtype=DTYPE
        ).exponential_(1.0 / strong_scale_sg)

        game_skills = weak_game_skills + mask * strong_game_skills
        return game_skills

    def comp_norm_skills(self, free_skills):
        return self.skill_limits * (
            1.0 - (1.0 + norm_skill_multiplier * free_skills).pow(-self.skill_powers)
        )

    def score_games(self, game_skills, norm_skills, p=1.0):
        if norm_skills.ndim == 1:
            x = norm_skills[None, None, :]   # [1, 1, n_skills]
        else:
            x = norm_skills[:, None, :]      # [batch, 1, n_skills]

        w = game_skills[None, :, :]          # [1, n_games, n_skills]
        denom = w.sum(dim=-1).clamp_min(1e-12)  # [1, n_games]

        return ((w * (x ** p)).sum(dim=-1) / denom).clamp_min(1e-12).pow(1.0 / p)

    def compute_qd_scores(self, delta):
        cur_norm_skills = self.comp_norm_skills(self.free_model_skills)               # [n_skills]
        next_free_skills_batch = self.free_model_skills[None, :] + self.game_skills   # [n_games, n_skills]
        next_norm_skills_batch = self.comp_norm_skills(next_free_skills_batch)        # [n_games, n_skills]

        cur_scores = self.score_games(self.game_skills, cur_norm_skills)[0]           # [n_games]

        transfer_matrix = (
            self.score_games(self.game_skills, next_norm_skills_batch)
            - cur_scores[None, :]
        )  # [n_games, n_games]

        numerator_div = transfer_matrix.diag()
        denominator_div = cur_scores - self.init_scores + eps

        raw_quality = transfer_matrix @ self.game_i_count  # [n_games]

        diversity = (numerator_div / denominator_div.clamp_min(1e-12)).clamp_min(1e-12).pow(delta)
        quality = raw_quality.clamp_min(1e-12).pow(1.0 - delta)

        return quality * diversity

    def pick_game(self, game_id):
        # game_id can be a Python int or a torch scalar on the GPU
        self.free_model_skills += self.game_skills[game_id]
        self.game_i_count[game_id] += 1.0

    def run_meta_game_step(self, delta):
        game_id = torch.argmax(self.compute_qd_scores(delta))  # stays on the GPU
        self.pick_game(game_id)


# ============================================================
# Helpers for figure generation
# ============================================================

def sample_world(seed=None) -> World:
    """
    Sample one latent world once, then reuse it for both greedy and random runs.
    """
    if seed is not None:
        set_all_seeds(seed)
    base = Run()
    return World(
        game_skills=base.game_skills.clone(),
        skill_powers=base.skill_powers.clone(),
        skill_limits=base.skill_limits.clone(),
    )

def compute_metrics(run):
    norm_skills = run.comp_norm_skills(run.free_model_skills)
    scores = run.score_games(run.game_skills, norm_skills).squeeze(0)

    return {
        "mean_score": scores.mean().item(),
        "geo_score": torch.exp(torch.log(scores.clamp_min(1e-12)).mean()).item(),
    }

def compute_num_selected_games(run):
    # Number of games that have been selected at least once so far
    return run.game_i_count.count_nonzero().item()

def run_policy_step(run, policy, delta):
    if policy == "greedy":
        run.run_meta_game_step(delta)
    elif policy == "random":
        game_id = torch.randint(n_games, (), device=DEVICE)
        run.pick_game(game_id)
    else:
        raise ValueError(f"Unknown policy: {policy}")

def run_policy_on_world(world, policy, delta, record_every, seed=None, track_selected_games=False):
    """
    policy = 'greedy' or 'random'
    Returns final metrics + a recorded trajectory.
    """
    if seed is not None:
        set_all_seeds(seed)
    run = Run(world=world)
    delta = float(delta)

    steps = [0]
    if track_selected_games:
        selected_games_curve = [compute_num_selected_games(run)]
    else:
        start = compute_metrics(run)
        geo_curve = [start["geo_score"]]

    with torch.inference_mode():
        for t in range(num_steps):
            run_policy_step(run, policy, delta)

            if (t + 1) % record_every == 0 or (t + 1) == num_steps:
                steps.append(t + 1)
                if track_selected_games:
                    selected_games_curve.append(compute_num_selected_games(run))
                else:
                    geo_curve.append(compute_metrics(run)["geo_score"])

    result = {
        "steps": np.array(steps, dtype=float),
    }
    if track_selected_games:
        result["selected_games_curve"] = np.array(selected_games_curve, dtype=float)
    else:
        end = compute_metrics(run)
        result["geo_curve"] = np.array(geo_curve, dtype=float)
        result["final_geo_score"] = end["geo_score"]

    return result

def mean_and_sem(arrays):
    x = np.asarray(arrays, dtype=float)
    mean = x.mean(axis=0)
    if x.shape[0] == 1:
        sem = np.zeros_like(mean)
    else:
        sem = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    return mean, sem


# ============================================================
# Run the simulations and generate the figures
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)
set_all_seeds(BASE_SEED)

# 1) Sample latent worlds once
worlds = [sample_world() for _ in range(N_WORLDS)]

# 2) Random baseline
random_runs = []
for world in worlds:
    random_runs.append(
        run_policy_on_world(
            world=world,
            policy="random",
            delta=0.0,
            record_every=RECORD_EVERY,
        )
    )

# 3) Greedy QD sweep over delta
greedy_runs_by_delta = {d: [] for d in DELTA_GRID}

for d in DELTA_GRID:
    print(f"Running delta = {d:.2f}")
    for world in worlds:
        greedy_runs_by_delta[d].append(
            run_policy_on_world(
                world=world,
                policy="greedy",
                delta=d,
                record_every=RECORD_EVERY,
            )
        )

# 4) Figure 1: Final model performance across different deltas
fig_delta_performance_path = os.path.join(OUT_DIR, "delta_performance.png")

x = np.array(DELTA_GRID, dtype=float)

greedy_final_geo = np.array(
    [[run["final_geo_score"] for run in greedy_runs_by_delta[d]] for d in DELTA_GRID],
    dtype=float,
)

greedy_geo_mean, greedy_geo_sem = mean_and_sem(greedy_final_geo.T)

random_final_geo = np.array([run["final_geo_score"] for run in random_runs], dtype=float)
random_geo_mean = float(random_final_geo.mean())
if len(random_final_geo) == 1:
    random_geo_sem = 0.0
else:
    random_geo_sem = float(random_final_geo.std(ddof=1) / np.sqrt(len(random_final_geo)))

best_delta_idx = int(np.argmax(greedy_geo_mean))
best_delta = DELTA_GRID[best_delta_idx]

plt.figure(figsize=(7, 4.5))
plt.plot(x, greedy_geo_mean, marker="o", label="Cognitive Training")
plt.fill_between(
    x,
    greedy_geo_mean - greedy_geo_sem,
    greedy_geo_mean + greedy_geo_sem,
    alpha=0.2,
)
plt.plot(
    [0.0, 1.0],
    [random_geo_mean, random_geo_mean],
    linestyle="--",
    label="Random game selection",
)
plt.fill_between(
    [x[0], x[-1]],
    [random_geo_mean - random_geo_sem, random_geo_mean - random_geo_sem],
    [random_geo_mean + random_geo_sem, random_geo_mean + random_geo_sem],
    alpha=0.15,
)
plt.xlabel(r"$\delta$")
plt.ylabel("Final model performance")
plt.title("")
plt.legend()
plt.tight_layout()
plt.savefig(fig_delta_performance_path, dpi=200)
plt.close()
print("Saved:")
print(fig_delta_performance_path)

# 5) Figure 2: Learning curves for the best delta and the random baseline
fig_learning_curves_path = os.path.join(OUT_DIR, "learning_curves_best_delta.png")

best_steps = greedy_runs_by_delta[best_delta][0]["steps"]

best_greedy_curves = np.stack(
    [run["geo_curve"] for run in greedy_runs_by_delta[best_delta]],
    axis=0,
)
best_random_curves = np.stack(
    [run["geo_curve"] for run in random_runs],
    axis=0,
)

best_greedy_mean, best_greedy_sem = mean_and_sem(best_greedy_curves)
best_random_mean, best_random_sem = mean_and_sem(best_random_curves)

plt.figure(figsize=(7, 4.5))
plt.plot(best_steps, best_greedy_mean, label=rf'Cognitive Training ($\delta$={best_delta:.1f})')
plt.fill_between(
    best_steps,
    best_greedy_mean - best_greedy_sem,
    best_greedy_mean + best_greedy_sem,
    alpha=0.2,
)
plt.plot(best_steps, best_random_mean, label="Random game selection")
plt.fill_between(
    best_steps,
    best_random_mean - best_random_sem,
    best_random_mean + best_random_sem,
    alpha=0.2,
)
plt.xlabel("Training step")
plt.ylabel("Model performance")
plt.title("")
plt.legend()
plt.tight_layout()
plt.savefig(fig_learning_curves_path, dpi=200)
plt.close()

print(f"Best delta = {best_delta:.2f}")
print("Saved:")
print(fig_learning_curves_path)

# 6) Figure 3: Number of distinct selected games over time

# Re-run only what is needed for this new visualisation
selected_random_runs = []
selected_greedy_runs = []

for world in worlds:
    selected_random_runs.append(
        run_policy_on_world(
            world=world,
            policy="random",
            delta=0.0,
            record_every=RECORD_EVERY,
            track_selected_games=True,
        )
    )

    selected_greedy_runs.append(
        run_policy_on_world(
            world=world,
            policy="greedy",
            delta=best_delta,
            record_every=RECORD_EVERY,
            track_selected_games=True,
        )
    )

# Aggregate mean and SEM
selected_steps = selected_greedy_runs[0]["steps"]

selected_greedy_curves = np.stack(
    [run["selected_games_curve"] for run in selected_greedy_runs],
    axis=0,
)  # [n_runs, n_points]

selected_random_curves = np.stack(
    [run["selected_games_curve"] for run in selected_random_runs],
    axis=0,
)  # [n_runs, n_points]

selected_greedy_mean, selected_greedy_sem = mean_and_sem(selected_greedy_curves)
selected_random_mean, selected_random_sem = mean_and_sem(selected_random_curves)

# Plot
fig_selected_games_path = os.path.join(OUT_DIR, "selected_games_over_time_best_delta.png")

plt.figure(figsize=(7, 4.5))

plt.fill_between(
    selected_steps,
    selected_greedy_mean - selected_greedy_sem,
    selected_greedy_mean + selected_greedy_sem,
    alpha=0.2,
)
plt.plot(
    selected_steps,
    selected_greedy_mean,
    linewidth=2.5,
    label=rf'Cognitive Training ($\delta$={best_delta:.1f})',
)

plt.fill_between(
    selected_steps,
    selected_random_mean - selected_random_sem,
    selected_random_mean + selected_random_sem,
    alpha=0.2,
)
plt.plot(
    selected_steps,
    selected_random_mean,
    linewidth=2.5,
    linestyle="--",
    label="Random game selection",
)

plt.xlabel("Training step")
plt.ylabel("Number of distinct selected games")
plt.title("")
plt.legend(loc="center right")
plt.tight_layout()
plt.savefig(fig_selected_games_path, dpi=200)
plt.close()

print("Saved:")
print(fig_selected_games_path)
