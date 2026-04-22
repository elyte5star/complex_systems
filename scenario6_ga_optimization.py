"""
══════════════════════════════════════════════════════════════════
SCENARIO 6: Genetic Algorithm Optimization (PyGAD)
══════════════════════════════════════════════════════════════════

Goal
----
Use a Genetic Algorithm (PyGAD library) to identify the combination
of intervention parameters that minimizes the total cumulative
number of agents passing through the Infectious compartment.

Genes (3 real-valued parameters)
    1.  beta            (transmission probability)   [0.05, 0.50]
    2.  epsilon         (agent mobility speed)       [0.005, 0.030]
    3.  immune_fraction (initial Recovered fraction) [0.00, 0.60]

Fitness
    Each candidate (beta, epsilon, immune_fraction) is plugged into
    a headless agent-based SEIR simulation. The fitness is the
    NEGATIVE total cumulative infections, because PyGAD maximizes.

Reference
    The optimum found here is compared against the literature-
    informed Scenario 5 parameters (beta=0.1, epsilon=0.01, 40%
    immunity) to quantify the gap between empirically observed
    mitigation strategies and the mathematically optimal strategy
    [Stanovov et al. 2022; Zheng et al. 2023].

Run
    pip install pygad
    python scenario6_ga_optimization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pygad

# ──────────────────────────────────────────────────────────────
# Simulation constants (shared with scenarios 1–5)
# ──────────────────────────────────────────────────────────────
N_AGENTS        = 500
T_MAX           = 80          # days per simulation
R_TRANSMISSION  = 0.05        # spatial transmission radius
P_REC           = 0.1         # gamma  = 1/10
P_EXP           = 0.192       # sigma  = 1/5.2
N_INFECTED_INIT = 5           # initial infectious seeds

# ──────────────────────────────────────────────────────────────
# Headless ABM-SEIR simulation (vectorised for GA throughput)
# ──────────────────────────────────────────────────────────────
def run_simulation(beta, epsilon, immune_fraction,
                   seed=0, t_max=T_MAX, track=False):
    """
    Run one SEIR agent-based simulation and return total cumulative
    infections (i.e. agents that left S during the run).

    If track=True, also return SEIR count trajectories over time.
    """
    rng = np.random.default_rng(seed)

    n_vac = int(N_AGENTS * immune_fraction)
    n_inf = N_INFECTED_INIT
    # Compose initial state: 0=S 1=E 2=I 3=R
    state = np.zeros(N_AGENTS, dtype=np.int8)
    state[:n_inf] = 2
    state[n_inf:n_inf + n_vac] = 3

    initial_S = int((state == 0).sum())

    # Random 2-D positions; velocity = unit vector * epsilon
    pos   = rng.random((N_AGENTS, 2))
    angle = rng.random(N_AGENTS) * 2 * np.pi
    vel   = np.column_stack([np.cos(angle), np.sin(angle)]) * epsilon

    if track:
        S_hist = [int((state == 0).sum())]
        E_hist = [int((state == 1).sum())]
        I_hist = [int((state == 2).sum())]
        R_hist = [int((state == 3).sum())]

    for _ in range(t_max):
        # Random-walk move with reflecting boundaries
        pos += vel
        low_hit  = pos < 0
        high_hit = pos > 1
        pos[low_hit]  = 0
        pos[high_hit] = 1
        vel[low_hit | high_hit] *= -1

        # Transmission: susceptibles that are within r of any infected
        sus_idx = np.where(state == 0)[0]
        inf_idx = np.where(state == 2)[0]
        if sus_idx.size and inf_idx.size:
            diff   = pos[sus_idx, None, :] - pos[inf_idx, None].reshape(1, inf_idx.size, 2)
            dist2  = (diff ** 2).sum(axis=2)
            close  = dist2 < R_TRANSMISSION ** 2
            n_near = close.sum(axis=1)
            # Independent Bernoulli(beta) per infected neighbour
            p_inf  = 1 - (1 - beta) ** n_near
            expose = rng.random(sus_idx.size) < p_inf
            state[sus_idx[expose]] = 1

        # E -> I
        e_idx = np.where(state == 1)[0]
        if e_idx.size:
            state[e_idx[rng.random(e_idx.size) < P_EXP]] = 2

        # I -> R
        i_idx = np.where(state == 2)[0]
        if i_idx.size:
            state[i_idx[rng.random(i_idx.size) < P_REC]] = 3

        if track:
            S_hist.append(int((state == 0).sum()))
            E_hist.append(int((state == 1).sum()))
            I_hist.append(int((state == 2).sum()))
            R_hist.append(int((state == 3).sum()))

        # Early exit if no more exposed or infectious agents
        if not ((state == 1).any() or (state == 2).any()):
            if track:
                # Pad the histories so all runs have equal length
                pad = t_max - (len(S_hist) - 1)
                S_hist += [S_hist[-1]] * pad
                E_hist += [E_hist[-1]] * pad
                I_hist += [I_hist[-1]] * pad
                R_hist += [R_hist[-1]] * pad
            break

    final_S          = int((state == 0).sum())
    total_infections = initial_S - final_S

    if track:
        return total_infections, np.array(S_hist), np.array(E_hist), \
               np.array(I_hist), np.array(R_hist)
    return total_infections


# ──────────────────────────────────────────────────────────────
# Fitness function for PyGAD
# ──────────────────────────────────────────────────────────────
N_SEEDS_AVERAGE = 2   # average over a few seeds to reduce variance

def fitness_function(ga_instance, solution, solution_idx):
    beta, epsilon, immune_fraction = solution
    totals = [run_simulation(beta, epsilon, immune_fraction, seed=s)
              for s in range(N_SEEDS_AVERAGE)]
    mean_infections = float(np.mean(totals))
    return -mean_infections        # PyGAD maximises


# ──────────────────────────────────────────────────────────────
# GA configuration
# ──────────────────────────────────────────────────────────────
gene_space = [
    {'low': 0.05,  'high': 0.50},   # beta
    {'low': 0.005, 'high': 0.030},  # epsilon
    {'low': 0.00,  'high': 0.60},   # immune_fraction
]

# Number of independent GA runs. The best solution across all runs
# is reported, together with mean ± std of the per-run optima, so
# the final answer is robust to the stochasticity of both the
# simulation and the GA search itself.
N_GA_RUNS = 30

def run_ga(seed, verbose=True):
    """Create and run one GA instance with a given random seed.

    Returns
    -------
    best_solution      : np.ndarray of shape (3,)   [beta, eps, immune_fraction]
    best_fitness       : float                       (negative total infections)
    fitness_per_gen    : np.ndarray                  best fitness per generation
    """
    def on_gen(ga):
        if verbose:
            sol, fit, _ = ga.best_solution()
            print(f"    gen {ga.generations_completed:2d} | "
                  f"infections = {-fit:6.1f} | "
                  f"beta={sol[0]:.3f}  eps={sol[1]:.4f}  "
                  f"imm={sol[2]:.3f}")

    ga = pygad.GA(
        num_generations        = 15,
        num_parents_mating     = 4,
        fitness_func           = fitness_function,
        sol_per_pop            = 8,
        num_genes              = 3,
        gene_space             = gene_space,
        gene_type              = float,
        parent_selection_type  = "tournament",
        K_tournament           = 3,
        keep_elitism           = 1,
        crossover_type         = "single_point",
        mutation_type          = "random",
        mutation_percent_genes = 25,
        random_seed            = seed,
        on_generation          = on_gen,
        suppress_warnings      = True,
    )
    ga.run()
    sol, fit, _ = ga.best_solution()
    return np.asarray(sol), float(fit), np.array(ga.best_solutions_fitness)


# ──────────────────────────────────────────────────────────────
# Run the GA N_GA_RUNS times and aggregate
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Running {N_GA_RUNS} independent GA runs "
          f"(8 individuals × 15 generations each)...\n")

    best_solutions   = []
    best_fitnesses   = []
    fitness_curves   = []

    for run_idx in range(N_GA_RUNS):
        print(f"─── GA run {run_idx + 1}/{N_GA_RUNS} "
              f"(random_seed = {run_idx}) ───")
        sol, fit, curve = run_ga(seed=run_idx, verbose=True)
        best_solutions.append(sol)
        best_fitnesses.append(fit)
        fitness_curves.append(curve)
        print(f"  -> best this run: infections = {-fit:.1f}, "
              f"beta={sol[0]:.4f} eps={sol[1]:.4f} imm={sol[2]:.4f}\n")

    best_solutions = np.vstack(best_solutions)                   # (N_GA_RUNS, 3)
    best_infections = -np.asarray(best_fitnesses)                # lower is better

    # Overall best across all 10 runs ----------------------------
    overall_idx            = int(np.argmin(best_infections))
    beta_opt, eps_opt, imm_opt = best_solutions[overall_idx]
    best_total_infections  = float(best_infections[overall_idx])

    # Per-run summary table --------------------------------------
    print("══════════════════════════════════════════════════════")
    print(f"Per-run best solutions ({N_GA_RUNS} independent runs)")
    print("══════════════════════════════════════════════════════")
    print(f"{'run':>3} | {'beta':>6} | {'epsilon':>7} | "
          f"{'immune':>6} | {'infections':>10}")
    print("-" * 50)
    for i, (sol, inf) in enumerate(zip(best_solutions, best_infections)):
        marker = "  <-- best" if i == overall_idx else ""
        print(f"{i:>3} | {sol[0]:>6.4f} | {sol[1]:>7.4f} | "
              f"{sol[2]:>6.4f} | {inf:>10.1f}{marker}")

    # Aggregate statistics (robustness across runs) --------------
    print("\n══════════════════════════════════════════════════════")
    print("Aggregate across runs (mean ± std)")
    print("══════════════════════════════════════════════════════")
    print(f"  beta            : {best_solutions[:, 0].mean():.4f} "
          f"± {best_solutions[:, 0].std():.4f}")
    print(f"  epsilon         : {best_solutions[:, 1].mean():.4f} "
          f"± {best_solutions[:, 1].std():.4f}")
    print(f"  immune_fraction : {best_solutions[:, 2].mean():.4f} "
          f"± {best_solutions[:, 2].std():.4f}")
    print(f"  infections      : {best_infections.mean():.1f} "
          f"± {best_infections.std():.1f}")

    print("\n══════════════════════════════════════════════════════")
    print(f"Overall best (run #{overall_idx})")
    print("══════════════════════════════════════════════════════")
    print(f"  beta (transmission prob)     : {beta_opt:.4f}")
    print(f"  epsilon (mobility)           : {eps_opt:.4f}")
    print(f"  immune_fraction (initial R)  : {imm_opt:.4f}")
    print(f"  mean total infections        : {best_total_infections:.1f}")

    # ------------------------------------------------------------
    # Compare overall best against Scenario 5 literature values
    # ------------------------------------------------------------
    print("\nRunning reference Scenario 5 for comparison...")
    S5_BETA, S5_EPS, S5_IMM = 0.10, 0.01, 0.40

    s5_totals = [run_simulation(S5_BETA, S5_EPS, S5_IMM, seed=s)
                 for s in range(N_SEEDS_AVERAGE)]
    s5_mean = float(np.mean(s5_totals))

    print(f"  Scenario 5 mean total infections : {s5_mean:.1f}")
    print(f"  GA (overall best) infections     : {best_total_infections:.1f}")
    print(f"  Absolute improvement              : "
          f"{s5_mean - best_total_infections:+.1f} agents")
    if s5_mean > 0:
        print(f"  Relative improvement              : "
              f"{100 * (s5_mean - best_total_infections) / s5_mean:+.1f}%")

    # ------------------------------------------------------------
    # Plots: all 10 convergence curves + SEIR comparison
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Convergence plot: faint individual runs + mean ± std band
    # Stack curves into (N_GA_RUNS, num_generations+1) then aggregate.
    curves_arr = np.vstack([-c for c in fitness_curves])   # infections
    gens       = np.arange(curves_arr.shape[1])
    mean_curve = curves_arr.mean(axis=0)
    std_curve  = curves_arr.std(axis=0)

    # Individual runs (very faint, no label — keep legend readable)
    for i, curve in enumerate(fitness_curves):
        if i == overall_idx:
            continue                                       # draw best last
        axes[0].plot(gens, -curve, color="gray", alpha=0.25, lw=0.8)

    # Std band (±1 std around the mean)
    axes[0].fill_between(
        gens, mean_curve - std_curve, mean_curve + std_curve,
        color="#1f77b4", alpha=0.18, label="±1 std",
    )
    # Mean curve
    axes[0].plot(gens, mean_curve, color="#1f77b4", lw=2.5,
                 label=f"mean ({N_GA_RUNS} runs)")
    # Overall best run highlighted on top
    axes[0].plot(gens, -fitness_curves[overall_idx],
                 color="#d62728", lw=2.5, marker="o", markersize=5,
                 label=f"best (run {overall_idx})", zorder=4)

    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best total infections")
    axes[0].set_title(f"GA convergence across {N_GA_RUNS} runs")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # (b) SEIR curves: Scenario 5 vs overall-best GA solution
    _, S_s5, E_s5, I_s5, R_s5 = run_simulation(
        S5_BETA, S5_EPS, S5_IMM, seed=0, track=True)
    _, S_ga, E_ga, I_ga, R_ga = run_simulation(
        beta_opt, eps_opt, imm_opt, seed=0, track=True)

    # Warm hues (red / orange) for Scenario 5, cool hues (blue / green)
    # for the GA optimum, so the two scenarios are easy to tell apart.
    t = np.arange(len(S_s5))
    axes[1].plot(t, I_s5, color="#d62728", linestyle="--", lw=2.0,
                 label="Scenario 5: Infectious")
    axes[1].plot(t, E_s5, color="#ff7f0e", linestyle="--", lw=2.0,
                 label="Scenario 5: Exposed")
    axes[1].plot(t, I_ga, color="#1f77b4", linestyle="-",  lw=2.5,
                 label="GA optimum: Infectious")
    axes[1].plot(t, E_ga, color="#2ca02c", linestyle="-",  lw=2.5,
                 label="GA optimum: Exposed")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Number of agents")
    axes[1].set_title("Scenario 5 vs GA-optimised (overall best)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
