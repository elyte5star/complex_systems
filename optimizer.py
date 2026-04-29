import pygad


from corona_epidemics import (
    TunableHyperParams,
    BaselineScenario,
    SimulationParams,
    EpidemicSimulation,
)


# Goal: minimise peak Infectious count (peak I) on the SEIR curve.
# This GA tunes four real-valued parameters with bounds drawn from
# the same literature-informed ranges as Scenario 7 in the report:
#
#     mobility_epsilon ∈ [0.005, 0.030]
#     transmission_radius ∈ [0.02, 0.05]
#     p_inf            ∈ [0.05, 0.50]
#     immune_fraction  ∈ [0.0, 0.6]
#
# The fourth gene replaces the previous `p_rec` lever — recovery rate
# is a biological property of the virus (~1/10 days), not a policy
# knob, so it is held constant at the literature value while the
# fraction of pre-existing immune individuals is optimised instead.
class EvolutionOptimizer:

    def __init__(
        self,
        hyper_params: TunableHyperParams = TunableHyperParams(),
        population: int = 30,
        generations: int = 20,
        mutation_rate: float = 0.2,
        elites: int = 2,
        sim_days: int = 200,
        n_infected: int = 5,
        n_agents: int = 500,
    ) -> None:
        self.hyper_params = hyper_params
        self.population = population
        self.generations = generations
        self.mutation_rates = mutation_rate
        self.elites = elites
        self.sim_days = sim_days
        self.n_infected = n_infected
        self.n_agents = n_agents

    def simulation(self, solutions: list[float]):
        mobility_epsilon, transmission_radius, p_inf, immune_fraction = solutions
        scenario = BaselineScenario(
            mobility_epsilon=float(mobility_epsilon),
            p_inf=float(p_inf),
            initial_immune_fraction=float(immune_fraction),
        )
        sim_params = SimulationParams(
            n_agents=self.n_agents,
            transmission_radius=float(transmission_radius),
        )

        sim = EpidemicSimulation(
            n_infected=self.n_infected,
            sim_params=sim_params,
            scenario=scenario,
        )
        sim.init()

        for day in range(self.sim_days):
            if sim.state.Icount[-1] == 0 and day > 0:  # stop when epidemic ends
                print(f"Epidemic ended after {day} days.")
                break
            sim.update()

        return sim

    def fitness_function(self, ga_instance, solution, solution_idx):
        sim = self.simulation(solution)
        peak_infected = max(sim.state.Icount)
        return -float(peak_infected)  # PyGAD maximises fitness, hence the negation

    def on_generation(self, ga_instance):
        solution, fitness, _ = ga_instance.best_solution()
        mobility_epsilon, transmission_radius, p_inf, immune_fraction = solution
        print(
            f"Gen {ga_instance.generations_completed:>2}/{self.generations} | "
            f"peak_infected={-fitness:.0f} | "
            f"mobility={mobility_epsilon:.4f} | "
            f"radius={transmission_radius:.4f} | "
            f"p_inf={p_inf:.3f} | "
            f"immune={immune_fraction:.3f}"
        )

    def run(self) -> dict:
        gene_space = [
            {
                "low":  self.hyper_params.mobility_epsilon[0],
                "high": self.hyper_params.mobility_epsilon[1],
            },
            {
                "low":  self.hyper_params.perception_ranges[0],
                "high": self.hyper_params.perception_ranges[1],
            },
            {
                "low":  self.hyper_params.infection_probabilities[0],
                "high": self.hyper_params.infection_probabilities[1],
            },
            {
                "low":  self.hyper_params.immune_fractions[0],
                "high": self.hyper_params.immune_fractions[1],
            },
        ]

        ga = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=self.population // 2,
            sol_per_pop=self.population,
            num_genes=4,
            gene_space=gene_space,
            fitness_func=self.fitness_function,
            on_generation=self.on_generation,
            parent_selection_type="tournament",
            K_tournament=3,
            crossover_type="uniform",
            crossover_probability=0.8,
            mutation_type="random",
            mutation_probability=self.mutation_rates,
            keep_elitism=self.elites,
            random_seed=42,
            suppress_warnings=True,
        )
        ga.run()

        solution, fitness, _ = ga.best_solution()
        mobility_epsilon, transmission_radius, p_inf, immune_fraction = solution

        self.best_solution = {
            "mobility_epsilon":    float(mobility_epsilon),
            "transmission_radius": float(transmission_radius),
            "p_inf":               float(p_inf),
            "immune_fraction":     float(immune_fraction),
            "peak_infected":       -fitness,
        }
        return self.best_solution
