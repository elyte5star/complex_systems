import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, computed_field, ConfigDict, Field
from enum import Enum
from math import floor
import pycxsimulator
import sys


rng = np.random.default_rng()


class DiseaseParams(BaseModel):
    p_inf: float = 0.5      # transmission probability beta
    p_rec: float = 0.1      # recovery probability gamma = 1/10
    p_exp: float = 0.192    # 1 / 5.2  incubation period sigma


class AgentHealthState(str, Enum):
    SUSCEPTIBLE = "cyan"
    INFECTED = "red"
    RECOVERED = "green"
    EXPOSED = "yellow"


class TunableHyperParams(BaseModel):
    """GA search bounds aligned with Scenario 7 in the report."""
    mobility_epsilon: list[float]        = [0.005, 0.030]
    perception_ranges: list[float]       = [0.02,  0.05]
    infection_probabilities: list[float] = [0.05,  0.50]
    immune_fractions: list[float]        = [0.0,   0.6]


# ─────────────────────────────────────────────────────────────────
# Scenarios — parameter values are literature-informed (see report
# Sections 4.1 and 5). Each single-mechanism scenario changes one
# knob; CombinedScenario stacks all four.
# ─────────────────────────────────────────────────────────────────

class BaselineScenario(DiseaseParams):
    """Pre-intervention conditions: full mobility, no masks, no immunity."""
    mobility_epsilon: float = 0.03
    initial_immune_fraction: float = 0.0
    social_distancing_effectiveness: float = 0.0


class LockDownScenarioParams(BaselineScenario):
    """Strict lockdown: sustained mobility reduction.

    eps reduced 0.03 -> 0.01 (literature: Hadjidemetriou 2020,
    Vinceti 2020, Shi 2025). The temporal multiplier from the
    earlier prototype has been removed so the effect is sustained
    across the full simulation, matching the report.
    """
    mobility_epsilon: float = 0.01


class SocialDistancingScenarioParams(BaselineScenario):
    """Social distancing: reduces effective transmission radius.

    r_effective = r * (1 - effectiveness); 0.6 produces r = 0.02
    (Qian 2020, Sun 2020, Maheshwari 2020, de Souza Melo 2021,
    Thu 2020, Prakash 2022).
    """
    social_distancing_effectiveness: float = 0.6


class VaccinationScenarioParams(BaselineScenario):
    """Vaccination campaign: 40% pre-existing immunity at t=0.

    Replaces the earlier daily-rollout model (1%/day, 95% efficacy)
    with the report's literature-informed initial-immunity
    formulation (Pooley 2023, Moore 2025, Song 2024).
    """
    initial_immune_fraction: float = 0.40


class MaskWearingScenarioParams(BaselineScenario):
    """High mask compliance: per-contact transmission probability
    lowered directly to 0.10 (Chu 2020, Li 2021, MacIntyre 2020).
    The old compound effect (p_inf=0.1 * effectiveness=0.5) is
    removed so the literature value is reached directly.
    """
    p_inf: float = 0.10


class CombinedScenario(BaselineScenario):
    """Combined real-world intervention stacking all four levers
    (Section 4.1(6) of the report)."""
    p_inf: float = 0.10                              # masks
    mobility_epsilon: float = 0.01                   # lockdown
    social_distancing_effectiveness: float = 0.6     # distancing -> r=0.02
    initial_immune_fraction: float = 0.40            # vaccination


# ─────────────────────────────────────────────────────────────────
# Agents and simulation
# ─────────────────────────────────────────────────────────────────

class AgentBehavior(BaseModel):
    position_coord: np.ndarray = Field(default_factory=lambda: rng.random(2))
    mobility_epsilon: float = 0.3
    health_state: AgentHealthState = AgentHealthState.SUSCEPTIBLE
    velocity_vector: np.ndarray = Field(
        default_factory=lambda: rng.random(2) - np.array([0.5, 0.5])
    )

    def model_post_init(self, ctx):
        self.velocity_vector *= self.mobility_epsilon / np.linalg.norm(
            self.velocity_vector
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.ndarray: lambda v: v.tolist(),
        },
    )


class Agent(AgentBehavior):

    def move(self):
        self.position_coord += self.velocity_vector
        if self.position_coord[0] < 0:
            self.position_coord[0] = 0
            self.velocity_vector[0] *= -1
        if self.position_coord[1] < 0:
            self.position_coord[1] = 0
            self.velocity_vector[1] *= -1
        if self.position_coord[0] > 1:
            self.position_coord[0] = 1
            self.velocity_vector[0] *= -1
        if self.position_coord[1] > 1:
            self.position_coord[1] = 1
            self.velocity_vector[1] *= -1


class SimulationState(BaseModel):
    Agents: list[Agent]
    Scount: list[int]
    Ecount: list[int]
    Icount: list[int]
    Rcount: list[int]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SimulationParams(BaseModel):
    n_agents: int = 500           # report uses N=500
    repulsion_force: float = 0.01
    transmission_radius: float = 0.05

    @computed_field
    def bins_per_dimension(self) -> int:
        return int(1 / self.transmission_radius) + 1


class EpidemicSimulation:

    def __init__(
        self,
        n_infected: int,
        sim_params: SimulationParams = SimulationParams(),
        disease_params: DiseaseParams = DiseaseParams(),
        agent: Agent = Agent(),
        scenario: (
            BaselineScenario
            | SocialDistancingScenarioParams
            | LockDownScenarioParams
            | VaccinationScenarioParams
            | MaskWearingScenarioParams
            | CombinedScenario
            | None
        ) = None,
    ):
        self.sim_params = sim_params
        self.n_infected = n_infected
        self.disease_params = disease_params
        self.agent = agent
        self.scenario = scenario if scenario is not None else BaselineScenario()
        self.agents: list[Agent] = []
        self.day = 0

    def init(self):
        self.agents = [Agent() for _ in range(self.sim_params.n_agents)]

        # Seed initial infectious agents
        for i in range(self.n_infected):
            self.agents[i].health_state = AgentHealthState.INFECTED

        # Seed initial immune fraction (vaccination / pre-existing immunity)
        immune_frac = getattr(self.scenario, "initial_immune_fraction", 0.0)
        n_immune = int(self.sim_params.n_agents * immune_frac)
        for j in range(self.n_infected, self.n_infected + n_immune):
            if j < self.sim_params.n_agents:
                self.agents[j].health_state = AgentHealthState.RECOVERED

        s = sum(1 for a in self.agents if a.health_state == AgentHealthState.SUSCEPTIBLE)
        e = sum(1 for a in self.agents if a.health_state == AgentHealthState.EXPOSED)
        i = sum(1 for a in self.agents if a.health_state == AgentHealthState.INFECTED)
        r = sum(1 for a in self.agents if a.health_state == AgentHealthState.RECOVERED)

        self.state = SimulationState(
            Agents=self.agents,
            Scount=[s],
            Ecount=[e],
            Icount=[i],
            Rcount=[r],
        )
        self.day = 0

    def observe(self):
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        plt.subplot(1, 2, 1)
        plt.cla()
        plt.scatter(
            [a.position_coord[0] for a in self.agents],
            [a.position_coord[1] for a in self.agents],
            c=[a.health_state.value for a in self.agents],
        )
        plt.axis("image")
        plt.axis([0, 1, 0, 1])
        plt.title(
            f"{self.scenario.__class__.__name__}, "
            f"S{self.state.Scount[-1]}, I{self.state.Icount[-1]}, "
            f"E{self.state.Ecount[-1]}, R{self.state.Rcount[-1]} "
        )
        plt.subplot(1, 2, 2)
        plt.cla()
        plt.plot(self.state.Scount, "c", label="Susceptible")
        plt.plot(self.state.Ecount, "y", label="Exposed")
        plt.plot(self.state.Icount, "r", label="Infected")
        plt.plot(self.state.Rcount, "g", label="Recovered")
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Number of agents")
        plt.title("SEIR Dynamics")
        plt.tight_layout()

    def agent_grid_cell(self, agent: Agent):
        return (
            int(floor(agent.position_coord[0] / self.sim_params.transmission_radius)),
            int(floor(agent.position_coord[1] / self.sim_params.transmission_radius)),
        )

    def update(self):
        self.day += 1

        # Pull per-scenario parameter values (with sensible fallbacks).
        # Using getattr means CombinedScenario picks up *all* the
        # relevant fields automatically without an isinstance maze.
        transmission_radius = self.sim_params.transmission_radius
        if hasattr(self, "scenario") and self.scenario is not None:
            p_inf    = getattr(self.scenario, "p_inf",    self.disease_params.p_inf)
            p_exp    = getattr(self.scenario, "p_exp",    self.disease_params.p_exp)
            p_rec    = getattr(self.scenario, "p_rec",    self.disease_params.p_rec)
            mobility = getattr(self.scenario, "mobility_epsilon",
                               self.agent.mobility_epsilon)
        else:
            if not hasattr(self, "disease_params"):
                self.disease_params = DiseaseParams()
            p_inf    = self.disease_params.p_inf
            p_exp    = self.disease_params.p_exp
            p_rec    = self.disease_params.p_rec
            mobility = self.agent.mobility_epsilon

        # Social-distancing radius reduction. Applies whenever the
        # scenario carries `social_distancing_effectiveness > 0`,
        # which covers SocialDistancingScenarioParams *and*
        # CombinedScenario.
        sd_eff = getattr(self.scenario, "social_distancing_effectiveness", 0.0)
        if sd_eff > 0:
            transmission_radius = transmission_radius * (1 - sd_eff)

        # Spatial binning
        bins = self.sim_params.bins_per_dimension
        n_agents = len(self.agents)

        bin_map = [[[] for _ in range(bins)] for _ in range(bins)]
        for idx, agent in enumerate(self.agents):
            i, j = self.agent_grid_cell(agent)
            bin_map[i][j].append(idx)

        forces = [np.zeros(2) for _ in range(n_agents)]
        newly_exposed = [False] * n_agents

        for idx, agent in enumerate(self.agents):
            i, j = self.agent_grid_cell(agent)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < bins and 0 <= nj < bins:
                        for idx_other in bin_map[ni][nj]:
                            if idx_other == idx:
                                continue
                            other = self.agents[idx_other]
                            dx = agent.position_coord[0] - other.position_coord[0]
                            dy = agent.position_coord[1] - other.position_coord[1]
                            dist = np.hypot(dx, dy)
                            if 0 < dist < transmission_radius:
                                forces[idx] += (
                                    self.sim_params.repulsion_force
                                    * np.array([dx, dy])
                                    / dist
                                )

                                if (
                                    agent.health_state == AgentHealthState.SUSCEPTIBLE
                                    and other.health_state == AgentHealthState.INFECTED
                                ):
                                    if rng.random() < p_inf:
                                        newly_exposed[idx] = True

        for idx, agent in enumerate(self.agents):
            agent.mobility_epsilon = mobility
            agent.velocity_vector += forces[idx]
            norm = np.linalg.norm(agent.velocity_vector)
            if norm > 0:
                agent.velocity_vector = agent.velocity_vector / norm * mobility
            agent.move()

        # State transitions
        newly_infected = [False] * n_agents
        newly_recovered = [False] * n_agents
        for i, agent in enumerate(self.agents):
            if agent.health_state == AgentHealthState.EXPOSED and rng.random() < p_exp:
                newly_infected[i] = True
            elif (
                agent.health_state == AgentHealthState.INFECTED and rng.random() < p_rec
            ):
                newly_recovered[i] = True

        for i, agent in enumerate(self.agents):
            if newly_exposed[i]:
                agent.health_state = AgentHealthState.EXPOSED
            elif newly_infected[i]:
                agent.health_state = AgentHealthState.INFECTED
            elif newly_recovered[i]:
                agent.health_state = AgentHealthState.RECOVERED

        s = sum(1 for a in self.agents if a.health_state == AgentHealthState.SUSCEPTIBLE)
        e = sum(1 for a in self.agents if a.health_state == AgentHealthState.EXPOSED)
        i = sum(1 for a in self.agents if a.health_state == AgentHealthState.INFECTED)
        r = sum(1 for a in self.agents if a.health_state == AgentHealthState.RECOVERED)

        self.state.Scount.append(s)
        self.state.Ecount.append(e)
        self.state.Icount.append(i)
        self.state.Rcount.append(r)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--ga" in sys.argv:
        import optimizer
        best_params = optimizer.EvolutionOptimizer(
            population=20, generations=15, n_agents=500, sim_days=200
        ).run()

        sim = EpidemicSimulation(
            n_infected=5,
            sim_params=SimulationParams(
                n_agents=500,
                transmission_radius=best_params["transmission_radius"],
            ),
            scenario=BaselineScenario(
                mobility_epsilon=best_params["mobility_epsilon"],
                p_inf=best_params["p_inf"],
                initial_immune_fraction=best_params["immune_fraction"],
            ),
        )
    else:
        # Default: run the literature-informed Combined scenario
        sim = EpidemicSimulation(
            n_infected=5,
            scenario=CombinedScenario(),
        )

    pycxsimulator.GUI().start(func=[sim.init, sim.observe, sim.update])
