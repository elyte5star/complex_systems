import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, computed_field, ConfigDict, Field
from enum import Enum
from math import floor

# import pycxsimulator

rng = np.random.default_rng()


class DiseaseParams(BaseModel):
    p_inf: float = 0.5  # infection probability beta
    p_rec: float = 0.1  # recovery probability gamma
    p_exp: float = 0.192  # 1 / 5.2  incubation period ≈ 5.2 days sigma


class AgentHealthState(str, Enum):
    SUSCEPTIBLE = "cyan"
    INFECTED = "red"
    RECOVERED = "green"
    EXPOSED = "yellow"


class TunableHyperParams(BaseModel):
    agent_speeds: list[float] = [0.001, 0.05]
    repulsion_forces: list[float] = [0.001, 0.1]
    perception_ranges: list[float] = [0.01, 0.05]
    infection_probabilities: list[float] = [0.1, 0.9]
    recovery_probabilities: list[float] = [0.01, 0.1]


class BaselineScenario(DiseaseParams):
    mobility_epsilon: float = 0.03


class LockDownScenarioParams(BaselineScenario):
    lock_down_duration: int = 10  # duration of lock down in days
    lock_down_effectiveness: float = 0.8  # effectiveness of lock down
    mobility_epsilon: float = 0.01


class SocialDistancingScenarioParams(BaselineScenario):
    social_distancing_effectiveness: float = 0.5  # effectiveness of social distancing


class VaccinationScenarioParams(BaselineScenario):
    vaccination_rate: float = 0.01  # percentage of population vaccinated per day
    vaccine_efficacy: float = 0.95  # efficacy of the vaccine


class MaskWearingScenarioParams(BaselineScenario):
    mask_wearing_effectiveness: float = (
        0.5  # effectiveness of mask wearing in reducing transmission
    )
    p_inf: float = 0.1


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


# class SimulationState(BaseModel):
#     agents: list[Agent]
#     Scount: list[int]
#     Ecount: list[int]
#     Icount: list[int]
#     Rcount: list[int]


class SimulationParams(BaseModel):
    n_agents: int = 1000  # number of agents
    repulsion_force: float = 0.01  # repulsion force constant
    transmission_radius: float = 0.05  # transmission radius

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
            | LockDownScenarioParams
            | VaccinationScenarioParams
            | MaskWearingScenarioParams
            | None
        ) = BaselineScenario(),
    ):
        self.sim_params = sim_params
        self.n_infected = n_infected
        self.disease_params = disease_params
        self.agent = agent
        self.scenario = scenario
        self.agents: list[Agent] = []
        self.day = 0

    def init(self):
        self.agents = [Agent() for _ in range(self.sim_params.n_agents)]
        for i in range(self.n_infected):
            self.agents[i].health_state = AgentHealthState.INFECTED

        self.Scount = [self.sim_params.n_agents - self.n_infected]
        self.Ecount = [0]
        self.Icount = [self.n_infected]
        self.Rcount = [0]

    def observe(self):
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
            f"{self.scenario.__class__.__name__}, S{self.Scount[-1]},I{self.Icount[-1]},E{self.Ecount[-1]},R{self.Rcount[-1]} "
        )
        plt.subplot(1, 2, 2)
        plt.cla()
        plt.plot(self.Scount, "c", label="Susceptible")
        plt.plot(self.Ecount, "y", label="Exposed")
        plt.plot(self.Icount, "r", label="Infected")
        plt.plot(self.Rcount, "g", label="Recovered")
        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Number of agents")
        plt.title("SEIR Dynamics")
        plt.tight_layout()

    def agent_grid_cell(self, agent: Agent):
        return int(
            floor(agent.position_coord[0] / self.sim_params.transmission_radius)
        ), int(floor(agent.position_coord[1] / self.sim_params.transmission_radius))

    def update(self):
        self.day += 1
        transmission_radius = self.sim_params.transmission_radius
        if hasattr(self, "scenario") and self.scenario is not None:
            p_inf = getattr(self.scenario, "p_inf", self.disease_params.p_inf)
            p_exp = getattr(self.scenario, "p_exp", self.disease_params.p_exp)
            p_rec = getattr(self.scenario, "p_rec", self.disease_params.p_rec)
            mobility = getattr(
                self.scenario, "mobility_epsilon", self.agent.mobility_epsilon
            )
        # Lockdown
        if isinstance(self.scenario, LockDownScenarioParams):
            print("Running LockDownScenario")
            if self.day < self.scenario.lock_down_duration:
                mobility *= 1 - self.scenario.lock_down_effectiveness

        # Social distancing
        if isinstance(self.scenario, SocialDistancingScenarioParams):
            print("Running SocialDistancingScenario")
            transmission_radius *= 1 - self.scenario.social_distancing_effectiveness

        # Mask wearing
        if isinstance(self.scenario, MaskWearingScenarioParams):
            print("Running MaskWearingScenario")
            p_inf *= 1 - self.scenario.mask_wearing_effectiveness

        # Vaccination (susceptible -> recovered)
        if isinstance(self.scenario, VaccinationScenarioParams):
            print("Running VaccinationScenario")
            n_vaccinated = int(
                self.scenario.vaccination_rate * len(self.agents),
            )
            susceptible_indices = [
                i
                for i, a in enumerate(self.agents)
                if a.health_state == AgentHealthState.SUSCEPTIBLE
            ]
            if susceptible_indices:
                chosen = rng.choice(
                    susceptible_indices,
                    size=min(n_vaccinated, len(susceptible_indices)),
                    replace=False,
                )
                for index in chosen:
                    if rng.random() < self.scenario.vaccine_efficacy:
                        self.agents[index].health_state = AgentHealthState.RECOVERED

        for agent in self.agents:
            agent.mobility_epsilon = mobility
            norm = np.linalg.norm(agent.velocity_vector)
            if norm > 0:
                agent.velocity_vector = agent.velocity_vector / norm * mobility
            agent.move()

        # Spatial binning
        bins = self.sim_params.bins_per_dimension
        n_agents = len(self.agents)

        bin_map = [[[] for _ in range(bins)] for _ in range(bins)]
        for idx, agent in enumerate(self.agents):
            i, j = self.agent_grid_cell(agent)
            bin_map[i][j].append(idx)

        newly_exposed = [False] * n_agents

        for idx_inf, inf_agent in enumerate(self.agents):
            if inf_agent.health_state != AgentHealthState.INFECTED:
                continue
            i, j = self.agent_grid_cell(inf_agent)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < bins and 0 <= nj < bins:
                        for idx_other in bin_map[ni][nj]:
                            if idx_other == idx_inf:
                                continue
                            other = self.agents[idx_other]
                            if other.health_state != AgentHealthState.SUSCEPTIBLE:
                                continue
                            # Euclidean distance
                            dx = inf_agent.position_coord[0] - other.position_coord[0]
                            dy = inf_agent.position_coord[1] - other.position_coord[1]
                            dist = np.hypot(dx, dy)
                            if dist < transmission_radius:
                                if rng.random() < p_inf:
                                    newly_exposed[idx_other] = True
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

        s = sum(
            1 for a in self.agents if a.health_state == AgentHealthState.SUSCEPTIBLE
        )
        e = sum(1 for a in self.agents if a.health_state == AgentHealthState.EXPOSED)
        i = sum(1 for a in self.agents if a.health_state == AgentHealthState.INFECTED)
        r = sum(1 for a in self.agents if a.health_state == AgentHealthState.RECOVERED)

        self.Scount.append(s)
        self.Ecount.append(e)
        self.Icount.append(i)
        self.Rcount.append(r)


if __name__ == "__main__":
    sim = EpidemicSimulation(n_infected=5, scenario=SocialDistancingScenarioParams())
    sim.init()

    max_days = 200
    for day in range(max_days):
        if sim.Icount[-1] == 0 and day > 0:  # stop when epidemic ends
            print(f"Epidemic ended after {day} days.")
            break
        sim.update()
        sim.observe()

        plt.pause(0.5)

    plt.ioff()
    plt.show()

    # pycxsimulator.GUI().start(func=[sim.init, sim.observe, sim.update])
