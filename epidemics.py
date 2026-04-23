import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, computed_field


class Agent:
    def __init__(self):
        rng = np.random.default_rng()
        self.x = rng.random(2)
        self.v = rng.random(2) - np.array([0.5, 0.5])
        self.v *= self.v / np.linalg.norm(self.v)
        self.s = 0

    def move(self):
        self.x += self.v
        if self.x[0] < 0:
            self.x[0] = 0
            self.v[0] *= -1
        if self.x[1] < 0:
            self.x[1] = 0
            self.v[1] *= -1
        if self.x[0] > 1:
            self.x[0] = 1
            self.v[0] *= -1
        if self.x[1] > 1:
            self.x[1] = 1
            self.v[1] *= -1


class SimulationParams(BaseModel):
    n_agents: int = 1000  # number of agents
    repulsion_force: float = 0.01  # repulsion force constant
    transmission_radius: float = 0.05  # transmission radius

    @computed_field
    def bins_per_dimension(self) -> int:
        return int(1 / self.transmission_radius) + 1


class DiseaseParams(BaseModel):
    p_inf: float = 0.5  # infection probability
    p_rec: float = 0.1  # recovery probability
    p_exp: float = 1 / 5.2  # incubation period ≈ 5.2 days


class AgentHealthState:
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    EXPOSED = 3


class TunableHyperParams(BaseModel):
    agent_speeds: list[float] = [0.001, 0.05]
    repulsion_forces: list[float] = [0.001, 0.1]
    perception_ranges: list[float] = [0.01, 0.05]
    infection_probabilities: list[float] = [0.1, 0.9]
    recovery_probabilities: list[float] = [0.01, 0.1]


class SimulationState(BaseModel):
    agents: list[Agent]
    Scount: list[int]
    Icount: list[int]
    Rcount: list[int]


class LockDownScenarioParams(BaseModel):
    lock_down_duration: int = 10  # duration of lock down in days
    lock_down_effectiveness: float = 0.8  # effectiveness of lock down


class SocialDistancingScenarioParams(BaseModel):
    social_distancing_effectiveness: float = 0.5  # effectiveness of social distancing


class VaccinationScenarioParams(BaseModel):
    vaccination_rate: float = 0.01  # percentage of population vaccinated per day
    vaccine_efficacy: float = 0.95  # efficacy of the vaccine


class MaskWearingScenarioParams(BaseModel):
    mask_wearing_effectiveness: float = (
        0.5  # effectiveness of mask wearing in reducing transmission
    )


class AgentBehavior(BaseModel):
    position_coord: tuple[int, int]
    mobility_epsilon: float
    health_state: AgentHealthState


class EpidemicSimulation:
    def __init__(self, sim_params: dict):
        n_agents = sim_params.get("n_agents")
        self.agents = [Agent() for _ in range(n_agents)]
        self.agents[0].s = 1  # Infect the first agent
        self.Scount = [n_agents - 1]
        self.Icount = [1]
        self.Rcount = [0]

    def observe(self):
        plt.subplot(1, 2, 1)
        plt.cla()
        colors = ["c", "r", "g"]
        plt.scatter(
            [a.x[0] for a in self.agents],
            [a.x[1] for a in self.agents],
            c=[colors[a.s] for a in self.agents],
        )
        plt.axis("image")
        plt.axis([0, 1, 0, 1])

        plt.subplot(1, 2, 2)
        plt.cla()
        plt.plot(self.Scount, label="Susceptible")
        plt.plot(self.Icount, label="Infected")
        plt.plot(self.Rcount, label="Recovered")
        plt.legend()
        plt.tight_layout()


# class AgentBehavior(BaseModel):
#     def update_velocity(
#         self, agent: Agent, neighbors: list[Agent], repulsion_force: float
#     ):
#         for neighbor in neighbors:
#             if neighbor is not agent:
#                 direction = agent.x - neighbor.x
#                 distance = np.linalg.norm(direction)
#                 if distance < 0.01:  # Avoid division by zero
#                     continue
#                 force = repulsion_force / distance**2
#                 agent.v += force * direction / distance

#     def update_health_state(
#         self, agent: Agent, neighbors: list[Agent], disease_params: DiseaseParams
#     ):
#         if agent.s == AgentHealthState.INFECTED:
#             if np.random.random() < disease_params.p_rec:
#                 agent.s = AgentHealthState.RECOVERED
#         elif agent.s == AgentHealthState.SUSCEPTIBLE:
#             for neighbor in neighbors:
#                 if neighbor.s == AgentHealthState.INFECTED:
#                     if np.random.random() < disease_params.p_inf:
#                         agent.s = AgentHealthState.INFECTED
#                         break
