import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, computed_field, ConfigDict
from enum import Enum
from math import floor

rng = np.random.default_rng()


class DiseaseParams(BaseModel):
    p_inf: float = 0.5  # infection probability beta
    p_rec: float = 0.1  # recovery probability gamma
    p_exp: float = 0.192  # 1 / 5.2  incubation period ≈ 5.2 days sigma


class AgentHealthState(str, Enum):
    SUSCEPTIBLE = "cyan"
    INFECTED = "orange"
    RECOVERED = "green"
    EXPOSED = "yellow"


class TunableHyperParams(BaseModel):
    agent_speeds: list[float] = [0.001, 0.05]
    repulsion_forces: list[float] = [0.001, 0.1]
    perception_ranges: list[float] = [0.01, 0.05]
    infection_probabilities: list[float] = [0.1, 0.9]
    recovery_probabilities: list[float] = [0.01, 0.1]


# class SimulationState(BaseModel):
#     agents: list[Agent]
#     Scount: list[int]
#     Icount: list[int]
#     Rcount: list[int]


class BaseLineScenerio(DiseaseParams):
    mobility_epsilon: float = 0.03


class LockDownScenarioParams(BaseLineScenerio):
    lock_down_duration: int = 10  # duration of lock down in days
    lock_down_effectiveness: float = 0.8  # effectiveness of lock down
    mobility_epsilon: float = 0.01


class SocialDistancingScenarioParams(BaseModel):
    social_distancing_effectiveness: float = 0.5  # effectiveness of social distancing


class VaccinationScenarioParams(BaseLineScenerio):
    vaccination_rate: float = 0.01  # percentage of population vaccinated per day
    vaccine_efficacy: float = 0.95  # efficacy of the vaccine


class MaskWearingScenarioParams(BaseLineScenerio):
    mask_wearing_effectiveness: float = (
        0.5  # effectiveness of mask wearing in reducing transmission
    )
    p_inf: float = 0.1


class AgentBehavior(BaseModel):
    position_coord: np.ndarray = rng.random(2)
    mobility_epsilon: float = 0.3
    health_state: AgentHealthState = AgentHealthState.SUSCEPTIBLE

    @computed_field
    def velocity_vector(self) -> np.ndarray:
        temp = rng.random(2) - np.array([0.5, 0.5])
        return temp * self.mobility_epsilon / np.linalg.norm(temp)

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
        sim_params: SimulationParams,
        agent: Agent,
        scenerio=None,
    ):
        self.sim_params = sim_params
        self.n_infected = n_infected
        self.agent = agent
        self.agents: list[Agent] = [Agent() for _ in range(sim_params.n_agents)]
        for i in range(self.n_infected):
            self.agents[i].health_state = AgentHealthState.INFECTED
        self.Scount = [sim_params.n_agents - self.n_infected]
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
        plt.subplot(1, 2, 2)
        plt.cla()
        plt.plot(self.Scount, "c", label="Susceptible")
        plt.plot(self.Ecount, "y", label="Exposed")
        plt.plot(self.Icount, "r", label="Infected")
        plt.plot(self.Rcount, "g", label="Recovered")
        plt.legend()
        plt.tight_layout()

    def agent_grid_cell(self, agent: Agent):
        return int(
            floor(agent.position_coord[0] / self.sim_params.transmission_radius)
        ), int(floor(agent.position_coord[1] / self.sim_params.transmission_radius))

    def update(self):
        pass


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
