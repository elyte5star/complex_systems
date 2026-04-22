import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel


class SimulationParams(BaseModel):
    n_agents: int = 1000  # number of agents
    agent_speed: float = 0.01  # speed of agent movement
    p_inf: float = 0.01  # probability of infection
    p_rec: float = 0.01  # probability of recovery
    repulsion_force: float = 0.01  # repulsion force constant
    perception_range: float = 0.02  # perception range
    bins_per_dimension: int = int(1 / 0.02) + 1


class Agent:
    def __init__(self):
        rng = np.random.default_rng()
        self.x = rng.random(2)
        self.v = rng.random(2) - np.array([0.5, 0.5])
        self.v *= v / np.linalg.norm(self.v)
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
