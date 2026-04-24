import pycxsimulator
from pylab import *

# ══════════════════════════════════════════════════════════════════
# SCENARIO 4: Vaccination Campaign (40% initial immunity)
# 40% of population starts in Recovered (immune) state at t=0
# to represent partial population immunity from vaccination and/or
# prior infection (Pooley et al. 2023; Moore et al. 2025;
# Song et al. 2024). Mobility (epsilon = 0.03) and beta = 0.5 are
# unchanged from baseline so the effect of immunity can be isolated.
# ══════════════════════════════════════════════════════════════════

n     = 500
v     = 0.03        # same mobility as baseline (epsilon = 0.03)
f     = 0.01
r     = 0.05
k     = int(1/r)+1

p_inf = 0.5         # same as baseline
p_rec = 0.1
p_exp = 0.192

vaccinated_frac = 0.4   # 40% start immune

STATE_COLOR = {0:'cyan', 1:'orange', 2:'red', 3:'green'}

class agent:
    def __init__(self):
        self.x  = rand(2)
        self.v  = rand(2) - array([0.5, 0.5])
        self.v *= v / norm(self.v)
        self.s  = 0
    def move(self):
        self.x += self.v
        if self.x[0] < 0: self.x[0] = 0; self.v[0] *= -1
        if self.x[1] < 0: self.x[1] = 0; self.v[1] *= -1
        if self.x[0] > 1: self.x[0] = 1; self.v[0] *= -1
        if self.x[1] > 1: self.x[1] = 1; self.v[1] *= -1

def initialize():
    global agents, Scount, Ecount, Icount, Rcount
    agents = [agent() for i in range(n)]

    n_vaccinated  = int(n * vaccinated_frac)   # 200 agents start immune
    n_infected    = 5

    # assign states
    for i in range(n):
        if i < n_infected:
            agents[i].s = 2   # Infectious
        elif i < n_infected + n_vaccinated:
            agents[i].s = 3   # Recovered / vaccinated
        else:
            agents[i].s = 0   # Susceptible

    Scount = [len([a for a in agents if a.s == 0])]
    Ecount = [0]
    Icount = [n_infected]
    Rcount = [n_vaccinated]

def observe():
    global agents, Scount, Ecount, Icount, Rcount
    subplot(1, 2, 1)
    cla()
    scatter([a.x[0] for a in agents],
            [a.x[1] for a in agents],
            c=[STATE_COLOR[a.s] for a in agents], s=10)
    axis('image')
    axis([0, 1, 0, 1])
    title('Scenario 4: Vaccination 40%%\nS=%d E=%d I=%d R=%d' %
          (Scount[-1], Ecount[-1], Icount[-1], Rcount[-1]))

    subplot(1, 2, 2)
    cla()
    plot(Scount, 'c',  label='Susceptible')
    plot(Ecount, 'y',  label='Exposed')
    plot(Icount, 'r',  label='Infectious')
    plot(Rcount, 'g',  label='Recovered/Vaccinated')
    legend()
    xlabel('Time (days)')
    ylabel('Number of agents')
    title('SEIR Dynamics')
    tight_layout()

def bin(a):
    return int(floor(a.x[0] / r)), int(floor(a.x[1] / r))

def update():
    global agents, Scount, Ecount, Icount, Rcount

    map = [[[] for i in range(k)] for j in range(k)]
    for a in agents:
        i, j = bin(a)
        map[i][j].append(a)

    for a in agents:
        i, j = bin(a)
        nbs = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if 0 <= i+di < k and 0 <= j+dj < k:
                    nbs.extend(map[i+di][j+dj])

        a.a  = array([0., 0.])
        a.ns = a.s

        for b in nbs:
            if a != b:
                d = norm(a.x - b.x)
                if d < r:
                    a.a += f * (a.x - b.x) / d
                    if a.s == 0 and b.s == 2:
                        if random() < p_inf:
                            a.ns = 1

    for a in agents:
        a.v += a.a
        a.v *= v / norm(a.v)
        a.move()

        if a.s == 1:
            if random() < p_exp:
                a.s = 2
        elif a.s == 2:
            if random() < p_rec:
                a.s = 3
        else:
            a.s = a.ns

    Scount.append(len([a for a in agents if a.s == 0]))
    Ecount.append(len([a for a in agents if a.s == 1]))
    Icount.append(len([a for a in agents if a.s == 2]))
    Rcount.append(len([a for a in agents if a.s == 3]))

pycxsimulator.GUI().start(func=[initialize, observe, update])
