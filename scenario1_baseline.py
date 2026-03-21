import pycxsimulator
from pylab import *

# ══════════════════════════════════════════════════════════════════
# SCENARIO 1: Baseline (No Intervention)
# High mobility, no masks, no vaccination
# beta=0.5, mobility=0.01
# ══════════════════════════════════════════════════════════════════

n     = 500         # number of agents
v     = 0.03        # speed of agent movement
f     = 0.01        # repulsion force constant
r     = 0.02        # perception / transmission range
k     = int(1/r)+1  # spatial bin count

p_inf = 0.5         # infection probability (beta) - HIGH, no intervention
p_rec = 0.1         # recovery probability (gamma = 1/10)
p_exp = 0.192       # exposed->infectious probability (sigma = 1/5.2)

# States: 0=Susceptible  1=Exposed  2=Infectious  3=Recovered
STATE_COLOR = {0:'cyan', 1:'orange', 2:'red', 3:'green'}

class agent:
    def __init__(self):
        self.x  = rand(2)
        self.v  = rand(2) - array([0.5, 0.5])
        self.v *= v / norm(self.v)
        self.s  = 0   # start Susceptible
    def move(self):
        self.x += self.v
        if self.x[0] < 0: self.x[0] = 0; self.v[0] *= -1
        if self.x[1] < 0: self.x[1] = 0; self.v[1] *= -1
        if self.x[0] > 1: self.x[0] = 1; self.v[0] *= -1
        if self.x[1] > 1: self.x[1] = 1; self.v[1] *= -1

def initialize():
    global agents, Scount, Ecount, Icount, Rcount
    agents = [agent() for i in range(n)]
    # seed 5 infectious agents
    for i in range(5):
        agents[i].s = 2
    Scount = [n - 5]
    Ecount = [0]
    Icount = [5]
    Rcount = [0]

def observe():
    global agents, Scount, Ecount, Icount, Rcount
    subplot(1, 2, 1)
    cla()
    scatter([a.x[0] for a in agents],
            [a.x[1] for a in agents],
            c=[STATE_COLOR[a.s] for a in agents], s=10)
    axis('image')
    axis([0, 1, 0, 1])
    title('Scenario 1: Baseline\nS=%d E=%d I=%d R=%d' %
          (Scount[-1], Ecount[-1], Icount[-1], Rcount[-1]))

    subplot(1, 2, 2)
    cla()
    plot(Scount, 'c',  label='Susceptible')
    plot(Ecount, 'y',  label='Exposed')
    plot(Icount, 'r',  label='Infectious')
    plot(Rcount, 'g',  label='Recovered')
    legend()
    xlabel('Time (days)')
    ylabel('Number of agents')
    title('SEIR Dynamics')
    tight_layout()

def bin(a):
    return int(floor(a.x[0] / r)), int(floor(a.x[1] / r))

def update():
    global agents, Scount, Ecount, Icount, Rcount

    # build spatial bin map
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
                if 0 < d < r:
                    a.a += f * (a.x - b.x) / d
                    # S exposed to I -> becomes E
                    if a.s == 0 and b.s == 2:
                        if random() < p_inf:
                            a.ns = 1   # -> Exposed

    for a in agents:
        a.v += a.a
        a.v *= v / norm(a.v)
        a.move()

        if a.s == 1:                    # Exposed -> Infectious
            if random() < p_exp:
                a.s = 2
        elif a.s == 2:                  # Infectious -> Recovered
            if random() < p_rec:
                a.s = 3
        else:
            a.s = a.ns                  # apply pending S->E transition

    Scount.append(len([a for a in agents if a.s == 0]))
    Ecount.append(len([a for a in agents if a.s == 1]))
    Icount.append(len([a for a in agents if a.s == 2]))
    Rcount.append(len([a for a in agents if a.s == 3]))

pycxsimulator.GUI().start(func=[initialize, observe, update])
