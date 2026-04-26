import pycxsimulator
from pylab import *

# ══════════════════════════════════════════════════════════════════
# SCENARIO 4: Social Distancing (Reduced effective contact radius r)
# Physical distancing is modelled by shrinking the effective
# transmission radius from r = 0.05 to r = 0.02, while keeping
#   - per-contact transmission probability (beta  = 0.5, baseline)
#   - mobility                             (eps   = 0.03, baseline)
# fixed. This isolates the mechanism of distancing (people keep
# further apart -> fewer close-contact events) from masking
# (per-contact risk; Scenario 3) and lockdown (mobility; Scenario 2).
#
# Literature support for reducing r:
#   - Qian & Jiang 2020  (1.5 m droplet reach)
#   - Sun & Zhai 2020    (Wells-Riley distance index, 1.6-3.0 m safe)
#   - Maheshwari & Albert 2020 (network-SIR edge removal; R_t 6 -> <1)
#   - de Souza Melo et al. 2021 (empirical isolation index)
#   - Thu et al. 2020    (cross-country distancing policy efficacy)
#   - Prakash et al. 2022 (Kalman-filter distancing efficacy)
# ══════════════════════════════════════════════════════════════════

n     = 500
v     = 0.03        # baseline mobility (epsilon)
f     = 0.01
r     = 0.02        # REDUCED effective transmission radius (distancing)
k     = int(1/r)+1  # spatial bin count

p_inf = 0.5         # baseline transmission probability (beta)
p_rec = 0.1         # gamma  = 1/10
p_exp = 0.192       # sigma  = 1/5.2

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
    for i in range(5):
        agents[i].s = 2     # 5 Infectious seeds
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
    title('Scenario 4: Social Distancing (r=0.02)\nS=%d E=%d I=%d R=%d' %
          (Scount[-1], Ecount[-1], Icount[-1], Rcount[-1]))

    subplot(1, 2, 2)
    cla()
    plot(Scount, 'c', label='Susceptible')
    plot(Ecount, 'y', label='Exposed')
    plot(Icount, 'r', label='Infectious')
    plot(Rcount, 'g', label='Recovered')
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
                if 0 < d < r:
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
