# Simple Network Dynamics simulator in Python
#
# *** Network Epidemics ***
#
# Copyright 2008-2012 Hiroki Sayama
# sayama@binghamton.edu

import matplotlib
import numpy as np

matplotlib.use("qt5agg")

import pylab as PL
import random as RD
import scipy as SP
import networkx as NX
import pycxsimulator as pycxsimulator

RD.seed()

populationSize = 500
linkProbability = 0.01
initialInfectedRatio = 0.01
infectionProb = 0.2
recoveryProb = 0.5

susceptible = 0
infected = 1


def init():
    """*** Network Epidemics Model ***"""
    global time, network, positions, nextNetwork

    time = 0

    network = NX.erdos_renyi_graph(populationSize, linkProbability)

    positions = NX.random_layout(network)

    for i in network.nodes():
        if RD.random() < initialInfectedRatio:
            network.nodes[i]["state"] = infected
        else:
            network.nodes[i]["state"] = susceptible

    nextNetwork = network.copy()


def draw():
    PL.cla()
    NX.draw(
        network,
        pos=positions,
        node_color=[network.nodes[i]["state"] for i in network.nodes()],
        with_labels=False,
        edge_color="c",
        cmap=PL.cm.YlOrRd,
        vmin=0,
        vmax=1,
    )
    PL.axis("image")
    PL.title("t = " + str(time))


def step():
    global time, network, nextNetwork

    time += 1

    for i in network.nodes():
        if network.nodes[i]["state"] == susceptible:
            nextNetwork.nodes[i]["state"] = susceptible
            for j in network.neighbors(i):
                if network.nodes[j]["state"] == infected:
                    if RD.random() < infectionProb:
                        nextNetwork.nodes[i]["state"] = infected
                        break
        else:
            if RD.random() < recoveryProb:
                nextNetwork.nodes[i]["state"] = susceptible
            else:
                nextNetwork.nodes[i]["state"] = infected

    network, nextNetwork = nextNetwork, network


pycxsimulator.GUI().start(func=[init, draw, step])
