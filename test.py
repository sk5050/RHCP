#!/usr/bin/env python

import sys

from graph import Node, Graph
from LAOStar import LAOStar
from models.grid_model import GRIDModel




if __name__ == '__main__':


    init_state = (0,0)
    size = (5,5)
    constraint_states = [(0,1),(1,1),(3,3),(4,3),(0,4)]
    model = GRIDModel(init_state, size, constraint_states, prob_right_transition=0.85, prob_right_observation=0.85)

    algo = LAOStar(model,ub=0.1,alpha=5.0)


    algo.solve()
