#!/usr/bin/env python

import sys

from graph import Node, Graph
from LAOStar import LAOStar
from models.grid_model import GRIDModel
from models.LAO_paper_model import LAOModel




if __name__ == '__main__':


    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    # model = LAOModel()

    algo = LAOStar(model)

    algo.solve()
