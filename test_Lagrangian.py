#!/usr/bin/env python

import sys

from graph import Node, Graph
from LAOStar import LAOStar
from models.grid_model import GRIDModel
from models.LAO_paper_model import LAOModel
import matplotlib.pyplot as plt


def linspace(start, stop, n):
    if n == 1:
        yield stop
        return
    h = (stop - start) / (n - 1)
    for i in range(n):
        yield start + h * i



if __name__ == '__main__':


    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    # model = LAOModel()

    # algo = LAOStar(model)

    alpha_list = list(linspace(0,10000,30))

    alpha_list = [200]
    weighted_value_list = []

    bound = 2

    for a in alpha_list:

        algo = LAOStar(model,constrained=True,bounds=[bound],alpha=[a],Lagrangian=True)

        policy = algo.solve()

        value = algo.graph.root.value
        print(value)
        weighted_value = value[0] + a*(value[1] - bound)
        weighted_value_list.append(weighted_value)

        model.print_policy(policy)


    print(algo.compute_value(algo.graph.nodes[(4,2)],'D'))
    print(algo.compute_value(algo.graph.nodes[(4,2)],'U'))
    print(algo.compute_value(algo.graph.nodes[(4,2)],'R'))
    print(algo.compute_value(algo.graph.nodes[(4,2)],'L'))


        
    # print(alpha_list)
    # print(weighted_value_list)
        
    plt.plot(alpha_list, weighted_value_list,'*')

    plt.show()
