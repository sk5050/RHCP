#!/usr/bin/env python

import sys
from utils import import_models
import_models()

from graph import Node, Graph
from LAOStar import LAOStar
from CSSPSolver import CSSPSolver
from grid_model import GRIDModel
from grid_model_multiple_bounds import GRIDModel_multiple_bounds
from LAO_paper_model import LAOModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def linspace(start, stop, n):
    if n == 1:
        yield stop
        return
    h = (stop - start) / (n - 1)
    for i in range(n):
        yield start + h * i


def test_LAOStar():

    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel_multiple_bounds(size, init_state, goal, prob_right_transition=0.85)



    alpha = [0.2, 0.1]
    bounds = [1.5, -0.7]


    algo = LAOStar(model,constrained=True,bounds=bounds,alpha=alpha,Lagrangian=True)

    policy = algo.solve()

    value = algo.graph.root.value
    weighted_value = value[0] + alpha[0]*(value[1] - bounds[0]) + alpha[1]*(value[2] - bounds[1])

    print(value[0])
    print(value[1])
    print(weighted_value)

def draw_lower_envelop():

    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    # model = LAOModel()

    # algo = LAOStar(model)

    alpha_list = list(linspace(0,0.6,100))

    # alpha_list = [200]
    weighted_value_list = []

    bound = 1.5

    for a in alpha_list:

        print(a)

        algo = LAOStar(model,constrained=True,bounds=[bound],alpha=[a],Lagrangian=True)

        policy = algo.solve()

        value = algo.graph.root.value
        weighted_value = value[0] + a*(value[1] - bound)
        weighted_value_list.append(weighted_value)

        model.print_policy(policy)


    # print(algo.compute_value(algo.graph.nodes[(4,2)],'D'))
    # print(algo.compute_value(algo.graph.nodes[(4,2)],'U'))
    # print(algo.compute_value(algo.graph.nodes[(4,2)],'R'))
    # print(algo.compute_value(algo.graph.nodes[(4,2)],'L'))


        
    # print(alpha_list)
    # print(weighted_value_list)

    print(alpha_list)
    print(weighted_value_list)
        
    plt.plot(alpha_list, weighted_value_list,'*')
    # plt.plot(0.05775379446627887, 9.357673380261254, 'r*') # with bound = 2
    # plt.plot(0.013834705882, 9.3420861953, 'r*')  # with bound = 3
    plt.plot(0.15374170009084218, 9.42260840432311, 'r*')  # with bound = 1.5


    plt.show()




def draw_lower_envelop_multiple_bounds():

    ## this is for two separate constraints

    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel_multiple_bounds(size, init_state, goal, prob_right_transition=0.85)

    # model = LAOModel()

    # algo = LAOStar(model)

    alpha_1_range = list(linspace(0.0,100,20))
    alpha_2_range = list(linspace(0.0,20,20))
    # alpha_1_range = [46.355221667242965]
    # alpha_2_range = [4.325548069037225]

    # alpha_2_range = [20]

    alpha_1_list = []
    alpha_2_list = []
    weighted_value_list = []

    # bounds = [0.5, -0.2]
    bounds = [1.5, 10]

    for a_1 in alpha_1_range:
        for a_2 in alpha_2_range:
            print([a_1, a_2])

            alpha_1_list.append(a_1)
            alpha_2_list.append(a_2)

            algo = LAOStar(model,constrained=True,bounds=bounds,alpha=[a_1, a_2],Lagrangian=True)

            policy = algo.solve()

            while policy == False:
                algo = LAOStar(model,constrained=True,bounds=bounds,alpha=[a_1, a_2],Lagrangian=True)
                policy = algo.solve()


            value = algo.graph.root.value
            weighted_value = value[0] + a_1*(value[1] - bounds[0]) + a_2*(value[2] - bounds[1])
            weighted_value_list.append(weighted_value)


    print(alpha_1_list)
    print(alpha_2_list)
    print(weighted_value_list)
            
    fig = plt.figure()
    ax = Axes3D(fig)
    
    plt.plot(alpha_1_list,alpha_2_list, weighted_value_list,'*')
    # plt.plot(0.15374170009084218, 9.42260840432311, 'r*')  # with bound = 1.5

    plt.show()
    





def draw_lower_envelop_multiple_bounds_lb_ub():

    ## this is for two separate constraints

    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel_multiple_bounds(size, init_state, goal, prob_right_transition=0.85)

    # model = LAOModel()

    # algo = LAOStar(model)

    alpha_1_range = list(linspace(0.01,1.0,15))
    alpha_2_range = list(linspace(0.01,0.3,15))
    # alpha_2_range = [20]

    alpha_1_list = []
    alpha_2_list = []
    weighted_value_list = []

    # bounds = [0.5, -0.2]
    bounds = [1.5, -0.9]

    for a_1 in alpha_1_range:
        for a_2 in alpha_2_range:
            print([a_1, a_2])

            alpha_1_list.append(a_1)
            alpha_2_list.append(a_2)

            algo = LAOStar(model,constrained=True,bounds=bounds,alpha=[a_1, a_2],Lagrangian=True)

            policy = algo.solve()

            # while policy == False:
            #     algo = LAOStar(model,constrained=True,bounds=bounds,alpha=[a_1, a_2],Lagrangian=True)
            #     policy = algo.solve()

            if policy == None:
                weighted_value = -200
                weighted_value_list.append(weighted_value)
                print("seems unbounded below")

            else:
                
                value = algo.graph.root.value
                weighted_value = value[0] + a_1*(value[1] - bounds[0]) + a_2*(value[2] - bounds[1])
                weighted_value_list.append(weighted_value)



    # min_val = 100000000
    # max_val = -10000000
    # for v in weighted_value_list:
    #     if v==-200:
    #         continue
    #     if v < min_val:
    #         min_val = v
    #     if v > max_val:
    #         max_val = v

    # diff = max_val - min_val

    # for i in range(len(weighted_value_list)):
    #     if weighted_value_list[i]<0:
    #         weighted_value_list[i] = 8.2
            

    print(alpha_1_list)
    print(alpha_2_list)
    print(weighted_value_list)
            
    fig = plt.figure()
    ax = Axes3D(fig)
    
    plt.plot(alpha_1_list,alpha_2_list, weighted_value_list,'*')
    # plt.plot(0.15374170009084218, 9.42260840432311, 'r*')  # with bound = 1.5
    ax.axes.set_zlim3d(bottom=9.0, top=9.4) 

    plt.show()




    


def test_dual_alg():
    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    bound = 1.5

    cssp_solver = CSSPSolver(model, bounds=[bound])

    cssp_solver.solve([[0,0.6]])

    policy = cssp_solver.algo.extract_policy()

    model.print_policy(policy)



def test_dual_alg_multiple_bounds():
    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel_multiple_bounds(size, init_state, goal, prob_right_transition=0.85)
    

    bounds = [1.5, 10]

    cssp_solver = CSSPSolver(model, bounds=bounds)

    cssp_solver.solve([[2,100],[0,20]])

    policy = cssp_solver.algo.extract_policy()

    model.print_policy(policy)

    
    


# draw_lower_envelop()
# test_dual_alg()
test_dual_alg_multiple_bounds()
# test_LAOStar()
# draw_lower_envelop_multiple_bounds()
# draw_lower_envelop_multiple_bounds_lb_ub()




