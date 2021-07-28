#!/usr/bin/env python

import sys
from utils import import_models
import_models()

from graph import Node, Graph
from LAOStar import LAOStar
from ILAOStar import ILAOStar
from value_iteration import VI
from CSSPSolver import CSSPSolver
from simple_grid_model import SIMPLEGRIDModel
from simple_grid_model_2 import SIMPLEGRIDModel2
from grid_model import GRIDModel
from grid_model_multiple_bounds import GRIDModel_multiple_bounds
from racetrack_model import RaceTrackModel
from LAO_paper_model import LAOModel
from manual_model import MANUALModel
from manual_model_2 import MANUALModel2
from manual_model_3 import MANUALModel3

# from grid import Grid
# # import functools

# from matplotlib.collections import LineCollection, PolyCollection
# from matplotlib.patches import Ellipse

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import time
import random
import cProfile

import json

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
    model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    alpha = [0.0]
    bounds = [1.5]


    
    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-1, convergence_epsilon=1e-5,\
                   bounds=bounds,alpha=alpha,Lagrangian=True)

    

    t = time.time()
    policy = algo.solve()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(algo.graph.nodes)))

    

    cssp_solver = CSSPSolver(model,bounds=[0.5])

    model.print_policy(policy)

    for state, node in algo.graph.nodes.items():
        print(len(cssp_solver.get_ancestors(node)))
    
    value_1 = algo.graph.root.value_1
    # weighted_value = value[0] + alpha[0]*(value[1] - bounds[0]) + alpha[1]*(value[2] - bounds[1])
    # value_2 = algo.graph.root.value_2
    print(value_1)
    # print(weighted_value)




def test_LAOStar_racetrack():
    sys.setrecursionlimit(5000)

    map_file = "models/racetrack_hard.txt"
    traj_check_dict_file = "models/racetrack_hard_traj_check_dict.json"
    heuristic_file = "models/racetrack_hard_heuristic.json"

    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)

    # init_state = (0,0)
    # size = (5,5)
    # goal = (4,4)
    # model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    alpha = [0.0]
    bounds = [10]


    
    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-5, convergence_epsilon=1e-5,\
                   bounds=bounds,alpha=alpha,Lagrangian=True)
    

    t = time.time()
    policy = algo.solve()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(algo.graph.nodes)))
    
    primary_value = algo.graph.root.value_1
    # weighted_value = value[0] + alpha[0]*(value[1] - bounds[0]) + alpha[1]*(value[2] - bounds[1])
    
    print(primary_value)
    # print(weighted_value)



    # map_text = open(map_file, 'r')
    # lines = map_text.readlines()

    # test_grid = Grid(len(lines[0]),len(lines))
    # axes = test_grid.draw()

    # offtrack = []
    # for i in range(len(lines)):
    #     for j in range(len(lines[0])):
    #         offtrack.append((j,i))

    # for pos in model.ontrack_pos_set:
    #     offtrack.remove(pos)

    # for pos in model.finishline_pos_set:
    #     offtrack.remove(pos)


    # initial = [test_grid.cell_verts(ix, iy) for ix,iy in model.initial_pos_set]
    # collection_initial = PolyCollection(initial, facecolors='g')
    # axes.add_collection(collection_initial)

    # finish = [test_grid.cell_verts(ix, iy) for ix,iy in model.finishline_pos_set]
    # collection_finish = PolyCollection(finish, facecolors='b')
    # axes.add_collection(collection_finish)

    # off = [test_grid.cell_verts(ix, iy) for ix,iy in offtrack]
    # collection_off = PolyCollection(off, facecolors='r')
    # axes.add_collection(collection_off)    


    # # for init_pos in racetrack_model.initial_pos_set:

    # #     if init_pos+(0,0) in 


    # state = init_state

    # while state[0:2]!=(-1,-1):

    #     action = policy[state]
    #     new_states = model.state_transitions(state,action)

    #     choices = random.choices([0,1],weights=[0.9,0.1])
    #     choice = choices[0]

    #     path = [state[0:2], new_states[choice][0][0:2]]
    #     test_grid.draw_path(axes,path,color='y')

    #     state = new_states[choice][0]
    

    # # for state,action in policy.items():
    # #     if action=='Terminal':
    # #         continue
        
    # #     new_states = model.state_transitions(state,action)

    # #     if new_states[0][0][0:2]!=(-1,-1):
    # #         path1 = [state[0:2], new_states[0][0][0:2]]
    # #         test_grid.draw_path(axes, path1, color='y')

    # #     if new_states[1][0][0:2]!=(-1,-1):
    # #         path2 = [state[0:2], new_states[1][0][0:2]]
    # #         test_grid.draw_path(axes, path2, color='y')

    # plt.show()





def test_LAOStar_racetrack1():
    sys.setrecursionlimit(5000)

    map_file = "models/racetrack1.txt"
    traj_check_dict_file = "models/racetrack1_traj_check_dict.json"
    heuristic_file = "models/racetrack1_heuristic.json"

    init_state = (1,5,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)

    # init_state = (0,0)
    # size = (5,5)
    # goal = (4,4)
    # model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    alpha = [0.0]
    bounds = [1.5]


    
    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-5, convergence_epsilon=1e-5,\
                   bounds=bounds,alpha=alpha,Lagrangian=True)
    

    t = time.time()
    policy = algo.solve()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(algo.graph.nodes)))
    
    primary_value = algo.graph.root.value_1
    # weighted_value = value[0] + alpha[0]*(value[1] - bounds[0]) + alpha[1]*(value[2] - bounds[1])
    
    print(primary_value)    




    
def test_VI_racetrack():

    map_file = "models/racetrack_hard.txt"
    traj_check_dict_file = "models/racetrack_hard_traj_check_dict.json"
    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)

    # init_state = (0,0)
    # size = (5,5)
    # goal = (4,4)
    # model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    alpha = [0.0]
    bounds = [1.5]


    
    algo = VI(model,constrained=True,VI_epsilon=1e-100,bounds=bounds, alpha=alpha)

    t = time.time()
    policy = algo.solve()
    print("elapsed time: "+str(time.time() - t))
    print("number of states explored: "+str(len(algo.graph.nodes)))
    
    primary_value = algo.graph.root.value_1
    # weighted_value = value[0] + alpha[0]*(value[1] - bounds[0]) + alpha[1]*(value[2] - bounds[1])
    
    print(primary_value)


    # map_text = open(map_file, 'r')
    # lines = map_text.readlines()

    # test_grid = Grid(len(lines[0]),len(lines))
    # axes = test_grid.draw()

    # offtrack = []
    # for i in range(len(lines)):
    #     for j in range(len(lines[0])):
    #         offtrack.append((j,i))

    # for pos in model.ontrack_pos_set:
    #     offtrack.remove(pos)

    # for pos in model.finishline_pos_set:
    #     offtrack.remove(pos)


    # initial = [test_grid.cell_verts(ix, iy) for ix,iy in model.initial_pos_set]
    # collection_initial = PolyCollection(initial, facecolors='g')
    # axes.add_collection(collection_initial)

    # finish = [test_grid.cell_verts(ix, iy) for ix,iy in model.finishline_pos_set]
    # collection_finish = PolyCollection(finish, facecolors='b')
    # axes.add_collection(collection_finish)

    # off = [test_grid.cell_verts(ix, iy) for ix,iy in offtrack]
    # collection_off = PolyCollection(off, facecolors='r')
    # axes.add_collection(collection_off)    


    # # for init_pos in racetrack_model.initial_pos_set:

    # #     if init_pos+(0,0) in 


    # # state = init_state

    # # while state[0:2]!=(-1,-1):

    # #     action = policy[state]
    # #     new_states = model.state_transitions(state,action)

    # #     choices = random.choices([0,1],weights=[0.9,0.1])
    # #     choice = choices[0]

    # #     path = [state[0:2], new_states[choice][0][0:2]]
    # #     test_grid.draw_path(axes,path,color='y')

    # #     state = new_states[choice][0]
    

    # for state,action in policy.items():
    #     if action=='Terminal':
    #         continue
        
    #     new_states = model.state_transitions(state,action)

    #     if new_states[0][0][0:2]!=(-1,-1):
    #         path1 = [state[0:2], new_states[0][0][0:2]]
    #         test_grid.draw_path(axes, path1, color='y')

    #     if new_states[1][0][0:2]!=(-1,-1):
    #         path2 = [state[0:2], new_states[1][0][0:2]]
    #         test_grid.draw_path(axes, path2, color='y')

    # plt.show()
    


def test_VI_racetrack1():

    map_file = "models/racetrack1.txt"
    traj_check_dict_file = "models/racetrack1_traj_check_dict.json"
    init_state = (1,5,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)

    # init_state = (0,0)
    # size = (5,5)
    # goal = (4,4)
    # model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    alpha = [0.0]
    bounds = [1.5]


    
    algo = VI(model,constrained=True,VI_epsilon=1e-100,bounds=bounds, alpha=alpha)

    t = time.time()
    policy = algo.solve()
    print("elapsed time: "+str(time.time() - t))
    print("number of states explored: "+str(len(algo.graph.nodes)))
    
    primary_value = algo.graph.root.value_1
    # weighted_value = value[0] + alpha[0]*(value[1] - bounds[0]) + alpha[1]*(value[2] - bounds[1])
    
    print(primary_value)

    cssp_solver = CSSPSolver(model,bounds=[0.5])

    cssp_solver.algo = algo

    print(len(policy))

    for state,action in policy.items():
        print(len(cssp_solver.get_head(algo.graph.nodes[state])))
    


    


def test_racetrack():

    map_file = "models/racetrack1.txt"
    init_state = (1,6,0,0)
    model = RaceTrackModel(map_file, init_state = init_state,slip_prob=0.1)

    # init_state = (0,0)
    # size = (5,5)
    # goal = (4,4)
    # model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    alpha = [0.0]
    bounds = [1.5]


    



    # map_text = open(map_file, 'r')
    # lines = map_text.readlines()

    # test_grid = Grid(len(lines[0]),len(lines))
    # axes = test_grid.draw()

    # offtrack = []
    # for i in range(len(lines)):
    #     for j in range(len(lines[0])):
    #         offtrack.append((j,i))

    # for pos in model.ontrack_pos_set:
    #     offtrack.remove(pos)

    # for pos in model.finishline_pos_set:
    #     offtrack.remove(pos)


    # initial = [test_grid.cell_verts(ix, iy) for ix,iy in model.initial_pos_set]
    # collection_initial = PolyCollection(initial, facecolors='g')
    # axes.add_collection(collection_initial)

    # finish = [test_grid.cell_verts(ix, iy) for ix,iy in model.finishline_pos_set]
    # collection_finish = PolyCollection(finish, facecolors='b')
    # axes.add_collection(collection_finish)

    # off = [test_grid.cell_verts(ix, iy) for ix,iy in offtrack]
    # collection_off = PolyCollection(off, facecolors='r')
    # axes.add_collection(collection_off)


    path = [(1,6), (1,5)]

    print(model.bresenham_check_crash(1,6,1,5))
    
    # path = [(11,7), (15,8)]
    # test_grid.draw_path(axes, path, color='y')



    # plt.show()


def compute_racetrack_traj_check():

    ## temporarily add "traj_check_dict" in racetrack model property, and add bresenham's result.

    map_file = "models/racetrack_simple.txt"
    init_state = (1,2,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, slip_prob=0.1)

    alpha = [0.0]
    bounds = [1.5]
    
    algo = VI(model,constrained=True,VI_epsilon=1e-100,bounds=bounds, alpha=alpha)

    algo.expand_all()

    with open('models/racetrack_simple_traj_check_dict.json', 'w') as outfile:
        json.dump(model.traj_check_dict, outfile)
    



def compute_racetrack_heuristic():

    map_file = "models/racetrack_simple.txt"
    init_state = (1,2,0,0)
    model = RaceTrackModel(map_file, init_state = init_state,slip_prob=0.1)

    algo = VI(model,constrained=False,VI_epsilon=1e-5)

    algo.expand_all()
    heuristic = model.heuristic_computation(algo.graph)

    print(len(heuristic))

    with open('models/racetrack_simple_heuristic.json', 'w') as outfile:
        json.dump(heuristic, outfile)

    


def test_LAOStar_multiple_bounds():

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




def draw_lower_envelop_racetrack():



    sys.setrecursionlimit(5000)

    map_file = "models/racetrack_hard.txt"
    traj_check_dict_file = "models/racetrack_hard_traj_check_dict.json"
    heuristic_file = "models/racetrack_hard_heuristic.json"

    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)

    
    alpha_list = list(linspace(0,0.25,30))

    weighted_value_list = []

    bound = 5

    for a in alpha_list:

        algo = ILAOStar(model,constrained=True,VI_epsilon=1e-100, convergence_epsilon=1e-100,\
                   bounds=[bound],alpha=[a],Lagrangian=True)

        policy = algo.solve()

        value_1 = algo.graph.root.value_1
        value_2 = algo.graph.root.value_2
        
        weighted_value = value_1 + a*(value_2 - bound)
        weighted_value_list.append(weighted_value)


        print("------------------")
        print(a)
        print(value_1)
        print(value_2)
        
    print(alpha_list)
    print(weighted_value_list)
        
    plt.plot(alpha_list, weighted_value_list,'*')
    plt.plot(0.10517319210982577, 23.447206345550008, 'r*')  # with bound = 1.5


    plt.show()    





def test_subgradient():

    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel_multiple_bounds(size, init_state, goal, prob_right_transition=0.85)

    bounds = [1.5, 10]
    
    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-1,convergence_epsilon=1e-100)


    cssp_solver.solve_sg([0,0])

    policy = cssp_solver.algo.extract_policy()

    alpha = cssp_solver.algo.alpha
    value_1 = cssp_solver.algo.graph.root.value_1
    value_2 = cssp_solver.algo.graph.root.value_2
    value_3 = cssp_solver.algo.graph.root.value_3
    weighted_value = value_1 + alpha[0]*(value_2 - bounds[0]) + alpha[1]*(value_3 - bounds[1])
    
    print(alpha)
    print(value_1)
    print(value_2)
    print(value_3)
    print(weighted_value)




def draw_lower_envelop_multiple_bounds():

    ## this is for two separate constraints

    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel_multiple_bounds(size, init_state, goal, prob_right_transition=0.85)

    # model = LAOModel()

    # algo = LAOStar(model)

    alpha_1_range = list(linspace(0,110,50))
    alpha_2_range = list(linspace(0,22,50))
    # alpha_1_range = [46.355221667242965]
    # alpha_2_range = [4.325548069037225]

    # alpha_2_range = [20]

    alpha_1_list = []
    alpha_2_list = []
    weighted_value_list = []

    bounds = [1.5, 10]

    for a_1 in alpha_1_range:
        for a_2 in alpha_2_range:
            print([a_1, a_2])

            alpha_1_list.append(a_1)
            alpha_2_list.append(a_2)

            algo = ILAOStar(model,constrained=True,VI_epsilon=1e-1, convergence_epsilon=1e-10,\
                   bounds=bounds,alpha=[a_1, a_2],Lagrangian=True)

            policy = algo.solve()


            value_1 = algo.graph.root.value_1
            value_2 = algo.graph.root.value_2
            value_3 = algo.graph.root.value_3
            weighted_value = value_1 + a_1*(value_2 - bounds[0]) + a_2*(value_3 - bounds[1])
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
    size = (10,10)
    goal = (4,4)
    model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    bound = 25

    cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-1,convergence_epsilon=1e-10)

    t = time.time()

    cssp_solver.solve([[0,0.6]])

    policy = cssp_solver.algo.extract_policy()

    cssp_solver.candidate_pruning = False

    # try:
    cssp_solver.incremental_update(300)
    # except:
    #     print(cssp_solver.candidate_set)
    
    k_best_solution_set = cssp_solver.k_best_solution_set
    k = 0
    for solution in k_best_solution_set:
        print("-"*20)
        k += 1
        print("num: " + str(k))
        print(solution[0])
        print(solution[1])
        # model.print_policy(solution[2])

    print(len(cssp_solver.candidate_set))

    print(time.time() - t)

    print(cssp_solver.anytime_solutions)




def test_copy_graph():



    sys.setrecursionlimit(8000)

    map_file = "models/racetrack1.txt"
    traj_check_dict_file = "models/racetrack1_traj_check_dict.json"
    heuristic_file = "models/racetrack1_heuristic.json"

    init_state = (1,5,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)


    bound = 5

    cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-1,convergence_epsilon=1e-10)

    cssp_solver.solve([[0,1.0]])

    policy = cssp_solver.algo.extract_policy()    


    t = time.time()
    a = []
    for i in range(500):
        a = cssp_solver.copy_graph(cssp_solver.algo.graph)

    print(time.time() - t)

    
    t = time.time()
    b = []
    for i in range(500):
        b = cssp_solver.copy_best_graph(cssp_solver.algo.graph)

    print(time.time() - t)

    print(len(b))
        

    t = time.time()
    for i in range(500):
        cssp_solver.return_to_best_graph(cssp_solver.algo.graph, b)


    print(time.time() - t)
    


    
    


def test_dual_alg_racetrack():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_hard (copy).txt"
    traj_check_dict_file = "models/racetrack_hard_traj_check_dict.json"
    heuristic_file = "models/racetrack_hard_heuristic.json"

    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)


    bound = 1

    cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-1,convergence_epsilon=1e-10)

    t = time.time()

    cssp_solver.solve([[0,1.0]])

    policy = cssp_solver.algo.extract_policy()

    cssp_solver.candidate_pruning = True

    try:
        cssp_solver.incremental_update(2)
    except:

        k_best_solution_set = cssp_solver.k_best_solution_set
        for solution in k_best_solution_set:
            print("-"*20)
            print(solution[0])
            print(solution[1])

        print(time.time() - t)

        print(cssp_solver.anytime_solutions)


    k_best_solution_set = cssp_solver.k_best_solution_set
    for solution in k_best_solution_set:
        print("-"*20)
        print(solution[0])
        print(solution[1])

    print(time.time() - t)

    print(cssp_solver.anytime_solutions)


def test_dual_alg_racetrack1():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack1.txt"
    traj_check_dict_file = "models/racetrack1_traj_check_dict.json"
    heuristic_file = "models/racetrack1_heuristic.json"

    init_state = (1,5,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)


    bound = 5

    cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-1,convergence_epsilon=1e-10)

    t = time.time()

    cssp_solver.solve([[0,1.0]])

    policy = cssp_solver.algo.extract_policy()


    cssp_solver.candidate_pruning = True

    cssp_solver.incremental_update(5)

    k_best_solution_set = cssp_solver.k_best_solution_set
    for solution in k_best_solution_set:
        print("-"*20)
        print(solution[0])
        print(solution[1])   
        
    print(time.time() - t)
    print(len(cssp_solver.candidate_set))

    print(cssp_solver.anytime_solutions)


def test_dual_alg_multiple_bounds():

    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel_multiple_bounds(size, init_state, goal, prob_right_transition=0.85)

    bounds = [1.5, 10]

    
    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-1,convergence_epsilon=1e-10)


    # cssp_solver.solve_dual_multiple([[0,100],[0,20]])
    # cssp_solver.solve_dual_line_search([[0,100],[0,20]])
    cssp_solver.solve_sg([100,20], h=0.5, rule='sqrt')

    policy = cssp_solver.algo.extract_policy()

    alpha = cssp_solver.algo.alpha
    value_1 = cssp_solver.algo.graph.root.value_1
    value_2 = cssp_solver.algo.graph.root.value_2
    value_3 = cssp_solver.algo.graph.root.value_3
    weighted_value = value_1 + alpha[0]*(value_2 - bounds[0]) + alpha[1]*(value_3 - bounds[1])
    
    print(alpha)
    print(value_1)
    print(value_2)
    print(value_3)
    print(weighted_value)

    


def test_manual_model():

    model = MANUALModel(cost=1.0)
    cssp_solver = CSSPSolver(model,bounds=[1])

    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-5, convergence_epsilon=1e-5,\
                   bounds=[1.0],alpha=[1.0],Lagrangian=True)
    
    policy_1 = algo.solve()

    head = cssp_solver.get_head(algo.graph.root,algo.graph.nodes["2"])
    print(policy_1)
    print("head: "+str([n.state for n in head]))
    

    # model.cost_param_2 = 10

    # algo = ILAOStar(model,constrained=True,VI_epsilon=1e-5, convergence_epsilon=1e-5,\
    #                bounds=[1.0],alpha=[1.0],Lagrangian=True)

    # policy_2 = algo.solve()

    # head = cssp_solver.get_head(algo.graph.root, algo.graph.nodes["3"])
    # print("head: "+str([n.state for n in head]))

    # is_head_eq = cssp_solver.is_head_eq(head, policy_1)

    # print(is_head_eq)
    
    
    # print(policy)
    # print(algo.graph.nodes)
    # for state,node in algo.graph.nodes.items():
    #     print("--------------------")
    #     print("state: "+state)

    #     children_state = []
    #     for action, children in node.children.items():
    #         for child in children:
    #             children_state.append(child[0].state)

    #     print("children: "+str(children_state))
    #     print("parents: "+str([n.state for n in node.best_parents_set]))


def test_manual_model_2():

    model = MANUALModel2(cost=1.0)
    cssp_solver = CSSPSolver(model,bounds=[1])

    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-5, convergence_epsilon=1e-5,\
                   bounds=[1.0],alpha=[1.0],Lagrangian=True)
    
    policy_1 = algo.solve()

    # head = cssp_solver.get_head(algo.graph.root,algo.graph.nodes["2"])
    print(policy_1)
    # print("head: "+str([n.state for n in head]))

    print(algo.graph.root.value_1)


    print(algo.graph.root.value_1)
    print(algo.graph.nodes["1"].value_1)
    print(algo.graph.nodes["2"].value_1)

    Q = np.array([[0,0.5,0.5],[0.1,0,0.1],[0,0,0]])
    N = np.linalg.inv(np.eye(3) - Q)
    print(N)


    

    # prob_matrix = []

    # for state,action in policy_1.items():
    #     if action=="Terminal":
    #         continue

    #     source_node = algo.graph.nodes[state]
    #     children = source_node.children[action]

    #     prob_vector = []

    #     for state_, action_ in policy_1.items():
    #         if action_=="Terminal":
    #             continue

    #         is_state_in = False
    #         for child,child_prob in children:
    #             if child.state == state_:
    #                 prob_vector.append(child_prob)
    #                 is_state_in = True
    #                 break

    #         if is_state_in == False:
    #             prob_vector.append(0)

    #     prob_matrix.append(prob_vector)

    # prob_matrix = np.matrix(prob_matrix)

    # N = np.linalg.inv(np.eye(len(policy_1)-1) - prob_matrix)
    
    # print(N)




def test_manual_model_3():

    model = MANUALModel3(cost=1.0)
    cssp_solver = CSSPSolver(model,bounds=[1])

    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-5, convergence_epsilon=1e-5,\
                   bounds=[1.0],alpha=[1.0],Lagrangian=True)
    
    policy_1 = algo.solve()

    # head = cssp_solver.get_head(algo.graph.root,algo.graph.nodes["2"])
    print(policy_1)
    # print("head: "+str([n.state for n in head]))

    print(algo.graph.root.value_1)
    print(algo.graph.nodes["1"].value_1)
    print(algo.graph.nodes["2"].value_1)

    Q = np.array([[0,0.5,0.5],[0,0,0],[0,0,0]])
    N = np.linalg.inv(np.eye(3) - Q)
    print(N)

    # prob_matrix = []

    # for state,action in policy_1.items():
    #     if action=="Terminal":
    #         continue

    #     source_node = algo.graph.nodes[state]
    #     children = source_node.children[action]

    #     prob_vector = []

    #     for state_, action_ in policy_1.items():
    #         if action_=="Terminal":
    #             continue

    #         is_state_in = False
    #         for child,child_prob in children:
    #             if child.state == state_:
    #                 prob_vector.append(child_prob)
    #                 is_state_in = True
    #                 break

    #         if is_state_in == False:
    #             prob_vector.append(0)

    #     prob_matrix.append(prob_vector)

    # prob_matrix = np.matrix(prob_matrix)

    # N = np.linalg.inv(np.eye(len(policy_1)-1) - prob_matrix)
    
    # print(N)
    


def test_grid_model_head():

    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)


    cssp_solver = CSSPSolver(model,bounds=[0.5])

    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-5, convergence_epsilon=1e-5,\
                   bounds=[1.0],alpha=[1.0],Lagrangian=True)
    
    policy = algo.solve()

    model.print_policy(policy)

    head = cssp_solver.get_head(algo.graph.root,algo.graph.nodes[(1,4)])
    print(policy)
    print(len(head))

    print("head: "+str([n.state for n in head]))
    



def draw_all_policies():

    model = SIMPLEGRIDModel2(prob_right_transition=0.85)

    alpha = [8.0]
    bounds = [1.0]


    
    algo = VI(model, constrained=True, VI_epsilon=1e-50, VI_max_iter=100000,bounds=bounds, alpha=alpha)


    # policy_set = [{(0, 0): 'V', (1, 0): 'V', (1, 1): 'Terminal', (0, 1): 'V'},
    #               {(0, 0): 'V', (1, 0): 'V', (1, 1): 'Terminal', (0, 1): 'L'},
    #               {(0, 0): 'V', (1, 0): 'L', (1, 1): 'Terminal', (0, 1): 'V'},
    #               {(0, 0): 'L', (1, 0): 'V', (1, 1): 'Terminal', (0, 1): 'V'},
    #               {(0, 0): 'L', (1, 0): 'L', (1, 1): 'Terminal', (0, 1): 'V'},
    #               {(0, 0): 'L', (1, 0): 'V', (1, 1): 'Terminal', (0, 1): 'L'},
    #               {(0, 0): 'V', (1, 0): 'L', (1, 1): 'Terminal', (0, 1): 'L'},
    #               {(0, 0): 'L', (1, 0): 'L', (1, 1): 'Terminal', (0, 1): 'L'}]

    
    # algo.expand_all()

    # x_list = list(linspace(0.0, 3.0, 1000))

    # for policy in policy_set:
    #     algo.policy_evaluation(policy, epsilon=1e-100)
    #     value_1 = algo.graph.root.value_1
    #     value_2 = algo.graph.root.value_2
    #     print("------------")
    #     print(value_1)
    #     print(value_2)
    #     print(value_1 + 1.7047619049945828*(value_2 - bounds[0]))

    #     y_list = []

    #     for x in x_list:
    #         y = value_1 + x*(value_2 - bounds[0])
    #         y_list.append(y)

    #     plt.plot(x_list, y_list)

    # # plt.ylim([0,25])




    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-100,convergence_epsilon=1e-300)

    cssp_solver.solve([[0,3.0]])

    policy = cssp_solver.algo.extract_policy()


    
    cssp_solver.incremental_update(32)

    alpha = cssp_solver.algo.alpha[0]

    k_best_solution_set = cssp_solver.k_best_solution_set

    k = 0
    k_list = []
    sol_list = []
    for solution in k_best_solution_set:
        print("-"*20)
        k += 1
        print("num: " + str(k))
        print(solution[0])
        print(solution[1])
        print(solution[2])

        k_list.append(k)
        sol_list.append(solution[0])

    plt.plot(k_list, sol_list, '*')
    plt.show()

    # k = 0
    # for solution in k_best_solution_set:
    #     print("-"*20)
    #     print(solution[0])
    #     print(solution[1])
    #     print(solution[2])
    #     k += 1

    #     value_1 = solution[1][0]
    #     value_2 = solution[1][1]

    #     y = value_1 + alpha*(value_2 - bounds[0])

    #     plt.plot(alpha, y, 'r*')
    #     plt.annotate(str(k), (alpha, y))





    
    # plt.show()






def test_pruning_rule():


    # init_state = (0,0)
    # size = (10,10)
    # goal = (4,4)
    # model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    # bound = 1.5

    # cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-1,convergence_epsilon=1e-200)

    # t = time.time()

    # cssp_solver.solve([[0,0.6]])


    
    # policy = cssp_solver.algo.extract_policy()


    # testing_states = [(7,7), (3,5), (1,6), (1,1),(2,1),(3,1), (3,3),(1,3)]

    # z = 7




    sys.setrecursionlimit(8000)

    map_file = "models/racetrack1.txt"
    traj_check_dict_file = "models/racetrack1_traj_check_dict.json"
    heuristic_file = "models/racetrack1_heuristic.json"

    init_state = (1,5,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)


    bound = 5

    cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-100,convergence_epsilon=1e-300)


    cssp_solver.algo.alpha = [0.0]

    policy = cssp_solver.algo.solve()

    cssp_solver.algo.policy_evaluation(policy, epsilon=1e-100)

    print(cssp_solver.algo.graph.root.value_1)


    testing_states_set = [(2, 5, 1, 0), (3, 5, 1, 0), (4, 5, 1, 0), (5, 5, 1, 0), (9, 5, 1, 0), (33, 7, 1, 3), (22, 1, 3, 0), (32, 4, 2, 2),(1,5,0,0)]


    z = 2
    
    testing_states = [testing_states_set[-1]]

    #####################################################


    state_list = []
    for state, action in policy.items():
        if action != 'Terminal':
            state_list.append(state)


    num_states = len(state_list)
        
    Q = np.empty((num_states, num_states))
    root_idx = None

    terminal_idx = []

    testing_state_indices = []
    
    i = 0
    for state in state_list:

        if policy[state]=='Terminal':
            Q[i,:] = np.zeros(num_states)
            Q[i,i] = 1
            terminal_idx.append(i)
            i += 1
            continue

        if state==model.init_state:
            root_idx = state_list.index(state)

        if state in testing_states:
            testing_state_indices.append(state_list.index(state))

        Q_vector = np.zeros(num_states)

        node = cssp_solver.algo.graph.nodes[state]
        children = node.children[node.best_action]

        # p =0
        # print(node.best_action)
        for child, child_prob in children:
            if child.terminal != True:
                # print(child.state)
                idx = state_list.index(child.state)
                if Q_vector[idx] > 0:
                    Q_vector[idx] += child_prob
                else:
                    Q_vector[idx] = child_prob

            
             # p+= child_prob

        # print(p)
        Q[i,:] = Q_vector

        i += 1

    # for i in range(num_states):
    #     print(sum(Q_vector))


    # Q[testing_state_idx,:] = np.zeros(num_states)

    I = np.identity(num_states)
    N1 = np.linalg.inv(I - Q)
    R = np.ones(num_states)

    for i in terminal_idx:
        R[i] = 0
    
    V = np.dot(N1, R)
    print(V[root_idx])

    # Q = np.delete(Q,testing_state_idx,0)
    # Q = np.delete(Q,testing_state_idx,1)

    for testing_state_idx in testing_state_indices:

        Q[testing_state_idx,:] = np.zeros(num_states)
        R[testing_state_idx] = 0

    # Q[testing_state_idx, testing_state_idx] = 1

    I = np.identity(num_states)
    N2 = np.linalg.pinv(I - Q)


    V = np.dot(N2, R)
    print(V[root_idx])
    print(N2[root_idx,root_idx])
    
    ############################################################


    nodes = []
    heads = []
    for testing_state in testing_states:
        nodes.append(cssp_solver.algo.graph.nodes[testing_state])
        heads.append(cssp_solver.get_head(nodes[-1]))
        # print(len(heads[-1]))

    # print(nodes[1] in heads[0])

    # print("prev testing node's value: " + str(node.value_1))

    
    new_policy = dict()
    for state, action in policy.items():
        state_in_head = True
        for head in heads:
            if cssp_solver.algo.graph.nodes[state] not in head:
                state_in_head = False
                break

        if state_in_head == True:
            new_policy[state] = action

    for node in nodes:
        new_policy[node.state] = 'Terminal'
        node.terminal = True
        node.value_1 = 0

    print(cssp_solver.algo.graph.root.value_1)
    

    cssp_solver.algo.policy_evaluation(new_policy, epsilon=1e-300)

    print(cssp_solver.algo.graph.root.value_1)


    # print(testing_state_idx)
    # print(N[root_idx, testing_state_idx])#/N[root_idx,root_idx])
    





def SM_update(B, u, v):
    return B - np.outer(B @ u, v @ B) / (1 + v.T @ B @ u)


def SM_update_2(B, v, idx):
    B_ = B[:,idx]
    out = np.outer(B_,v)
    return B - (out @ B) / (1 + np.trace(out))


def SM_update_3(B, v, n, idx):
    B_new = np.copy(B)
    t1 = B[:,idx]
    t2 = np.empty(n)
    for i in range(n):
        t2[i] = np.dot(v, B[:,i])

    l = t2[idx]

    for i in range(n):
        vt = t1[i] / (1 + l)
        B[i,:] = B[i,:] - vt*t2

    return B_new
        


def SMInv(Ainv, u, v, e=None): 
    u = u.reshape((len(u),1)) 
    v = v.reshape((len(v),1)) 
    if e is not None: 
        g = np.dot(Ainv, u) / (e + np.dot(v.T, np.dot(Ainv, u)))
        return (Ainv / e) - np.dot(g, np.dot(v.T, Ainv/e)) 
    else: 
        return Ainv - np.dot(Ainv, np.dot(np.dot(u,v.T), Ainv)) / ( 1 + np.dot(v.T, np.dot(Ainv, u)))


def test_pruning_rule_2():

    sys.setrecursionlimit(8000)

    map_file = "models/racetrack1.txt"
    traj_check_dict_file = "models/racetrack1_traj_check_dict.json"
    heuristic_file = "models/racetrack1_heuristic.json"

    init_state = (1,5,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)


    bound = 5

    cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-100,convergence_epsilon=1e-300)


    cssp_solver.algo.alpha = [0.0]

    policy = cssp_solver.algo.solve()

    cssp_solver.algo.policy_evaluation(policy, epsilon=1e-100)

    policy_value = cssp_solver.algo.graph.root.value_1

    

    #####################################################

    t = time.time()


    state_list = []
    for state, action in policy.items():
        if action != 'Terminal':
            state_list.append(state)


    num_states = len(state_list)
        
    Q = np.empty((num_states, num_states))
    root_idx = None

    terminal_idx = []
    
    i = 0
    state_idx_dict = dict()
    for state in state_list:
        state_idx_dict[state] = i

        if state==model.init_state:
            root_idx = state_list.index(state)

        Q_vector = np.zeros(num_states)

        node = cssp_solver.algo.graph.nodes[state]
        children = node.children[node.best_action]

        for child, child_prob in children:
            if child.terminal != True:
                idx = state_list.index(child.state)
                if Q_vector[idx] > 0:
                    Q_vector[idx] += child_prob
                else:
                    Q_vector[idx] = child_prob

        Q[i,:] = Q_vector

        i += 1




        
    I = np.identity(num_states)
    N = np.linalg.inv(I - Q)
    R = np.ones(num_states)

    epsilon = 0.001


    N_vector = N[root_idx]

    sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]

    initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<1e-8]

    for pruned_state in initial_pruned_states:
        state = pruned_state[0]
        idx = pruned_state[1]
        Q[idx,:] = np.zeros(num_states)
        R[idx] = 0

    print(len(initial_pruned_states))
    
    N_new = np.linalg.inv(I - Q)

    V = np.dot(N_new, R)
    new_value = V[root_idx]

    epsilon -= (policy_value - new_value) / policy_value

    prev_value = new_value


    if epsilon < 0:
        raise ValueError("initial pruning was too aggressive!")
    else:
        N = N_new


    candidate_generating_states = []

    accumulated_head = set(cssp_solver.algo.graph.nodes.values())

    pruned_states = []
    
    
    for state in sorted_states:
        idx = state_idx_dict[state]

        if (state,idx) in initial_pruned_states:
            continue
        
        node = cssp_solver.algo.graph.nodes[state]

        if node not in accumulated_head:
            continue

        else:
            
            v = Q[idx,:]
            u = np.zeros(num_states)
            u[idx] = 1


            # N_new = SMInv(N,u,v)
            N_new = SM_update(N,u,v)
            # N_new = SM_update_2(N,v,idx)
            # N_new = SM_update_3(N,v,num_states,idx)
            
            R[idx] = 0

            V = np.dot(N_new, R)
            new_value = V[root_idx]

            if prev_value < new_value:
                if abs(prev_value - new_value) > 1e-8:
                    print(prev_value)
                    print(new_value)
                    raise ValueError("something went wrong.")

            elif (prev_value - new_value) / policy_value < epsilon:
                ## can be pruned
                N = N_new
                epsilon -= (prev_value - new_value) / policy_value
                head = cssp_solver.get_head(node)
                accumulated_head = accumulated_head.intersection(head)

                prev_value = new_value

                pruned_states.append(state)

            else:
                ## cannot be pruned
                R[idx] = 1
                head = cssp_solver.get_head(node)
                blocked_action_set = cssp_solver.get_blocked_action_set(node,head)
                candidate_generating_states.append((state, head, blocked_action_set))
        

    print(time.time() - t)
    print(num_states)
    print(len(pruned_states))
    print(len(candidate_generating_states))

    for state, idx in initial_pruned_states:
        pruned_states.append(state)

    print(len(pruned_states))
    # print(pruned_states)







    


# draw_lower_envelop()
# test_dual_alg()
# test_dual_alg_multiple_bounds()
# test_LAOStar()

# compute_racetrack_heuristic()
# compute_racetrack_traj_check()

# test_LAOStar_racetrack()
# draw_lower_envelop_racetrack()
test_dual_alg_racetrack()
# test_dual_alg_racetrack1()

# test_manual_model()
# test_manual_model_2()
# test_manual_model_3()

# test_grid_model_head()

# cProfile.run('test_LAOStar_racetrack()')

# test_VI_racetrack()
# test_racetrack()

# draw_lower_envelop_multiple_bounds()
# draw_lower_envelop_multiple_bounds_lb_ub()

# draw_all_policies()


# test_VI_racetrack1()
# test_LAOStar_racetrack1()

# test_copy_graph()

# test_pruning_rule()
# test_pruning_rule_2()
# test_subgradient()







