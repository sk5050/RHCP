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
from grid_model import GRIDModel
from grid_model_multiple_bounds import GRIDModel_multiple_bounds
from racetrack_model import RaceTrackModel
from LAO_paper_model import LAOModel
from manual_model import MANUALModel

# from grid import Grid
# # import functools

# from matplotlib.collections import LineCollection, PolyCollection
# from matplotlib.patches import Ellipse

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
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



def compute_racetrack_heuristic():

    map_file = "models/racetrack_hard.txt"
    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state = init_state,slip_prob=0.1)

    algo = VI(model,constrained=False,VI_epsilon=1e-5)

    algo.expand_all()
    heuristic = model.heuristic_computation(algo.graph)

    print(len(heuristic))

    with open('models/racetrack_hard_heuristic.json', 'w') as outfile:
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

    cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-1,convergence_epsilon=1e-10)

    cssp_solver.solve([[0,0.6]])

    policy = cssp_solver.algo.extract_policy()

    # try:
    cssp_solver.incremental_update(20)
    # except:
    #     print(cssp_solver.candidate_set)
    
    k_best_solution_set = cssp_solver.k_best_solution_set
    for solution in k_best_solution_set:
        print("-"*20)
        print(solution[0])
        print(solution[1])
        model.print_policy(solution[2])

    


def test_dual_alg_racetrack():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_hard.txt"
    traj_check_dict_file = "models/racetrack_hard_traj_check_dict.json"
    heuristic_file = "models/racetrack_hard_heuristic.json"

    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)


    bound = 5

    cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-1,convergence_epsilon=1e-10)

    cssp_solver.solve([[0,1.0]])

    policy = cssp_solver.algo.extract_policy()

    
    cssp_solver.incremental_update(3)

    k_best_solution_set = cssp_solver.k_best_solution_set
    for solution in k_best_solution_set:
        print("-"*20)
        print(solution[0])
        print(solution[1])

    


def test_dual_alg_multiple_bounds():
    init_state = (0,0)
    size = (5,5)
    goal = (4,4)
    model = GRIDModel_multiple_bounds(size, init_state, goal, prob_right_transition=0.85)
    

    bounds = [1.5, 12]

    cssp_solver = CSSPSolver(model, bounds=bounds)

    cssp_solver.solve([[0,100],[0,20]])

    policy = cssp_solver.algo.extract_policy()

    model.print_policy(policy)

    


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

    model = SIMPLEGRIDModel(prob_right_transition=0.85)

    alpha = [8.0]
    bounds = [3.0]


    
    algo = VI(model, constrained=True, VI_epsilon=1e-50, VI_max_iter=100000,bounds=bounds, alpha=alpha)


    policy_set = [{(0, 0): 'V', (1, 0): 'V', (1, 1): 'Terminal', (0, 1): 'V'},
                  {(0, 0): 'V', (1, 0): 'V', (1, 1): 'Terminal', (0, 1): 'L'},
                  {(0, 0): 'V', (1, 0): 'L', (1, 1): 'Terminal', (0, 1): 'V'},
                  {(0, 0): 'L', (1, 0): 'V', (1, 1): 'Terminal', (0, 1): 'V'},
                  {(0, 0): 'L', (1, 0): 'L', (1, 1): 'Terminal', (0, 1): 'V'},
                  {(0, 0): 'L', (1, 0): 'V', (1, 1): 'Terminal', (0, 1): 'L'},
                  {(0, 0): 'V', (1, 0): 'L', (1, 1): 'Terminal', (0, 1): 'L'},
                  {(0, 0): 'L', (1, 0): 'L', (1, 1): 'Terminal', (0, 1): 'L'}]

    
    algo.expand_all()

    x_list = list(linspace(0.0, 3.0, 1000))

    for policy in policy_set:
        algo.policy_evaluation(policy, epsilon=1e-100)
        value_1 = algo.graph.root.value_1
        value_2 = algo.graph.root.value_2
        print("------------")
        print(value_1)
        print(value_2)
        print(value_1 + 1.7047619049945828*(value_2 - bounds[0]))

        y_list = []

        for x in x_list:
            y = value_1 + x*(value_2 - bounds[0])
            y_list.append(y)

        plt.plot(x_list, y_list)

    # plt.ylim([0,25])




    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-100,convergence_epsilon=1e-300)

    cssp_solver.solve([[0,3.0]])

    policy = cssp_solver.algo.extract_policy()


    
    cssp_solver.incremental_update(8)

    alpha = cssp_solver.algo.alpha[0]

    k_best_solution_set = cssp_solver.k_best_solution_set

    k = 0
    for solution in k_best_solution_set:
        print("-"*20)
        print(solution[0])
        print(solution[1])
        print(solution[2])
        k += 1

        value_1 = solution[1][0]
        value_2 = solution[1][1]

        y = value_1 + alpha*(value_2 - bounds[0])

        plt.plot(alpha, y, 'r*')
        plt.annotate(str(k), (alpha, y))





    
    plt.show()



# draw_lower_envelop()
test_dual_alg()
# test_dual_alg_multiple_bounds()
# test_LAOStar()
# compute_racetrack_heuristic()

# test_LAOStar_racetrack()
# draw_lower_envelop_racetrack()
# test_dual_alg_racetrack()

# test_manual_model()
# test_grid_model_head()

# cProfile.run('test_LAOStar_racetrack()')

# test_VI_racetrack()
# test_racetrack()
# draw_lower_envelop_multiple_bounds()
# draw_lower_envelop_multiple_bounds_lb_ub()

# draw_all_policies()
