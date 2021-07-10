#!/usr/bin/env python

import sys
from utils import import_models
import_models()

from graph import Node, Graph
from LAOStar import LAOStar
from ILAOStar import ILAOStar
from value_iteration import VI
from CSSPSolver import CSSPSolver
from MILPSolver import MILPSolver


from simple_grid_model import SIMPLEGRIDModel
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




def test_racetrack_hard():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_hard.txt"
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
    
    cssp_solver.incremental_update(1)

    k_best_solution_set = cssp_solver.k_best_solution_set
    for solution in k_best_solution_set:
        print("-"*20)
        print(solution[0])
        print(solution[1])

    print(time.time() - t)
    print(cssp_solver.anytime_solutions)

    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")


    solver = MILPSolver(model, bound, cssp_solver.algo)

    solver.encode_MILP()




def test_racetrack():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack1.txt"
    traj_check_dict_file = "models/racetrack1_traj_check_dict.json"
    heuristic_file = "models/racetrack1_heuristic.json"

    init_state = (1,5,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)


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
    print(cssp_solver.anytime_solutions)

    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")


    solver = MILPSolver(model, bound, cssp_solver.algo)

    solver.encode_MILP()




def test_racetrack_easy():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_easy.txt"
    traj_check_dict_file = "models/racetrack_easy_traj_check_dict.json"
    heuristic_file = "models/racetrack_easy_heuristic.json"

    init_state = (1,2,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)


    bound = 20

    # cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-1,convergence_epsilon=1e-10)

    # t = time.time()

    # cssp_solver.solve([[0,1.0]])

    # policy = cssp_solver.algo.extract_policy()


    # cssp_solver.candidate_pruning = True

    # cssp_solver.incremental_update(5)

    # k_best_solution_set = cssp_solver.k_best_solution_set
    # for solution in k_best_solution_set:
    #     print("-"*20)
    #     print(solution[0])
    #     print(solution[1])   
        
    # print(time.time() - t)
    # print(cssp_solver.anytime_solutions)

    # print("-------------------------------------")
    # print("-------------------------------------")
    # print("-------------------------------------")
    # print("-------------------------------------")


    solver = MILPSolver(model, bound)#, cssp_solver.algo)

    solver.encode_MILP()
    

def test_racetrack_simple():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_simple.txt"
    traj_check_dict_file = "models/racetrack_simple_traj_check_dict.json"
    heuristic_file = "models/racetrack_simple_heuristic.json"

    init_state = (1,2,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)


    bound = 10

    # cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-100,convergence_epsilon=1e-100)

    # t = time.time()

    # cssp_solver.solve([[0,1.0]])

    # policy = cssp_solver.algo.extract_policy()


    # cssp_solver.candidate_pruning = True

    # cssp_solver.incremental_update(5)

    # k_best_solution_set = cssp_solver.k_best_solution_set
    # for solution in k_best_solution_set:
    #     print("-"*20)
    #     print(solution[0])
    #     print(solution[1])   
        
    # print(time.time() - t)
    # print(cssp_solver.anytime_solutions)

    # print("-------------------------------------")
    # print("-------------------------------------")
    # print("-------------------------------------")
    # print("-------------------------------------")


    
    solver = MILPSolver(model, bound)#, cssp_solver.algo)

    solver.encode_MILP()
    


def test_grid():

    init_state = (0,0)
    size = (10,10)
    goal = (4,4)
    model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)
    bound = 29

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
    print(cssp_solver.anytime_solutions)

    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")


    solver = MILPSolver(model, bound, cssp_solver.algo)

    solver.encode_MILP()




def test_encoding():

    init_state = (0,0)
    size = (100,100)
    goal = (4,4)
    model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)
    bound = 29


    # map_file = "models/racetrack1.txt"
    # traj_check_dict_file = "models/racetrack1_traj_check_dict.json"
    # init_state = (1,5,0,0)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)
    # bound = 20
   
    solver = MILPSolver(model, bound)

    solver.encode_MILP()

    # bound_list = list(range(30))
    # for bound in bound_list:
    #     solver.bound = bound
    #     solver.encode_MILP()







def test_LAOStar():

    init_state = (0,0)
    size = (30,30)
    goal = (4,4)
    model = GRIDModel(size, init_state, goal, prob_right_transition=0.85)

    alpha = [0.0]
    bounds = [1.5]


    
    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-1, convergence_epsilon=1e-100,\
                   bounds=bounds,alpha=alpha,Lagrangian=True)

    

    t = time.time()
    policy = algo.solve()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(algo.graph.nodes)))

    

    cssp_solver = CSSPSolver(model,bounds=[0.5])


    model.print_policy(policy)

    
    value_1 = algo.graph.root.value_1
    print(value_1)


    

# test_encoding()

test_racetrack_hard()
# test_racetrack()
# test_racetrack_easy()
# test_racetrack_simple()


# test_grid()
