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
from IDUAL import IDUAL


from simple_grid_model import SIMPLEGRIDModel
from grid_model import GRIDModel
from grid_model_multiple_bounds import GRIDModel_multiple_bounds
from racetrack_model import RaceTrackModel
from LAO_paper_model import LAOModel
from manual_model import MANUALModel
from manual_model_2 import MANUALModel2
from manual_model_3 import MANUALModel3
from elevator import ELEVATORModel
from elevator_2011 import ELEVATORModel_2011
from elevator_2011_2 import ELEVATORModel_2011_2
from elevator_2 import ELEVATORModel_2
from routing_model import ROUTINGModel
from cc_routing_model import CCROUTINGModel

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

    solver.solve_opt()



def test_racetrack_hard_full_expansion():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_hard_2.txt"
    traj_check_dict_file = "models/racetrack_hard_traj_check_dict.json"
    heuristic_file = "models/racetrack_hard_heuristic.json"

    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)


    bound = [1]


    solver = MILPSolver(model, bound)

    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))



def test_racetrack_ring_full_expansion():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_ring_2.txt"
    traj_check_dict_file = "models/racetrack_ring_traj_check_dict.json"
    heuristic_file = "models/racetrack_ring_heuristic.json"

    init_state = (1,23,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)

    bound = [1]


    solver = MILPSolver(model, bound)


    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))


 
def test_racetrack_small_full_expansion():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack1.txt"
    traj_check_dict_file = "models/racetrack1_traj_check_dict.json"
    heuristic_file = "models/racetrack1_heuristic.json"

    init_state = (1,5,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)

    bound = [1]


    solver = MILPSolver(model, bound)


    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))   

    

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





def test_elevator():

    
    # model = ELEVATORModel(n=20, w=2, h=2, prob=0.75, init_state=((4,17),(0,0),7), px_dest=(10,9), hidden_dest=(19,2), hidden_origin=(5,16))
    model = ELEVATORModel(n=20, w=2, h=2, prob=0.75, init_state=((random.randint(1,20),random.randint(1,20)),(0,0),random.randint(1,20)), px_dest=(random.randint(1,20),random.randint(1,20)), hidden_dest=(random.randint(1,20),random.randint(1,20)), hidden_origin=(random.randint(1,20),random.randint(1,20)))


    bounds = [30,30,30,30,30,30,15,21]


    solver = MILPSolver(model, bounds)


    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))

    


def test_elevator_2011():

    # model = ELEVATORModel_2011_2(n=20, w=2, h=2, prob=0.75, init_state=((4,17),(0,0),7,1), px_dest=(10,9), hidden_dest=(19,2), hidden_origin=(5,16))

    # model = ELEVATORModel_2011_2(n=20, w=2, h=1, prob=0.75, init_state=((random.randint(1,20),random.randint(1,20)),(0,),random.randint(1,20), 0), px_dest=(random.randint(1,20),random.randint(1,20)), hidden_dest=(random.randint(1,20),), hidden_origin=(random.randint(1,20),))


    init_state = ((11, 11), (0,), 15, 0)
    px_dest = (1, 8)
    hidden_dest = (19,)
    hidden_origin = (7,)

    init_state = ((17, 5), (0,), 11, 0)
    px_dest = (17, 9)
    hidden_dest = (15,)
    hidden_origin = (15,) 

    model = ELEVATORModel_2011_2(n=20, w=2, h=1, prob=0.75,
                                 init_state=init_state,
                                 px_dest=px_dest,
                                 hidden_dest=hidden_dest,
                                 hidden_origin=hidden_origin)

    bounds = [30,30,30,30,15,21]


    solver = MILPSolver(model, bounds)


    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))





def test_elevator_2011_two_consts():

    # model = ELEVATORModel_2011_2(n=20, w=2, h=2, prob=0.75, init_state=((4,17),(0,0),7,1), px_dest=(10,9), hidden_dest=(19,2), hidden_origin=(5,16))

    # model = ELEVATORModel_2011_2(n=20, w=2, h=1, prob=0.75, init_state=((random.randint(1,20),random.randint(1,20)),(0,),random.randint(1,20), 0), px_dest=(random.randint(1,20),random.randint(1,20)), hidden_dest=(random.randint(1,20),), hidden_origin=(random.randint(1,20),))


    init_state = ((11, 11), (0,), 15, 0)
    px_dest = (1, 8)
    hidden_dest = (19,)
    hidden_origin = (7,)

    init_state = ((9, 10), (0,), 1, 0)
    px_dest = (2, 19)
    hidden_dest = (17,)
    hidden_origin = (16,) 

    model = ELEVATORModel_2011_2(n=20, w=2, h=1, prob=0.75,
                                 init_state=init_state,
                                 px_dest=px_dest,
                                 hidden_dest=hidden_dest,
                                 hidden_origin=hidden_origin)

    bounds = [15,21]


    solver = MILPSolver(model, bounds)


    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))

    
    


def test_elevator_2():


    # init_state = ((15, 11), (0,), (2, 0), (19, 0))
    # px_dest = (16, 13)
    # hidden_dest = (8,)
    # hidden_origin = (9,) 

    

    # init_state = ((7, 16), (0,), (18, 0), (15, 0))
    # px_dest = (20, 15)
    # hidden_dest = (20,)
    # hidden_origin = (7,) 


    # init_state = ((7, 13), (0,), (5, 0), (1, 0))
    # px_dest = (13, 19)
    # hidden_dest = (6,)
    # hidden_origin = (20,)


    # init_state = ((5, 16), (0,), (12, 0), (16, 0))
    # px_dest = (2, 2)
    # hidden_dest = (16,)
    # hidden_origin = (2,)



    # init_state = ((15, 11), (0,), (14, 0), (2, 0))
    # px_dest = (2, 14)
    # hidden_dest = (16,)
    # hidden_origin = (20,)


    # init_state = ((14, 9), (0,), (15, 0), (1, 0))
    # px_dest = (13, 16)
    # hidden_dest = (2,)
    # hidden_origin = (15,)



    # init_state = ((7, 2), (0,), (17, 0), (12, 0))
    # px_dest = (6, 7)
    # hidden_dest = (8,)
    # hidden_origin = (11,) 


    # init_state = ((7, 15), (0,), (5, 0), (11, 0))
    # px_dest = (6, 12)
    # hidden_dest = (20,)
    # hidden_origin = (6,)



    # init_state = ((10, 3), (0,), (11, 0), (4, 0))
    # px_dest = (18, 20)
    # hidden_dest = (9,)
    # hidden_origin = (1,)



    init_state = ((10, 11), (0,), (15, 0), (1, 0))
    px_dest = (20, 9)
    hidden_dest = (16,)
    hidden_origin = (3,)

    model = ELEVATORModel_2(n=20, w=2, h=1, prob=0.75, init_state=init_state, \
                                 px_dest=px_dest, \
                                 hidden_dest=hidden_dest, \
                                 hidden_origin=hidden_origin)

    # model = ELEVATORModel_2(n=20, w=2, h=1, prob=0.75, init_state=((random.randint(1,20),random.randint(1,20)),(0,),(random.randint(1,20), 0), (random.randint(1,20),0)), \
    #                              px_dest=(random.randint(1,20),random.randint(1,20)), \
    #                              hidden_dest=(random.randint(1,20),), \
    #                              hidden_origin=(random.randint(1,20),))

    bounds = [15,21,15,21,15,21]


    solver = MILPSolver(model, bounds)


    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))



def test_elevator_2_idual():


    # init_state = ((15, 11), (0,), (2, 0), (19, 0))
    # px_dest = (16, 13)
    # hidden_dest = (8,)
    # hidden_origin = (9,) 

    

    # init_state = ((7, 16), (0,), (18, 0), (15, 0))
    # px_dest = (20, 15)
    # hidden_dest = (20,)
    # hidden_origin = (7,) 


    # init_state = ((7, 13), (0,), (5, 0), (1, 0))
    # px_dest = (13, 19)
    # hidden_dest = (6,)
    # hidden_origin = (20,)


    # init_state = ((5, 16), (0,), (12, 0), (16, 0))
    # px_dest = (2, 2)
    # hidden_dest = (16,)
    # hidden_origin = (2,)



    # init_state = ((15, 11), (0,), (14, 0), (2, 0))
    # px_dest = (2, 14)
    # hidden_dest = (16,)
    # hidden_origin = (20,)


    # init_state = ((14, 9), (0,), (15, 0), (1, 0))
    # px_dest = (13, 16)
    # hidden_dest = (2,)
    # hidden_origin = (15,)



    # init_state = ((7, 2), (0,), (17, 0), (12, 0))
    # px_dest = (6, 7)
    # hidden_dest = (8,)
    # hidden_origin = (11,) 


    # init_state = ((7, 15), (0,), (5, 0), (11, 0))
    # px_dest = (6, 12)
    # hidden_dest = (20,)
    # hidden_origin = (6,)



    # init_state = ((10, 3), (0,), (11, 0), (4, 0))
    # px_dest = (18, 20)
    # hidden_dest = (9,)
    # hidden_origin = (1,)



    # init_state = ((10, 11), (0,), (15, 0), (1, 0))
    # px_dest = (20, 9)
    # hidden_dest = (16,)
    # hidden_origin = (3,)





    model = ELEVATORModel_2(n=20, w=2, h=1, prob=0.75, init_state=init_state, \
                                 px_dest=px_dest, \
                                 hidden_dest=hidden_dest, \
                                 hidden_origin=hidden_origin)

    # model = ELEVATORModel_2(n=20, w=2, h=1, prob=0.75, init_state=((random.randint(1,20),random.randint(1,20)),(0,),(random.randint(1,20), 0), (random.randint(1,20),0)), \
    #                              px_dest=(random.randint(1,20),random.randint(1,20)), \
    #                              hidden_dest=(random.randint(1,20),), \
    #                              hidden_origin=(random.randint(1,20),))

    bounds = [15,21,15,21,15,21]

    t = time.time()

    solver = IDUAL(model, bounds)
    solver.solve_LP_and_MILP()

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))

    


def test_idual():
    
    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_hard (copy).txt"
    traj_check_dict_file = "models/racetrack_hard_traj_check_dict.json"
    heuristic_file = "models/racetrack_hard_heuristic.json"

    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)


    bound = [1]

    t = time.time()
    solver = IDUAL(model, bound)

    # solver.solve()

    solver.solve_LP_and_MILP()


    print("number of nodes expanded: " + str(len(solver.graph.nodes)))
    print("elapsed time: "+str(time.time() - t))








def test_routing():


    init_state = ((0,0),(3,1))
    size = (12,12)
    goal = (5,5)
    model = ROUTINGModel(size, init_state, goal, prob_right_transition=0.8)

    bounds = [2] 

    solver = MILPSolver(model, bounds)


    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))




def test_routing_IDUAL():


    init_state = ((0,0),(3,1))
    size = (12,12)
    goal = (5,5)
    model = ROUTINGModel(size, init_state, goal, prob_right_transition=0.8)

    bounds = [2] 


    t = time.time()
    solver = IDUAL(model, bounds)
    solver.solve_LP_and_MILP()

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))
    





def racetrack_large_IDUAL():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_hard_2.txt"
    traj_check_dict_file = "models/racetrack_hard_traj_check_dict.json"
    heuristic_file = "models/racetrack_hard_heuristic.json"

    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)


    bounds = [1]


    t = time.time()
    solver = IDUAL(model, bounds)
    solver.solve_LP_and_MILP()

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))
        



def racetrack_ring_IDUAL():


    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_ring_2.txt"
    traj_check_dict_file = "models/racetrack_ring_traj_check_dict.json"
    heuristic_file = "models/racetrack_ring_heuristic.json"

    init_state = (1,23,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    # model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, slip_prob=0.1)


    bounds = [1]


    t = time.time()
    solver = IDUAL(model, bounds)
    solver.solve_LP_and_MILP()

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))





def test_cc_routing():

    # init_state = ((0,0),(2,0))
    # size = (3,2)
    # goal = (2,1)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir='U',obs_boundary=[(0,0)], init_state=init_state, goal=goal, prob_right_transition=0.99)

    # bounds = [0.1]

    #########################################################################################################


    # # # obj 1 scenario 1

    # init_state = ((0,0),(1,5))
    # size = (10,10)
    # goal = (5,7)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir=['R'],obs_boundary=[(2,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    # bounds = [0.001]
    



    # obj 1 scenario 2

    # init_state = ((0,0),(6,1))
    # size = (10,10)
    # goal = (8,8)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir=['U'],obs_boundary=[(1,2)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    # bounds = [0.001]

    

    # # # # obj 1 scenario 3

    # init_state = ((0,0),(2,2))
    # size = (10,15)
    # goal = (5,6)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir=['R'],obs_boundary=[(1,2)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    # bounds = [0.001]   
    



    # # # # obj 1 scenario 4

    # init_state = ((0,0),(7,5))
    # size = (10,15)
    # goal = (8,12)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir=['L'],obs_boundary=[(1,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    # bounds = [0.001]


    # # # obj 1 scenario 5

    # init_state = ((0,0),(12,5))
    # size = (15,15)
    # goal = (6,5)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir=['L'],obs_boundary=[(2,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    # bounds = [0.001]    

    
    # # # obj 1 scenario 6

    init_state = ((0,0),(7,7))
    size = (15,15)
    goal = (13,13)
    model = CCROUTINGModel(size, obs_num=1, obs_dir=['L'],obs_boundary=[(2,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.001]



    # # obj 2 scenario 1 (goal is close, so i-dual performs well)
    # init_state = ((0,0),(7,1),(1,8))
    # size = (10,10)
    # goal = (5,2)
    # model = CCROUTINGModel(size, obs_num=2, obs_dir=['U','R'],obs_boundary=[(1,2),(1,1)], init_state=init_state, goal=goal, prob_right_transition=0.9)

    # bounds = [0.001]

    # obj 2 scenario 2 
    # init_state = ((0,0),(7,1),(1,8))
    # size = (10,10)
    # goal = (8,5)
    # model = CCROUTINGModel(size, obs_num=2, obs_dir=['U','R'],obs_boundary=[(1,2),(1,1)], init_state=init_state, goal=goal, prob_right_transition=0.9)

    # bounds = [0.002]


    # obj 2 scenario 3
    # init_state = ((1,1),(7,5),(5,10))
    # size = (15,10)
    # goal = (13,8)
    # model = CCROUTINGModel(size, obs_num=2, obs_dir=['R','D'],obs_boundary=[(2,1),(1,2)], init_state=init_state, goal=goal, prob_right_transition=0.6)

    # bounds = [0.002]
    


    solver = MILPSolver(model, bounds)


    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))



def test_cc_routing_idual():

    #########################################################################################################


    # # obj 1 scenario 1

    # init_state = ((0,0),(1,5))
    # size = (10,10)
    # goal = (5,7)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir=['R'],obs_boundary=[(2,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    # bounds = [0.001]



    # obj 1 scenario 2

    # init_state = ((0,0),(6,1))
    # size = (10,10)
    # goal = (8,8)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir=['U'],obs_boundary=[(1,2)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    # bounds = [0.001]

    

    # # # # obj 1 scenario 3

    # init_state = ((0,0),(2,2))
    # size = (10,15)
    # goal = (5,6)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir=['R'],obs_boundary=[(1,2)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    # bounds = [0.001]   
    



    # # # # obj 1 scenario 4

    # init_state = ((0,0),(7,5))
    # size = (10,15)
    # goal = (8,12)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir=['L'],obs_boundary=[(1,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    # bounds = [0.001]


    # # # obj 1 scenario 5

    # init_state = ((0,0),(12,5))
    # size = (15,15)
    # goal = (6,5)
    # model = CCROUTINGModel(size, obs_num=1, obs_dir=['L'],obs_boundary=[(2,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    # bounds = [0.001]      

    init_state = ((0,0),(7,7))
    size = (15,15)
    goal = (13,13)
    model = CCROUTINGModel(size, obs_num=1, obs_dir=['L'],obs_boundary=[(2,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.001]



    # # obj 2 scenario 1
    # init_state = ((0,0),(7,1),(1,8))
    # size = (10,10)
    # goal = (5,2)
    # model = CCROUTINGModel(size, obs_num=2, obs_dir=['U','R'],obs_boundary=[(1,2),(1,1)], init_state=init_state, goal=goal, prob_right_transition=0.9)

    # bounds = [0.001]


    # # obj 2 scenario 2 
    # init_state = ((0,0),(7,1),(1,8))
    # size = (10,10)
    # goal = (8,5)
    # model = CCROUTINGModel(size, obs_num=2, obs_dir=['U','R'],obs_boundary=[(1,2),(1,1)], init_state=init_state, goal=goal, prob_right_transition=0.9)

    # bounds = [0.002]



    t = time.time()
    solver = IDUAL(model, bounds)
    solver.solve_LP_and_MILP()
    
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))


    
    

# test_encoding()

# test_racetrack_hard()
# test_racetrack()
# test_racetrack_easy()
# test_racetrack_simple()

# test_idual()

# test_racetrack_hard_full_expansion()


# test_grid()

# test_elevator()

# test_elevator_2011()
# test_elevator_2()
# test_elevator_2_idual()

# test_elevator_2011_two_consts()
# test_routing()
# test_routing_IDUAL()
# test_racetrack_ring_full_expansion()
# test_racetrack_small_full_expansion()


# racetrack_large_IDUAL()
# racetrack_ring_IDUAL()


# test_cc_routing()
test_cc_routing_idual()

