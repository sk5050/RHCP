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
from elevator import ELEVATORModel
from elevator_2011 import ELEVATORModel_2011
from elevator_2011_2 import ELEVATORModel_2011_2
from elevator_2 import ELEVATORModel_2

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




def test_elevator():

    model = ELEVATORModel(n=20, w=2, h=2, prob=0.75, init_state=((4,17),(0,0),7), px_dest=(10,9), hidden_dest=(19,2), hidden_origin=(5,16))

    # print(model.init_state)
    # print(model.state_transitions(model.init_state, 'U'))
    # time.sleep(100)

    alpha = [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    bounds = [30,30,30,30,30,30,15,21]


    
    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-10, convergence_epsilon=1e-10,\
                   bounds=bounds,alpha=alpha,Lagrangian=True)

    

    t = time.time()
    policy = algo.solve()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(algo.graph.nodes)))


    print(policy)

    # for state,node in algo.graph.nodes.items():
    #     print(state)
    #     print(node.terminal)
    
    value_1 = algo.graph.root.value_1
    value_2 = algo.graph.root.value_2
    value_3 = algo.graph.root.value_3
    value_4 = algo.graph.root.value_4
    value_5 = algo.graph.root.value_5
    value_6 = algo.graph.root.value_6
    value_7 = algo.graph.root.value_7
    value_8 = algo.graph.root.value_8
    value_9 = algo.graph.root.value_9

    print(value_1)
    print(value_2)
    print(value_3)
    print(value_4)
    print(value_5)
    print(value_6)
    print(value_7)
    print(value_8)
    print(value_9)



def test_elevator_dual_alg():

    model = ELEVATORModel(n=20, w=2, h=2, prob=0.75, init_state=((4,17),(0,0),7), px_dest=(10,9), hidden_dest=(19,2), hidden_origin=(5,16))

    # print(model.init_state)
    # print(model.state_transitions(model.init_state, 'U'))
    # time.sleep(100)

    alpha = [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    bounds = [30,30,30,30,30,30,15,21]

    
    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-1,convergence_epsilon=1e-10)


    t = time.time()
    try:
        cssp_solver.find_dual_multiple_bounds_generalized([[0,10],[0,10],[0,10],[0,10],[0,10],[0,10],[0,10],[0,10]])
    except:
        
        print("elapsed time: "+str(time.time()-t))
        print("number of states explored: "+str(len(cssp_solver.graph.nodes)))
        print(cssp_solver.anytime_solutions)


    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(cssp_solver.graph.nodes)))
    print(cssp_solver.anytime_solutions)



def test_elevator_2011_dual_alg():

    model = ELEVATORModel_2011_2(n=20, w=2, h=1, prob=0.75,
                                 init_state=((random.randint(1,20),random.randint(1,20)),(0,),random.randint(1,20), 0),
                                 px_dest=(random.randint(1,20),random.randint(1,20)),
                                 hidden_dest=(random.randint(1,20),),
                                 hidden_origin=(random.randint(1,20),))

    print(model.init_state)
    print(model.px_dest)
    print(model.hidden_dest)
    print(model.hidden_origin)




    # init_state = ((11, 11), (0,), 15, 0)
    # px_dest = (1, 8)
    # hidden_dest = (19,)
    # hidden_origin = (7,)
    
    # model = ELEVATORModel_2011_2(n=20, w=2, h=1, prob=0.75,
    #                              init_state=init_state,
    #                              px_dest=px_dest,
    #                              hidden_dest=hidden_dest,
    #                              hidden_origin=hidden_origin)

    bounds = [30,30,30,30,15,21]

    
    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-1,convergence_epsilon=1e-10, VI_max_iter=1000000)


    t = time.time()
    try:

        cssp_solver.find_dual_multiple_bounds_generalized([[0,10],[0,10],[0,10],[0,10],[0,10],[0,10]])
    except:
        
        print("elapsed time: "+str(time.time()-t))
        print("number of states explored: "+str(len(cssp_solver.graph.nodes)))
        print(cssp_solver.anytime_solutions)



        cssp_solver.candidate_pruning = False

        try:
            cssp_solver.incremental_update(5)
            print(cssp_solver.anytime_solutions)
        except:
            print(cssp_solver.anytime_solutions)


    # print("elapsed time: "+str(time.time()-t))
    # print("number of states explored: "+str(len(cssp_solver.graph.nodes)))
    # print(cssp_solver.anytime_solutions)





def test_elevator_2011_two_consts_dual_alg():

    model = ELEVATORModel_2011_2(n=20, w=2, h=1, prob=0.75,
                                 init_state=((random.randint(1,20),random.randint(1,20)),(0,),random.randint(1,20), 0),
                                 px_dest=(random.randint(1,20),random.randint(1,20)),
                                 hidden_dest=(random.randint(1,20),),
                                 hidden_origin=(random.randint(1,20),))

    print(model.init_state)
    print(model.px_dest)
    print(model.hidden_dest)
    print(model.hidden_origin)


    # init_state = ((1, 19), (0,), 18, 0)
    # px_dest = (8, 13)
    # hidden_dest = (7,)
    # hidden_origin = (8,)
    
    # model = ELEVATORModel_2011_2(n=20, w=2, h=1, prob=0.75,
    #                              init_state=init_state,
    #                              px_dest=px_dest,
    #                              hidden_dest=hidden_dest,
    #                              hidden_origin=hidden_origin)



    bounds = [15,21]

    
    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-1,convergence_epsilon=1e-10)


    t = time.time()
    # try:
    cssp_solver.find_dual_line_search([[0,10],[0,10]])


    # except:
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(cssp_solver.graph.nodes)))
    print(cssp_solver.anytime_solutions)


    



def test_elevator_2011():

    # model = ELEVATORModel_2011_2(n=20, w=2, h=2, prob=0.75, init_state=((4,17),(0,0),7,1), px_dest=(10,9), hidden_dest=(19,2), hidden_origin=(5,16))

    model = ELEVATORModel_2011_2(n=20, w=2, h=1, prob=0.75,
                                 init_state=((random.randint(1,20),random.randint(1,20)),(0,),random.randint(1,20), 0),
                                 px_dest=(random.randint(1,20),random.randint(1,20)),
                                 hidden_dest=(random.randint(1,20),),
                                 hidden_origin=(random.randint(1,20),))

    print(model.init_state)
    print(model.px_dest)
    print(model.hidden_dest)
    print(model.hidden_origin)




    alpha = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [30,30,30,30,15,21]


    
    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-10, convergence_epsilon=1e-10,\
                   bounds=bounds,alpha=alpha,Lagrangian=True)

    

    t = time.time()
    policy = algo.solve()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(algo.graph.nodes)))


    print(policy)

    # for state,node in algo.graph.nodes.items():
    #     print(state)
    #     print(node.terminal)
    
    value_1 = algo.graph.root.value_1
    value_2 = algo.graph.root.value_2
    value_3 = algo.graph.root.value_3
    value_4 = algo.graph.root.value_4
    value_5 = algo.graph.root.value_5
    value_6 = algo.graph.root.value_6
    value_7 = algo.graph.root.value_7
    # value_8 = algo.graph.root.value_8
    # value_9 = algo.graph.root.value_9

    print(value_1)
    print(value_2)
    print(value_3)
    print(value_4)
    print(value_5)
    print(value_6)
    print(value_7)
    # print(value_8)
    # print(value_9)







def test_elevator_2():

    model = ELEVATORModel_2(n=20, w=2, h=1, prob=0.75, init_state=((random.randint(1,20),random.randint(1,20)),(0,),(random.randint(1,20), 0), (random.randint(1,20),0)), \
                                 px_dest=(random.randint(1,20),random.randint(1,20)), \
                                 hidden_dest=(random.randint(1,20),), \
                                 hidden_origin=(random.randint(1,20),))


    alpha = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [20,20,20,20,10,21]
    
    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-10, convergence_epsilon=1e-10,\
                   bounds=bounds,alpha=alpha,Lagrangian=True)

    t = time.time()
    policy = algo.solve()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(algo.graph.nodes)))


    print(policy)

    
    value_1 = algo.graph.root.value_1
    value_2 = algo.graph.root.value_2 - bounds[0]
    value_3 = algo.graph.root.value_3 - bounds[1]
    value_4 = algo.graph.root.value_4 - bounds[2]
    value_5 = algo.graph.root.value_5 - bounds[3]
    value_6 = algo.graph.root.value_6 - bounds[4]
    value_7 = algo.graph.root.value_7 - bounds[5]


    print(value_1)
    print(value_2)
    print(value_3)
    print(value_4)
    print(value_5)
    print(value_6)
    print(value_7)





    

# test_elevator()

# test_elevator_dual_alg()

# test_elevator_2011()

# test_elevator_2011_dual_alg()

test_elevator_2011_two_consts_dual_alg()


# test_elevator_2()
