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

    model = ELEVATORModel(n=20, w=1, h=2, prob=0.75, init_state=[[15],[0,0],1], px_dest=(10,), hidden_dest=(2,5), hidden_origin=(8,10))

    alpha = [0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [30,30,30,15,30]


    
    algo = ILAOStar(model,constrained=True,VI_epsilon=1e-1, convergence_epsilon=1e-5,\
                   bounds=bounds,alpha=alpha,Lagrangian=True)

    

    t = time.time()
    policy = algo.solve()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(algo.graph.nodes)))


    value_1 = algo.graph.root.value_1
    value_2 = algo.graph.root.value_2
    value_3 = algo.graph.root.value_3
    value_4 = algo.graph.root.value_4
    value_5 = algo.graph.root.value_5


test_elevator()




