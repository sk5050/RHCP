#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import math
# import numpy as np
import time

class MANUALModel3(object):

    def __init__(self, cost=1.0):

        self.init_state="0"

        self.cost_param = cost
        self.cost_param_2 = cost
 

        
    def actions(self, state):
            return ['A']

    
    def is_terminal(self, state):
        return state == "3"

    
    def state_transitions(self, state, action):
        if state=="0":
            return [["1",0.5],["2",0.5]]
        elif state=="1":
            # return [["0",0.1],["2",0.1],["3",0.8]]
            return [["3",1.0]]
        elif state=="2":
            return [["3",1.0]]
            # return [["1",0.3],["3",0.7]]
            # return [["0",0.1],["1",0.3],["3",0.6]]


        return new_states




    def cost(self,state,action):  # cost function should return vector of costs, even though there is a single cost function.
        return self.cost_param, self.cost_param
        # return 1.0, 1.0

    
    def heuristic(self, state,depth=None):
        return 0.0, 0.0

