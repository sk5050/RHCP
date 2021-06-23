#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import math
# import numpy as np
import time

class MANUALModel(object):

    def __init__(self, cost=1.0):

        self.init_state="0"

        self.cost_param = cost
        self.cost_param_2 = cost
 

        
    def actions(self, state):
        if state=="2":
            return ["A","B"]
        else:
            return ['A']

    
    def is_terminal(self, state):
        return state == "9"

    
    def state_transitions(self, state, action):
        if action=="A":

            if state=="0":
                return [["1",0.5],["2",0.5]]
            elif state=="1":
                return [["3",0.5],["4",0.5]]
            elif state=="2":
                return [["5",0.5],["6",0.5]]
            elif state=="3":
                return [["7",1.0]]
            elif state=="4":
                return [["7",1.0]]
            elif state=="5":
                return [["9",1.0]]
            elif state=="6":
                return [["8",0.5],["9",0.5]]
            elif state=="7":
                return [["3",0.5],["9",0.5]]
            elif state=="8":
                return [["9",1.0]]
            elif state=="10":
                return [["9",1.0]]
            elif state=="11":
                return [["9",1.0]]

        if action=="B":
            if state=="2":
                return [["10",0.5],["11",0.5]]
            else:
                raise ValueError("something wrong")

        return new_states




    def cost(self,state,action):  # cost function should return vector of costs, even though there is a single cost function.
        if action=="B":
            return self.cost_param_2, self.cost_param_2
        else:
            return self.cost_param, self.cost_param
        # return 1.0, 1.0

    
    def heuristic(self, state,depth=None):
        return 0.0, 0.0

