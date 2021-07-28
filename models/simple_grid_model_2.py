#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import math
# import numpy as np
import time
import random

class SIMPLEGRIDModel2(object):

    def __init__(self, size=(3,2),init_state=(0,0), goal=(2,1), prob_right_transition=0.9):

        # grid with size (x,y): width is x and height is y. index start from 0.
        # example of (3,3)
        # _____________
        # |0,2|1,2|2,2|
        # |___|___|___|
        # |0,1|1,1|2,1|
        # |___|___|___|
        # |0,0|1,0|2,0|
        # |___|___|___|
        #

        self.init_state=init_state
        self.prob_right_transition = prob_right_transition
        self.size = size
        self.state_list = []
        self.goal = goal

        for i in range(size[0]):
            for j in range(size[1]):
                self.state_list.append((i,j))

        self.action_list = ["L","V"]

        self.cost1_L = random.randrange(1,10)
        self.cost1_V = random.randrange(1,10)
        self.cost2 = random.randrange(1,10)

        print(self.cost1_L)
        print(self.cost1_V)
        print(self.cost2)
        
    def actions(self, state):
        return self.action_list

    
    def is_terminal(self, state):
        return state == self.goal

    
    def state_transitions(self, state, action):
        if action=="L":
            if state == (0,0):
                new_states = [[(1,0),self.prob_right_transition],
                              [(0,1),1-self.prob_right_transition]]

            elif state == (0,1):
                new_states = [[(1,1),self.prob_right_transition],
                              [(0,0),1-self.prob_right_transition]]

            elif state == (1,0):
                new_states = [[(2,0),self.prob_right_transition],
                              [(1,1),1-self.prob_right_transition]]

            elif state == (1,1):
                new_states = [[(2,1),self.prob_right_transition],
                              [(1,0),1-self.prob_right_transition]]

            elif state == (2,0):
                new_states = [[(2,0),self.prob_right_transition],
                              [(2,1),1-self.prob_right_transition]]

            else:
                raise ValueError("invalid state transition.")

        elif action=="V":
            if state == (0,0):
                new_states = [[(0,1),self.prob_right_transition],
                              [(1,0),1-self.prob_right_transition]]

            elif state == (0,1):
                new_states = [[(0,0),self.prob_right_transition],
                              [(1,1),1-self.prob_right_transition]]

            elif state == (1,0):
                new_states = [[(1,1),self.prob_right_transition],
                              [(2,0),1-self.prob_right_transition]]

            elif state == (1,1):
                new_states = [[(1,0),self.prob_right_transition],
                              [(2,1),1-self.prob_right_transition]]

            elif state == (2,0):
                new_states = [[(2,1),self.prob_right_transition],
                              [(2,0),1-self.prob_right_transition]]

            else:
                raise ValueError("invalid state transition.")

        return new_states


    def cost(self,state,action):  # cost function should return vector of costs, even though there is a single cost function.
        # if action=="L":
        #     cost1 = 10.0
        # elif action=="V":
        #     cost1 = 1.0
            
        # if state==(1,0):
        #     cost2 = 7.0
        # else:
        #     cost2 = 0.0

        if action=="L":
            cost1 = self.cost1_L
        elif action=="V":
            cost1 = self.cost1_V
            
        if state==(1,0):
            cost2 = self.cost2
        else:
            cost2 = 0.0

        return cost1, cost2

    
    def heuristic(self, state,depth=None):
        heuristic1 = math.sqrt((state[0]-self.goal[0])**2 + (state[1]-self.goal[1])**2)
        heuristic2 = 0
        return heuristic1, heuristic2

