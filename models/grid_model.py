#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import math
# import numpy as np
import time

class GRIDModel(object):

    def __init__(self, size=(3,3),init_state=(0,0), goal=(4,4), prob_right_transition=0.9):

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

        self.action_list = ["U","L","R","D"]

        
    def actions(self, state):
        return self.action_list

    
    def is_terminal(self, state):
        return state == self.goal

    
    def state_transitions(self, state, action):
        if action=="U":
            new_states_temp = [[(state[0],state[1]+1), self.prob_right_transition],
                               [(state[0]+1,state[1]), (1-self.prob_right_transition)/2],
                               [(state[0]-1,state[1]), (1-self.prob_right_transition)/2]]

        elif action=="D":
            new_states_temp = [[(state[0],state[1]-1), self.prob_right_transition],
                               [(state[0]+1,state[1]), (1-self.prob_right_transition)/2],
                               [(state[0]-1,state[1]), (1-self.prob_right_transition)/2]]

        elif action=="L":
            new_states_temp = [[(state[0]-1,state[1]), self.prob_right_transition],
                               [(state[0],state[1]+1), (1-self.prob_right_transition)/2],
                               [(state[0],state[1]-1), (1-self.prob_right_transition)/2]]

        elif action=="R":
            new_states_temp = [[(state[0]+1,state[1]), self.prob_right_transition],
                               [(state[0],state[1]+1), (1-self.prob_right_transition)/2],
                               [(state[0],state[1]-1), (1-self.prob_right_transition)/2]]

        prob_stay = 0
        new_states = []
        for new_state in new_states_temp:
            if new_state[0] not in self.state_list:
                prob_stay = prob_stay + new_state[1]
            else:
                new_states.append(new_state)

        if prob_stay > 0:
            new_states.append([state, prob_stay])

        return new_states


    def cost(self,state,action):
        return 1.0

    
    def heuristic(self, state,depth=None):
        return math.sqrt((state[0]-self.goal[0])**2 + (state[1]-self.goal[1])**2)

