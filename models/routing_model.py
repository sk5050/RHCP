#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import math
# import numpy as np
import time

class ROUTINGModel(object):

    def __init__(self, size=(10,10),init_state=((0,0),(5,5)), goal=(9,9), prob_right_transition=0.9):


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
        return state[0] == self.goal

    
    def state_transitions(self, state, action):
        ac_state = state[0]
        if action=="U":
            new_ac_state = (ac_state[0], ac_state[1]+1)
            if new_ac_state not in self.state_list:
                new_ac_state = ac_state
            
        elif action=="D":
            new_ac_state = (ac_state[0], ac_state[1]-1)
            if new_ac_state not in self.state_list:
                new_ac_state = ac_state

        elif action=="L":
            new_ac_state = (ac_state[0]-1, ac_state[1])
            if new_ac_state not in self.state_list:
                new_ac_state = ac_state

        elif action=="R":
            new_ac_state = (ac_state[0]+1, ac_state[1]+1)
            if new_ac_state not in self.state_list:
                new_ac_state = ac_state


        obs_state = state[1]
        obs_states_temp = [[(obs_state[0],obs_state[1]+1), self.prob_right_transition],
                           [(obs_state[0]+1,obs_state[1]), (1-self.prob_right_transition)/2],
                           [(obs_state[0]-1,obs_state[1]), (1-self.prob_right_transition)/2]]

        prob_stay = 0
        new_states = []
        for new_obs_state in obs_states_temp:
            if new_obs_state[0] not in self.state_list:
                prob_stay = prob_stay + new_obs_state[1]
            else:
                new_state = [(new_ac_state, new_obs_state[0]), new_obs_state[1]]
                new_states.append(new_state)

        if prob_stay > 0:
            new_states.append([(new_ac_state,obs_state), prob_stay])

        return new_states





    def cost(self,state,action):  # cost function should return vector of costs, even though there is a single cost function. 
        cost1 = 1.0


        if abs(state[0][0]-state[1][0]) <= 3 and abs(state[0][1]-state[1][1]) <= 3:
            cost2 = 1.0
        else:
            cost2 = 0.0

        
        return cost1, cost2

    
    def heuristic(self, state,depth=None):
        heuristic1 = math.sqrt((state[0][0]-self.goal[0])**2 + (state[0][1]-self.goal[1])**2)

        heuristic2 = 0

        return heuristic1, heuristic2


