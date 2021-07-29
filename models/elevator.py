#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import math
# import numpy as np
import time

class ELEVATORModel(object):

    def __init__(self, n=20, w=1, h=1, prob=0.75, init_state=None, px_dest=None, hidden_dest=None, hidden_origin=None):

        self.init_state=init_state
        self.n = n
        self.w = w
        self.h = h
        self.prob = prob
        self.px_dest = px_dest
        self.hidden_dest = hidden_dest
        self.hidden_origin = hidden_origin
        # self.goal = goal

        self.action_list = ["U","D"]#,"S"]

        
    def actions(self, state):
        return self.action_list

    
    def is_terminal(self, state):

        if max(state[0]) == -2 and max(state[1]) == -2:
            return True
        else:
            return False

    
    def state_transitions(self, state, action):

        px_at = state[0]
        hidden_at = state[1]
        current_pos = state[2]

        new_px_pos = []
        for px_ind in range(self.w):
            px_pos = px_at[px_ind]
            if px_pos==-2:
                new_px_pos.append(px_pos)
                continue
            elif px_pos==-1:
                if current_pos==self.px_dest[px_ind]:
                    new_px_pos.append(-2)
                else:
                    new_px_pos.append(px_pos)
            else:
                if current_pos==px_pos:
                    new_px_pos.append(-1)
                else:
                    new_px_pos.append(px_pos)


        new_hidden_pos = []
        unarrived_hidden_ind_list = []
        for hidden_ind in range(self.h):
            hidden_pos = hidden_at[hidden_ind]
            if hidden_pos==0:
                unarrived_hidden_ind_list.append(hidden_ind)
                new_hidden_pos.append(0)
                continue
                
            elif hidden_pos==-2:
                new_hidden_pos.append(hidden_pos)
                continue
            elif hidden_pos==-1:
                if current_pos==self.hidden_dest[hidden_ind]:
                    new_hidden_pos.append(-2)
                else:
                    new_hidden_pos.append(hidden_pos)
            else:
                if current_pos==hidden_pos:
                    new_hidden_pos.append(-1)
                else:
                    new_hidden_pos.append(hidden_pos)


        if action=='U':
            new_pos = min(20, current_pos + 1)
        elif action=='D':
            new_pos = max(1, current_pos - 1)
        elif action=='S':
            new_pos = current_pos
                    
        new_states = [[[new_px_pos, new_hidden_pos, new_pos], 1.0]]

        for unarrived_hidden_ind in unarrived_hidden_ind_list:
            new_states_temp = []
            for new_state in new_states:
                for trans in ['NA', 'A']:
                    new_hidden_pos_temp = new_state[0][1].copy()
                    if trans=='NA':
                        new_states_temp.append([[new_px_pos, new_hidden_pos_temp, new_pos], new_state[1]*0.25])
                    else:
                        new_hidden_pos_temp[unarrived_hidden_ind] = self.hidden_origin[unarrived_hidden_ind]
                        new_states_temp.append([[new_px_pos, new_hidden_pos_temp, new_pos], new_state[1]*0.75])

            new_states = new_states_temp
                


        new_states_in_tuple = []
        for new_state in new_states:
            new_states_in_tuple.append([(tuple(new_state[0][0]), tuple(new_state[0][1]), new_state[0][2]), new_state[1]])

        return new_states_in_tuple



    def cost(self,state,action):  # cost function should return vector of costs, even though there is a single cost function.
        if action=='S':
            cost1 = 0.0
        else:
            cost1 = 1.0

        if state[0][0]>0:
            cost2 = 1.0
        else:
            cost2 = 0.0

        if state[0][0]==-1:
            cost3 = 1.0
        else:
            cost3 = 0.0

        if state[0][1]>0:
            cost4 = 1.0
        else:
            cost4 = 0.0

        if state[0][1]==-1:
            cost5 = 1.0
        else:
            cost5 = 0.0

        if state[1][0]>0:
            cost6 = 1.0
        else:
            cost6 = 0.0

        if state[1][0]==-1:
            cost7 = 1.0
        else:
            cost7 = 0.0

        if state[1][1]>0:
            cost8 = 1.0
        else:
            cost8 = 0.0

        if state[1][1]==-1:
            cost9 = 1.0
        else:
            cost9 = 0.0
            
            

        return cost1, cost2, cost3, cost4, cost5, cost6, cost7, cost8, cost9

    
    def heuristic(self, state,depth=None):
        return 0,0,0,0,0,0,0,0,0


