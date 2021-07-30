#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import math
# import numpy as np
import time

class ELEVATORModel_2011(object):

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

        self.action_list = ["OU","OD","C","M"]

        
    def actions(self, state):

        if state[4]==1:
            return ['C']
        else:
            return ['OU','OD','M']

    
    def is_terminal(self, state):

        if max(state[0]) == -2 and max(state[1]) == -2:
            return True
        else:
            return False

    
    # def state_transitions(self, state, action):

    #     px_at = state[0]
    #     hidden_at = state[1]
    #     current_pos = state[2]

    #     new_px_pos = []
    #     for px_ind in range(self.w):
    #         px_pos = px_at[px_ind]
    #         if px_pos==-2:
    #             new_px_pos.append(px_pos)
    #             continue
    #         elif px_pos==-1:
    #             if current_pos==self.px_dest[px_ind]:
    #                 new_px_pos.append(-2)
    #             else:
    #                 new_px_pos.append(px_pos)
    #         else:
    #             if current_pos==px_pos:
    #                 new_px_pos.append(-1)
    #             else:
    #                 new_px_pos.append(px_pos)


    #     new_hidden_pos = []
    #     unarrived_hidden_ind_list = []
    #     for hidden_ind in range(self.h):
    #         hidden_pos = hidden_at[hidden_ind]
    #         if hidden_pos==0:
    #             unarrived_hidden_ind_list.append(hidden_ind)
    #             new_hidden_pos.append(0)
    #             continue
                
    #         elif hidden_pos==-2:
    #             new_hidden_pos.append(hidden_pos)
    #             continue
    #         elif hidden_pos==-1:
    #             if current_pos==self.hidden_dest[hidden_ind]:
    #                 new_hidden_pos.append(-2)
    #             else:
    #                 new_hidden_pos.append(hidden_pos)
    #         else:
    #             if current_pos==hidden_pos:
    #                 new_hidden_pos.append(-1)
    #             else:
    #                 new_hidden_pos.append(hidden_pos)


    #     if action=='U':
    #         new_pos = min(20, current_pos + 1)
    #     elif action=='D':
    #         new_pos = max(1, current_pos - 1)
    #     elif action=='S':
    #         new_pos = current_pos
                    
    #     new_states = [[[new_px_pos, new_hidden_pos, new_pos], 1.0]]

    #     for unarrived_hidden_ind in unarrived_hidden_ind_list:
    #         new_states_temp = []
    #         for new_state in new_states:
    #             for trans in ['NA', 'A']:
    #                 new_hidden_pos_temp = new_state[0][1].copy()
    #                 if trans=='NA':
    #                     new_states_temp.append([[new_px_pos, new_hidden_pos_temp, new_pos], new_state[1]*0.25])
    #                 else:
    #                     new_hidden_pos_temp[unarrived_hidden_ind] = self.hidden_origin[unarrived_hidden_ind]
    #                     new_states_temp.append([[new_px_pos, new_hidden_pos_temp, new_pos], new_state[1]*0.75])

    #         new_states = new_states_temp
                


    #     new_states_in_tuple = []
    #     for new_state in new_states:
    #         new_states_in_tuple.append([(tuple(new_state[0][0]), tuple(new_state[0][1]), new_state[0][2]), new_state[1]])

            
    #     return new_states_in_tuple




    def state_transitions(self, state, action):

        px_at = state[0]
        hidden_at = state[1]
        current_pos = state[2]
        current_dir = state[3]
        is_open = state[4]

        new_px_pos = []
        for px_ind in range(self.w):
            px_pos = px_at[px_ind]
            if px_pos==-2:
                new_px_pos.append(px_pos)
                continue
            elif px_pos==-1:
                if current_pos==self.px_dest[px_ind] and (action=='OU' or action=='OD'):
                    new_px_pos.append(-2)
                else:
                    new_px_pos.append(px_pos)
            else:
                if current_pos==px_pos and (action=='OU' or action=='OD'):
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
                if current_pos==self.hidden_dest[hidden_ind] and (action=='OU' or action=='OD'):
                    new_hidden_pos.append(-2)
                else:
                    new_hidden_pos.append(hidden_pos)
            else:
                if current_pos==hidden_pos and (action=='OU' or action=='OD'):
                    new_hidden_pos.append(-1)
                else:
                    new_hidden_pos.append(hidden_pos)


        if action=='M':
            if current_dir==1:
                new_pos = min(20, current_pos + 1)
            elif current_dir==-1:
                new_pos = max(1, current_pos - 1)
        else:
            new_pos = current_pos

        if action=='OU':
            new_dir = 1
        elif action=='OD':
            new_dir = -1
        else:
            new_dir = current_dir

        if action=='OU' or action=='OD':
            new_is_open = 1
        elif action=='C':
            new_is_open = 0
        else:
            new_is_open = is_open

        new_states = [[[new_px_pos, new_hidden_pos, new_pos, new_dir, new_is_open], 1.0]]

        for unarrived_hidden_ind in unarrived_hidden_ind_list:
            new_states_temp = []
            for new_state in new_states:
                for trans in ['NA', 'A']:
                    new_hidden_pos_temp = new_state[0][1].copy()
                    if trans=='NA':
                        new_states_temp.append([[new_px_pos, new_hidden_pos_temp, new_pos,new_dir, new_is_open], new_state[1]*0.25])
                    else:
                        new_hidden_pos_temp[unarrived_hidden_ind] = self.hidden_origin[unarrived_hidden_ind]
                        new_states_temp.append([[new_px_pos, new_hidden_pos_temp, new_pos,new_dir,new_is_open], new_state[1]*0.75])

            new_states = new_states_temp
                


        new_states_in_tuple = []
        for new_state in new_states:
            new_states_in_tuple.append([(tuple(new_state[0][0]), tuple(new_state[0][1]), new_state[0][2], new_state[0][3], new_state[0][4]), new_state[1]])

  

        # print("----------------")
        # print(state)
        # print(action)
        # print(new_states)
        # time.sleep(1)
            
        return new_states_in_tuple
    

    


    def cost(self,state,action):  # cost function should return vector of costs, even though there is a single cost function.
        
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


        px_pos = state[0]
        hidden_pos = state[1]
        elev_pos = state[2]

        px_1_w = 0
        px_1_t = 0
        px_2_w = 0
        px_2_t = 0
        h_1_w = 0
        h_1_t = 0


        if px_pos[0]>0:
            px_1_w = abs(elev_pos - px_pos[0])

        if px_pos[0]==-1:
            px_1_t = abs(self.px_dest[0]-elev_pos)
        elif px_pos[0]>0:
            px_1_t = abs(self.px_dest[0]-px_pos[0])
        elif px_pos[0]==-2:
            px_1_t = 0


        if px_pos[1]>0:
            px_2_w = abs(elev_pos - px_pos[1])

        if px_pos[1]==-1:
            px_2_t = abs(self.px_dest[1]-elev_pos)
        elif px_pos[1]>0:
            px_2_t = abs(self.px_dest[1]-px_pos[1])
        elif px_pos[1]==-2:
            px_2_t = 0


        if hidden_pos[0]>0:
            h_1_w = abs(elev_pos - hidden_pos[0])

        if hidden_pos[0]==-1:
            h_1_t = abs(self.hidden_dest[0]-elev_pos)
        elif hidden_pos[0]>0:
            h_1_t = abs(self.hidden_dest[0]-hidden_pos[0])
        elif hidden_pos[0]==-2:
            h_1_t = 0

            

        h_cost = max(px_1_w+px_1_t, px_2_w+px_2_t, h_1_w+h_1_t)
            
            
        
        return h_cost, px_1_w, px_1_t, px_2_w, px_2_t, h_1_w, h_1_t  #,0,0






        
        return 0,0,0,0,0,0,0,0,0


