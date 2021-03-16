#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import sys
from utils import import_models
import_models()
from grid_model import GRIDModel
from raostar_lagrange import RAOStar
from raostar import RAOStar as RAOStar_original

from raostar_k_shortest import RAOStar_K_Shortest
import graph_to_json
import time
import copy
import random
# import numpy as np
from collections import deque
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# from iterative_raostar import *

# Now you can give command line cc argument after filename
if __name__ == '__main__':

    ## define model
    size = (5,5)
    constraint_states = [(0,1),(1,1),(3,3),(4,3),(0,4)]
    model = GRIDModel(size, constraint_states, prob_right_transition=0.85, prob_right_observation=0.85)

    
    ## set parameters
    cc = 0.3
    horizon = 3
    max_k = 10000
    epsilon = 0.01
    

    ## define problem
    b_init = {(0, 0): 1.0}
    # b_init = {(3, 0): 0.06243192399373029, (2, 1): 0.000528949488345008, (2, 0): 0.7755024284319243, (1, 0): 0.14694153248748237, (0, 1): 0.00696688425189551, (0, 0): 0.0006576128774019019, (1, 1): 0.006480822559902801, (1, 2): 3.7842173252027612e-06, (0, 2): 0.00048606169199271013}
    num_steps = 1
    alpha_zero = 0
    alpha_infty = 10000
    report = True

    # wrong_exs = [{(1, 0): 0.47887323943661975, (0, 1): 0.042253521126760576, (0, 0): 0.47887323943661975},
    #              {(2, 0): 0.06805471064973803, (0, 0): 0.08606919288055105, (1, 0): 0.839341431346769, (1, 1): 0.0005298377126709711, (0, 1): 0.006004827410271005},
    #              {(3, 0): 0.06243192399373029, (2, 1): 0.000528949488345008, (2, 0): 0.7755024284319243, (1, 0): 0.14694153248748237, (0, 1): 0.00696688425189551, (0, 0): 0.0006576128774019019, (1, 1): 0.006480822559902801, (1, 2): 3.7842173252027612e-06, (0, 2): 0.00048606169199271013}]

    

    prev_num_nodes = 0


    

    ## output file
    if report == True:
        F = open('online_results.txt', 'w')
    else:
        F = None


    for it in range(num_steps):

        if F!=None:
            F.write("------------------------------------------------------------------\n")
            F.write(" step number: "+str(it)+"\n")
            F.write("*** MILP *** \n")


        ####################### MILP start ###########################
        
        if it==0:
            b_init_MILP = b_init
        else:
            b_init_MILP = algo.graph.root.state.belief

            
        MILP_algo = RAOStar_original(model, cc=cc, debugging=False, cc_type='o', fixed_horizon = horizon, \
                                random_node_selection=False)


        MILP_t = time.time()
            
        MILP_G = MILP_algo.get_complete_graph(b_init_MILP)

        if F!=None:
            F.write("  time before write: "+str(time.time() - MILP_t) + "\n\n")            

        MILP_algo.write_data(MILP_algo, MILP_G, it)
        MILP_algo.write_model(MILP_algo, MILP_G, it)

        if F!=None:
            F.write("  time: "+str(time.time() - MILP_t) + "\n\n")
            
        ####################### MILP end ###########################

        print(time.time() - MILP_t)

        # raise ValueError("hi")


        ####################### RAO* start ###########################
        
        RAO_algo = RAOStar_original(model, cc=cc, debugging=False, cc_type='o', fixed_horizon = horizon, \
                                    random_node_selection=False)


        RAO_t = time.time()

        RAO_algo.search(b_init_MILP,time_limit=60)

        F.write("*** RAO* *** \n")
        F.write("  value  : "+str(RAO_algo.graph.root.value) + "\n")
        F.write("  risk   : "+str(RAO_algo.graph.root.exec_risk) + "\n")
        F.write("  time   : "+str(time.time() - RAO_t) + "\n")
        if RAO_algo.graph.root.best_action != None:
            F.write("  action : "+ RAO_algo.graph.root.best_action.name + "\n\n")
        
        ####################### RAO* end ###########################




        
        ####################### Dual start ###########################

        if it==0:
            algo = RAOStar(model, cc=cc, debugging=False, cc_type='o', fixed_horizon = horizon, \
                           random_node_selection=False, lagrange_multiplier=alpha_zero)
        
        Dual_t = time.time()


        ##### phase 1
        # zero case
        algo.lagrange_multiplier = 2.5#alpha_zero
        if it>0:
            algo.rewire(algo.graph.root, algo.graph.root)


        k_shortest_algo = RAOStar_K_Shortest(1, algo, b_init, epsilon)
        anytime_solutions, best_k, k = k_shortest_algo.k_shortest_search(-float('inf'), float('inf'),0,F)
        
        f_plus = anytime_solutions[-1][0]
        g_plus = anytime_solutions[-1][1] - cc

        print("---------------------------")
        print(f_plus)
        print(g_plus)
        print(anytime_solutions)

        raise ValueError(22)

        # infinite case
        algo.lagrange_multiplier = alpha_infty
        algo.rewire(algo.graph.root, algo.graph.root)
        anytime_solutions, best_k, k = k_shortest_algo.k_shortest_search(-float('inf'), float('inf'),time.time()-Dual_t,F)

        f_minus = anytime_solutions[-1][0]
        g_minus = anytime_solutions[-1][1] - cc

        print("---------------------------")
        print(f_minus)
        print(g_minus)
        

        
        # phase 1 interation to compute alpha
        while True:

            # update alpha
            alpha = (f_minus - f_plus) / (g_plus - g_minus)
            L = f_plus + alpha*g_plus
            UB = float('inf')

           
            # evaluate L(u), f, g
            algo.lagrange_multiplier = alpha
            algo.rewire(algo.graph.root, algo.graph.root)
            anytime_solutions, best_k, k = k_shortest_algo.k_shortest_search(-float('inf'), float('inf'),time.time()-Dual_t,F)
            L_u = anytime_solutions[-1][0] + alpha * (anytime_solutions[-1][1] - cc)
            f = anytime_solutions[-1][0]
            g = anytime_solutions[-1][1] - cc


            print("---------------------------")
            print(f)
            print(g)


            # cases
            if abs(L_u - L)<0.1**10 and g < 0:
                LB = L_u
                UB = min(f, UB)
                break
            
            elif abs(L_u - L)<0.1**10 and g > 0:
                LB = L_u
                UB = f_minus
                break
            
            elif L_u < L and g > 0:
                f_plus = f
                g_plus = g

            elif L_u < L and g < 0:
                f_minus = f
                g_minus = g
                UB = min(UB, f)

            elif g==0:
                raise ValueError("opt solution found during phase 1. not implented for this case yet.")

            elif L_u > L :
                print(L_u)
                print(L)
                raise ValueError("impossible case. Something must be wrong")
                # LB = f_minus
                # break

        ## compute phase 1 time
        phase_1_time = time.time() - Dual_t

        print(alpha)
        raise ValueError(11)



        if LB>=UB:
            ## optimal solutiion found during phase 1
            if F!=None:
                F.write("*** DUAL *** \n")
                F.write("optimal solution found during phase 1!\n")
                F.write("optimal solution: \n")
                F.write("      value: "+str(algo.graph.root.value) + "\n")
                F.write("       risk: "+str(algo.graph.root.exec_risk) + "\n")
                F.write("     action: "+ algo.graph.root.best_action.name + "\n")
                F.write("  ph 1 time: "+str(phase_1_time) + "\n")
                F.write(" total time: "+str(phase_1_time) + "\n")
                F.write(" expanded nodes: "+str(len(algo.graph.nodes)-prev_num_nodes) + "\n")
                F.flush()

        else:
            algo.lagrange_multiplier = alpha
            algo.rewire(algo.graph.root, algo.graph.root)
            k_shortest_algo.K = max_k

            ## phase 2            
            anytime_solutions, best_k, k = k_shortest_algo.k_shortest_search(LB, UB, phase_1_time, F)
            elapsed_time = time.time() - Dual_t

            if F!=None:
                F.write("*** DUAL *** \n")
                F.write("optimal solution found after searching "+str(k)+"-th solutions!\n")
                F.write("optimal solution: \n")
                F.write("               value: "+str(anytime_solutions[-1][0]) + "\n")
                F.write("                risk: "+str(anytime_solutions[-1][1]) + "\n")
                F.write("              action: "+ anytime_solutions[-1][3] + "\n")
                F.write("           ph 1 time: "+str(phase_1_time) + "\n")
                F.write("  best solution time: "+str(anytime_solutions[-1][2]) + "\n")
                F.write("          total time: "+str(elapsed_time) + "\n")
                F.write("      expanded nodes: "+str(len(algo.graph.nodes)-prev_num_nodes) + "\n")         

                
        #################### Simulation for next step #####################

        root = algo.graph.root
        policy = algo.extract_policy()

        ## simulate next belief state
        ops = algo.graph.all_node_operators(algo.graph.root)
        for op in ops:
            if op.name == anytime_solutions[-1][3]:
                best_action = op
                break

        children = algo.graph.hyperedge_successors(root, best_action)

        probs = []
        for child in children:
            probs.append(child.probability)

        rand = random.random()
        total_p = 0
        for idx in range(len(probs)):
            total_p = total_p + probs[idx]
            if rand <= total_p:
                choice = children[idx]
                break



        ### code for testing wrong examples ###
        
        # selected = 0

        # for child in children:
        #     if child.state.belief == wrong_exs[it]:
        #         choice = child
        #         selected = 1
        #         break

        # if selected == 0:
        #     raise ValueError("no child")

        #######################################
        
        prev_root = algo.graph.root
        algo.graph.root = choice

      

        
        ## extend horizon by 1
        algo.fixed_horizon = algo.fixed_horizon + 1

        
        ## reset terminal nodes as non-terminal nodes
        for n in algo.graph.nodes:
            algo.graph.nodes[n].terminal = False


        ## delete unnecessary nodes
        all_action_operators = algo.graph.all_node_operators(prev_root)
        del_queue = []
        del_queue_descendants = []
        for act_idx, act in enumerate(all_action_operators):
            if act==prev_root.best_action:
                children = algo.graph.hyperedge_successors(prev_root, act)
                for child in children:
                    if child != choice:
                        del_queue.append(child)

            else:
                children = algo.graph.hyperedge_successors(prev_root, act)
                for child in children:
                    del_queue.append(child)

        for del_child in del_queue:
            temp_queue = [del_child]

            while len(temp_queue)!=0:
                temp_node = temp_queue.pop(0)

                descendants = algo.graph.all_descendants(temp_node)

                del_queue_descendants.extend(descendants)
                temp_queue.extend(descendants)

        for del_child in del_queue:
            if del_child.name in algo.graph.nodes:
                del algo.graph.nodes[del_child.name]

        for del_child in del_queue_descendants:
            if del_child.name in algo.graph.nodes:
                del algo.graph.nodes[del_child.name]


        ## reset value and risk weights of the nodes
        algo.reset_probs()

        prev_num_nodes = len(algo.graph.nodes)

        F.flush()

    F.close()

