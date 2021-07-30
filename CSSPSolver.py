#!/usr/bin/env python

import sys
import time
import math
from utils import *
from graph import Node, Graph
from LAOStar import LAOStar
from ILAOStar import ILAOStar

import copy
from heapq import *


import numpy as np

class CSSPSolver(object):

    def __init__(self, model, VI_epsilon=1e-50, VI_max_iter=100000, convergence_epsilon=1e-50, resolve_epsilon=1e-5, bounds=[]):

        self.model = model
        self.bounds = bounds
        self.resolve_epsilon = resolve_epsilon

        # self.algo = LAOStar(self.model,constrained=True,VI_epsilon=VI_epsilon, VI_max_iter=VI_max_iter, \
        #                     convergence_epsilon=convergence_epsilon,\
        #                     bounds=self.bounds,Lagrangian=True)
        
        self.algo = ILAOStar(self.model,constrained=True,VI_epsilon=VI_epsilon, convergence_epsilon=convergence_epsilon,\
                             bounds=self.bounds,alpha=[0.0],Lagrangian=True)
        self.graph = self.algo.graph


        #### for second stage (incremental_update)
        self.k_best_solution_set = []
        self.candidate_set = []
        self.current_best_policy = None

        self.candidate_idx = 0   ## this index is used for tie-breaking in heap queue.

        self.candidate_pruning = False

        self.t_start = time.time()
        self.anytime_solutions = []


    def solve(self, initial_alpha_set):

        self.find_dual(initial_alpha_set)
        # self.find_dual_multiple_bounds(initial_alpha_set)
        # self.anytime_update()


    def solve_sg(self, initial_alpha, h=0.1, rule='constant'):

        self.find_dual_sg(initial_alpha, h, rule)


    def solve_dual_multiple(self, initial_alpha):

        self.find_dual_multiple_bounds(initial_alpha)


    def solve_dual_line_search(self, initial_alpha):

        self.find_dual_line_search(initial_alpha)

 
    def find_dual(self, initial_alpha_set):

        start_time = time.time()

        ##### phase 1
        # zero case
        self.algo.alpha = [initial_alpha_set[0][0]]

        policy = self.algo.solve()
        value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9)
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))
        
        f_plus = value_1
        g_plus = value_2 - self.bounds[0]

        print("-"*50)
        print("time elapsed: "+str(time.time() - start_time))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f_plus))
        print("     g: "+str(g_plus))

        self.add_anytime_solution(f_plus,g_plus)

        # print("-------------------------------")
        # print(f_plus + self.algo.alpha[0]*g_plus)
        # print("-------------------------------")
        
        # infinite case
        policy = self.resolve_LAOStar([initial_alpha_set[0][1]])
        value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9)

        del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

        # self.algo = ILAOStar(self.model,constrained=True,bounds=self.bounds,alpha=[initial_alpha_set[0][1]],Lagrangian=True)
        # self.graph = self.algo.graph
        # self.algo.solve()

        f_minus = value_1
        g_minus = value_2 - self.bounds[0]

        self.add_anytime_solution(f_minus,g_minus)

        print("-"*50)
        print("time elapsed: "+str(time.time() - start_time))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f_minus))
        print("     g: "+str(g_minus))


        # print("-------------------------------")
        # print(f_minus + self.algo.alpha[0]*g_minus)
        # print("-------------------------------")
        
        # phase 1 interation to compute alpha
        while True:

            # update alpha
            alpha = (f_minus - f_plus) / (g_plus - g_minus)
            L = f_plus + alpha*g_plus
            UB = float('inf')
           
            # evaluate L(u), f, g
            policy = self.resolve_LAOStar([alpha])
            value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
            weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9)

            del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
            self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

            # self.algo = ILAOStar(self.model,constrained=True,bounds=self.bounds,alpha=[alpha],Lagrangian=True)
            # self.graph = self.algo.graph
            # self.algo.solve()

            L_u = value_1 + alpha*(value_2 - self.bounds[0])
            f = value_1
            g = value_2 - self.bounds[0]

            self.add_anytime_solution(f, g)

            print("-"*50)
            print("time elapsed: "+str(time.time() - start_time))
            print("nodes expanded: "+str(len(self.algo.graph.nodes)))
            print("     f: "+str(f))
            print("     g: "+str(g))
            print("     L: "+str(L_u))

            # print("-------------------------------")
            # print(L_u)
            # print("-------------------------------")
            
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



        if LB>=UB:
            ## optimal solutiion found during phase 1

            print("optimal solution found during phase 1!")
 

        else:

            print("dual optima with the following values:")
            print(" alpha:"+str(alpha))
            print("     L: "+str(L))
            print("     f: "+str(f))
            print("     g: "+str(g))
           
            print("total elapsed time: "+str(time.time()-start_time))
            print("total nodes expanded: "+str(len(self.algo.graph.nodes)))







    def find_dual_sg(self, initial_alpha, h=0.1, rule='constant'):

        alpha_1_list = []
        alpha_2_list = []
        weighted_value_list = []

        self.algo.value_num = 3

        self.algo.alpha = initial_alpha
        k = 1

        g = [-1,-1]

        prev_weighted_value = float('inf')


        policy = self.algo.solve()
        value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9)
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))
        
        f = value_1
        g[0] = value_2 - self.bounds[0]
        g[1] = value_3 - self.bounds[1]

        weighted_value = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1]

        self.add_anytime_solution(f,g)
        
        print("-"*50)
        print("time elapsed: "+str(time.time() - self.t_start))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f))
        print("     g: "+str(g))
        print("   w_v: "+str(weighted_value))
        print(" alpha: "+str(self.algo.alpha))

        alpha_1_list.append(self.algo.alpha[0])
        alpha_2_list.append(self.algo.alpha[1])
        weighted_value_list.append(weighted_value)
        
        
        while abs(prev_weighted_value - weighted_value) > 1e-5:
            prev_weighted_value = weighted_value

            if rule=='constant':
                step_size = h
            elif rule=='sqrt':
                step_size = h/math.sqrt(k)
            elif rule=='quotient':
                step_size = h/k
                
            self.algo.alpha[0] = max(0, self.algo.alpha[0] + g[0] * step_size)
            self.algo.alpha[1] = max(0, self.algo.alpha[1] + g[1] * step_size)


            policy = self.resolve_LAOStar()
            value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
            weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9)

            del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
            self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

            f = value_1
            g[0] = value_2 - self.bounds[0]
            g[1] = value_3 - self.bounds[1]

            weighted_value = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1]

            alpha_1_list.append(self.algo.alpha[0])
            alpha_2_list.append(self.algo.alpha[1])
            weighted_value_list.append(weighted_value)

            self.add_anytime_solution(f,g)

            print("-"*50)
            print("time elapsed: "+str(time.time() - self.t_start))
            print("nodes expanded: "+str(len(self.algo.graph.nodes)))
            print("     f: "+str(f))
            print("     g: "+str(g))
            print("   w_v: "+str(weighted_value))
            print(" alpha: "+str(self.algo.alpha))
            
            k += 1

            if k>2000:
                break

        print(alpha_1_list)
        print(alpha_2_list)
        print(weighted_value_list)





            



    def find_dual_multiple_bounds(self, initial_alpha_set):

        k = 0

        alpha_1_list = []
        alpha_2_list = []
        weighted_value_list = []

        self.algo.value_num = 3

        ###### TODO: functionize "solve_LAOStar with 'resolve' as an option"


        # overall zero case
        self.algo.alpha = [alpha_set[0] for alpha_set in initial_alpha_set]

        policy = self.algo.solve()
        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3

        f_plus = value_1
        g_plus = list()
        g_plus.append(value_2 - self.bounds[0])
        g_plus.append(value_3 - self.bounds[1])

        print("---------- zero case --------")


        ###### TODO: Need to check whether this solution is feasible, which is the case that we already found optima.


        # overall infinite case
        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]
        self.resolve_LAOStar()

        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3

        f_minus = value_1
        g_minus = list()
        g_minus.append(value_2 - self.bounds[0])
        g_minus.append(value_3 - self.bounds[1])

        print("---------- infinite case --------")

        ###### TODO: Need to check whether this solution is infeasible, which is the case that the problem is infeasible, or alpha_max is not large enough.


        
        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]

        L_new = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1]
        while True:

            
            L_prev = L_new
            
            for bound_idx in range(len(initial_alpha_set)):  # looping over each bound (coorindate)

                alpha_1_list.append(self.algo.alpha[0])
                alpha_2_list.append(self.algo.alpha[1])
                weighted_value_list.append(f_minus + g_minus[0]*self.algo.alpha[0] + g_minus[1]*self.algo.alpha[1])

                print("*"*20)

                # zero case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][0]
                self.resolve_LAOStar()



                value_1 = self.algo.graph.root.value_1
                value_2 = self.algo.graph.root.value_2
                value_3 = self.algo.graph.root.value_3

                f_plus = value_1
                g_plus = list()
                g_plus.append(value_2 - self.bounds[0])
                g_plus.append(value_3 - self.bounds[1])




                # infinite case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][1]
                self.resolve_LAOStar()

                value_1 = self.algo.graph.root.value_1
                value_2 = self.algo.graph.root.value_2
                value_3 = self.algo.graph.root.value_3

                f_minus = value_1
                g_minus = list()
                g_minus.append(value_2 - self.bounds[0])
                g_minus.append(value_3 - self.bounds[1])

                


                while True:

                    new_alpha_comp = (f_plus - f_minus)

                    for bound_idx_inner in range(len(initial_alpha_set)):
                        if bound_idx_inner==bound_idx:
                            continue

                        new_alpha_comp += self.algo.alpha[bound_idx_inner] * (g_plus[bound_idx_inner] - g_minus[bound_idx_inner])

                    new_alpha_comp = new_alpha_comp / (g_minus[bound_idx]- g_plus[bound_idx])

                    self.algo.alpha[bound_idx] = new_alpha_comp


                    L = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1]
                    UB = float('inf')


                    self.resolve_LAOStar()

                    value_1 = self.algo.graph.root.value_1
                    value_2 = self.algo.graph.root.value_2
                    value_3 = self.algo.graph.root.value_3

                    f = value_1
                    g = list()
                    g.append(value_2 - self.bounds[0])
                    g.append(value_3 - self.bounds[1])

                    L_u = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1]

                    

                    print("-"*50)
                    print("time elapsed: "+str(time.time() - self.t_start))
                    print("nodes expanded: "+str(len(self.algo.graph.nodes)))
                    print("     f: "+str(f))
                    print("     g: "+str(g))
                    print("     L: "+str(L_u))


                    # cases
                    if abs(L_u - L)<0.1**10 and g[bound_idx] < 0:
                        LB = L_u
                        UB = min(f, UB)
                        break

                    elif abs(L_u - L)<0.1**10 and g[bound_idx] > 0:
                        LB = L_u
                        UB = f_minus
                        break

                    elif L_u < L and g[bound_idx] > 0:
                        f_plus = f
                        g_plus = g

                    elif L_u < L and g[bound_idx] < 0:
                        f_minus = f
                        g_minus = g
                        UB = min(UB, f)

                    elif g[bound_idx]==0:
                        raise ValueError("opt solution found during phase 1. not implented for this case yet.")

                    elif L_u > L :
                        print(L_u)
                        print(L)
                        raise ValueError("impossible case. Something must be wrong")



                k += 1

                    
            # if k>10:
            #     print(alpha_1_list)
            #     print(alpha_2_list)
            #     print(weighted_value_list)
            #     break



                ## optimality check for this entire envelop    

            L_new = L_u

            if abs(L_new - L_prev) < 0.1**5:
                print(alpha_1_list)
                print(alpha_2_list)
                print(weighted_value_list)
                break

            
        print("optimal solution found during phase 1!")
        print("dual optima with the following values:")
        print(" alpha:"+str(self.algo.alpha))
        print("     L: "+str(L))
        print("     f: "+str(f))
        print("     g: "+str(g))

        print(alpha_1_list)
        print(alpha_2_list)
        print(weighted_value_list)





    def find_dual_multiple_bounds_generalized(self, initial_alpha_set):

        k = 0

        alpha_1_list = []
        alpha_2_list = []
        weighted_value_list = []

        self.algo.value_num = 7

        ###### TODO: functionize "solve_LAOStar with 'resolve' as an option"


        # overall zero case
        self.algo.alpha = [alpha_set[0] for alpha_set in initial_alpha_set]

        policy = self.algo.solve()
        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3
        value_4 = self.algo.graph.root.value_4
        value_5 = self.algo.graph.root.value_5
        value_6 = self.algo.graph.root.value_6
        value_7 = self.algo.graph.root.value_7

        f_plus = value_1
        g_plus = list()
        g_plus.append(value_2 - self.bounds[0])
        g_plus.append(value_3 - self.bounds[1])
        g_plus.append(value_4 - self.bounds[2])
        g_plus.append(value_5 - self.bounds[3])
        g_plus.append(value_6 - self.bounds[4])
        g_plus.append(value_7 - self.bounds[5])

        self.add_anytime_solution(f_plus, g_plus)

        print("---------- zero case --------")
        print("time elapsed: "+str(time.time() - self.t_start))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f_plus))
        print("     g: "+str(g_plus))


        ###### TODO: Need to check whether this solution is feasible, which is the case that we already found optima.


        # overall infinite case
        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]
        self.resolve_LAOStar()

        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3
        value_4 = self.algo.graph.root.value_4
        value_5 = self.algo.graph.root.value_5
        value_6 = self.algo.graph.root.value_6
        value_7 = self.algo.graph.root.value_7

        f_minus = value_1
        g_minus = list()
        g_minus.append(value_2 - self.bounds[0])
        g_minus.append(value_3 - self.bounds[1])
        g_minus.append(value_4 - self.bounds[2])
        g_minus.append(value_5 - self.bounds[3])
        g_minus.append(value_6 - self.bounds[4])
        g_minus.append(value_7 - self.bounds[5])

        self.add_anytime_solution(f_minus, g_minus)

        print("---------- infinite case --------")
        print("time elapsed: "+str(time.time() - self.t_start))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f_minus))
        print("     g: "+str(g_minus))


        ###### TODO: Need to check whether this solution is infeasible, which is the case that the problem is infeasible, or alpha_max is not large enough.


        
        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]

        L_new = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1] + self.algo.alpha[2]*g_plus[2] \
                       + self.algo.alpha[3]*g_plus[3] + self.algo.alpha[4]*g_plus[4] + self.algo.alpha[5]*g_plus[5]
        while True:

            for bound_idx in range(len(initial_alpha_set)):  # looping over each bound (coorindate)

                L_prev = L_new

                # zero case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][0]
                self.resolve_LAOStar()



                value_1 = self.algo.graph.root.value_1
                value_2 = self.algo.graph.root.value_2
                value_3 = self.algo.graph.root.value_3
                value_4 = self.algo.graph.root.value_4
                value_5 = self.algo.graph.root.value_5
                value_6 = self.algo.graph.root.value_6
                value_7 = self.algo.graph.root.value_7

                f_plus = value_1
                g_plus = list()
                g_plus.append(value_2 - self.bounds[0])
                g_plus.append(value_3 - self.bounds[1])
                g_plus.append(value_4 - self.bounds[2])
                g_plus.append(value_5 - self.bounds[3])
                g_plus.append(value_6 - self.bounds[4])
                g_plus.append(value_7 - self.bounds[5])

                
                self.add_anytime_solution(f_plus, g_plus)


                # print("-"*50)
                # print("time elapsed: "+str(time.time() - self.t_start))
                # print("nodes expanded: "+str(len(self.algo.graph.nodes)))
                # print("     f: "+str(f_plus))
                # print("     g: "+str(g_plus))
                # print(" alpha:"+str(self.algo.alpha))


                # infinite case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][1]
                self.resolve_LAOStar()

                value_1 = self.algo.graph.root.value_1
                value_2 = self.algo.graph.root.value_2
                value_3 = self.algo.graph.root.value_3
                value_4 = self.algo.graph.root.value_4
                value_5 = self.algo.graph.root.value_5
                value_6 = self.algo.graph.root.value_6
                value_7 = self.algo.graph.root.value_7

                f_minus = value_1
                g_minus = list()
                g_minus.append(value_2 - self.bounds[0])
                g_minus.append(value_3 - self.bounds[1])
                g_minus.append(value_4 - self.bounds[2])
                g_minus.append(value_5 - self.bounds[3])
                g_minus.append(value_6 - self.bounds[4])
                g_minus.append(value_7 - self.bounds[5])
                
                self.add_anytime_solution(f_minus, g_minus)


                # print("-"*50)
                # print("time elapsed: "+str(time.time() - self.t_start))
                # print("nodes expanded: "+str(len(self.algo.graph.nodes)))
                # print("     f: "+str(f_minus))
                # print("     g: "+str(g_minus))
                # print(" alpha:"+str(self.algo.alpha))
                


                while True:

                    if g_minus[bound_idx] == g_plus[bound_idx]:
                        f = f_minus
                        g = g_minus
                        L_u = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1] + self.algo.alpha[2]*g[2] + self.algo.alpha[3]*g[3] \
                            + self.algo.alpha[4]*g[4] + self.algo.alpha[5]*g[5]

                        break

                    new_alpha_comp = (f_plus - f_minus)

                    for bound_idx_inner in range(len(initial_alpha_set)):
                        if bound_idx_inner==bound_idx:
                            continue

                        new_alpha_comp += self.algo.alpha[bound_idx_inner] * (g_plus[bound_idx_inner] - g_minus[bound_idx_inner])

                    new_alpha_comp = new_alpha_comp / (g_minus[bound_idx]- g_plus[bound_idx])

                    self.algo.alpha[bound_idx] = new_alpha_comp


                    L = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1] + self.algo.alpha[2]*g_plus[2] + self.algo.alpha[3]*g_plus[3] \
                               + self.algo.alpha[4]*g_plus[4] + self.algo.alpha[5]*g_plus[5]
                    UB = float('inf')


                    self.resolve_LAOStar()

                    value_1 = self.algo.graph.root.value_1
                    value_2 = self.algo.graph.root.value_2
                    value_3 = self.algo.graph.root.value_3
                    value_4 = self.algo.graph.root.value_4
                    value_5 = self.algo.graph.root.value_5
                    value_6 = self.algo.graph.root.value_6
                    value_7 = self.algo.graph.root.value_7
                    

                    f = value_1
                    g = list()
                    g.append(value_2 - self.bounds[0])
                    g.append(value_3 - self.bounds[1])
                    g.append(value_4 - self.bounds[2])
                    g.append(value_5 - self.bounds[3])
                    g.append(value_6 - self.bounds[4])
                    g.append(value_7 - self.bounds[5])

                    self.add_anytime_solution(f, g)

                    L_u = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1] + self.algo.alpha[2]*g[2] + self.algo.alpha[3]*g[3] \
                            + self.algo.alpha[4]*g[4] + self.algo.alpha[5]*g[5]



                    # cases
                    if abs(L_u - L)<0.1**10 and g[bound_idx] < 0:
                        LB = L_u
                        UB = min(f, UB)
                        break

                    elif abs(L_u - L)<0.1**10 and g[bound_idx] > 0:
                        LB = L_u
                        UB = f_minus
                        break

                    elif L_u < L and g[bound_idx] > 0:
                        f_plus = f
                        g_plus = g

                    elif L_u < L and g[bound_idx] < 0:
                        f_minus = f
                        g_minus = g
                        UB = min(UB, f)

                    elif g[bound_idx]==0:
                        raise ValueError("opt solution found during phase 1. not implented for this case yet.")

                    elif L_u > L :
                        print(L_u)
                        print(L)
                        raise ValueError("impossible case. Something must be wrong")


                print("-"*50)
                print("time elapsed: "+str(time.time() - self.t_start))
                print("nodes expanded: "+str(len(self.algo.graph.nodes)))
                print("     f: "+str(f))
                print("     g: "+str(g))
                print("     L: "+str(L_u))


                k += 1



                ## optimality check for this entire envelop    

                L_new = L_u

                if abs(L_new - L_prev) < 0.1**5:
                    print("dual optima with the following values:")
                    print(" alpha:"+str(self.algo.alpha))
                    print("     L: "+str(L))
                    print("     f: "+str(f))
                    print("     g: "+str(g))

                    # return True

                    





    def find_dual_line_search(self, initial_alpha_set):

        k = 0

        alpha_1_list = []
        alpha_2_list = []
        weighted_value_list = []

        self.algo.value_num = 3



        # overall zero case
        self.algo.alpha = [alpha_set[0] for alpha_set in initial_alpha_set]

        policy = self.algo.solve()
        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3

        f = value_1
        g = list()
        g.append(value_2 - self.bounds[0])
        g.append(value_3 - self.bounds[1])

        self.add_anytime_solution(f, g)

        if g[0]<=0 and g[1]<=0:
            print("trivially feasible")
            return True
        

        # overall infinite case
        print("---------- infinite case --------")
        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]
        self.resolve_LAOStar()

        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3

        f = value_1
        g = list()
        g.append(value_2 - self.bounds[0])
        g.append(value_3 - self.bounds[1])

        self.add_anytime_solution(f, g)

        L_new = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1]

        while True:


            alpha_1_list.append(self.algo.alpha[0])
            alpha_2_list.append(self.algo.alpha[1])
            weighted_value_list.append(L_new)

            L_prev = L_new

            if k==0:
                direction = g
            else:
                direction = [-g_minus[1]+g_plus[1], -g_plus[0]+g_minus[0]]
            alpha_bound_1, alpha_bound_2 = self.compute_alpha_bounds(direction, initial_alpha_set)

            line_vector = [alpha_bound_2[0] - alpha_bound_1[0], alpha_bound_2[1] - alpha_bound_1[1]]

            #### TODO: if prev g_minus* and g_plus* in terms of this new line vector has diff sign (or at least one is zero), then dual opt is found.


            # if k==0 and alpha_bound_1 == [alpha_set[1] for alpha_set in initial_alpha_set]:
            #     f_1 = f
            #     g_1 = g

            # else:
            self.algo.alpha = alpha_bound_1
            self.resolve_LAOStar()

            value_1 = self.algo.graph.root.value_1
            value_2 = self.algo.graph.root.value_2
            value_3 = self.algo.graph.root.value_3

            f_1 = value_1
            g_1 = list()
            g_1.append(value_2 - self.bounds[0])
            g_1.append(value_3 - self.bounds[1])

            self.add_anytime_solution(f_1, g_1)

            gradient_1 = g_1[0]*line_vector[0] + g_1[1]*line_vector[1]

            if gradient_1==0:
                print("this case not implemented yet: this solution is optimal")


            # if k==0 and alpha_bound_2 == [alpha_set[1] for alpha_set in initial_alpha_set]:
            #     f_2 = f
            #     g_2 = g

            # else:
            self.algo.alpha = alpha_bound_2
            self.resolve_LAOStar()

            value_1 = self.algo.graph.root.value_1
            value_2 = self.algo.graph.root.value_2
            value_3 = self.algo.graph.root.value_3

            f_2 = value_1
            g_2 = list()
            g_2.append(value_2 - self.bounds[0])
            g_2.append(value_3 - self.bounds[1])

            self.add_anytime_solution(f_2, g_2)

            gradient_2 = g_2[0]*line_vector[0] + g_2[1]*line_vector[1]

            if gradient_2==0:
                print("this solution is optimal")

            if gradient_1 * gradient_2 > 0:
                print(self.anytime_solutions)
                # time.sleep(10000)
                # # L1 = f_1 + self.algo.alpha[0]*g_1[0] + self.algo.alpha[1]*g_1[1]
                # # L2 = f_2 + self.algo.alpha[0]*g_2[0] + self.algo.alpha[1]*g_2[1]

                # # if L1 > L2:
                # #     L_u = L1
                # # else:
                # #     L_u = L2

                # # k += 1
                # # L_new = L_u

                # # if abs(L_new - L_prev) < 1e-5:
                # #     print(alpha_1_list)
                # #     print(alpha_2_list)
                # #     print(weighted_value_list)
                # #     break

                # # continue


            if gradient_1 > 0:
                f_plus = f_1
                g_plus = g_1
                
                f_minus = f_2
                g_minus = g_2
                
            else:
                f_plus = f_2
                g_plus = g_2
                
                f_minus = f_1
                g_minus = g_1


            while True:

                self.algo.alpha = self.compute_new_alpha(f_minus, g_minus, f_plus, g_plus, line_vector)

                L = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1]
                UB = float('inf')

                self.resolve_LAOStar()

                value_1 = self.algo.graph.root.value_1
                value_2 = self.algo.graph.root.value_2
                value_3 = self.algo.graph.root.value_3

                f = value_1
                g = list()
                g.append(value_2 - self.bounds[0])
                g.append(value_3 - self.bounds[1])

                self.add_anytime_solution(f, g)

                gradient = g[0]*line_vector[0] + g[1]*line_vector[1]

                L_u = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1]

                print("-"*50)
                print("time elapsed: "+str(time.time() - self.t_start))
                print("nodes expanded: "+str(len(self.algo.graph.nodes)))
                print("     f: "+str(f))
                print("     g: "+str(g))
                print("     L: "+str(L_u))


                # cases
                if abs(L_u - L)<0.1**3 and gradient < 0:    ## this should be modified
                    f_minus = f
                    g_minus = g
                    LB = L_u
                    UB = min(f, UB)   ## this should be modified
                    break

                elif abs(L_u - L)<0.1**3 and gradient > 0:    ## this should be modified
                    f_plus = f
                    g_plus = g
                    LB = L_u
                    UB = f_minus   ## this should be modified
                    break

                elif L_u < L and gradient > 0:
                    f_plus = f
                    g_plus = g

                elif L_u < L and gradient < 0:
                    f_minus = f
                    g_minus = g
                    UB = min(UB, f)

                elif gradient==0:
                    raise ValueError("opt solution found during phase 1. not implented for this case yet.")

                elif L_u > L :
                    print(L_u)
                    print(L)
                    raise ValueError("impossible case. Something must be wrong")


            k += 1

            L_new = L_u

            if abs(L_new - L_prev) < 1e-5:
                print(alpha_1_list)
                print(alpha_2_list)
                print(weighted_value_list)
                break







    # def find_dual_line_search_generalized(self, initial_alpha_set):

    #     k = 0

    #     alpha_1_list = []
    #     alpha_2_list = []
    #     weighted_value_list = []

    #     self.algo.value_num = 7

    #     # overall infinite case
    #     print("---------- infinite case --------")
    #     self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]
    #     self.algo.solve()

    #     value_1 = self.algo.graph.root.value_1
    #     value_2 = self.algo.graph.root.value_2
    #     value_3 = self.algo.graph.root.value_3
    #     value_4 = self.algo.graph.root.value_4
    #     value_5 = self.algo.graph.root.value_5
    #     value_6 = self.algo.graph.root.value_6
    #     value_7 = self.algo.graph.root.value_7

    #     f = value_1
    #     g = list()
    #     g.append(value_2 - self.bounds[0])
    #     g.append(value_3 - self.bounds[1])
    #     g.append(value_4 - self.bounds[2])
    #     g.append(value_5 - self.bounds[3])
    #     g.append(value_6 - self.bounds[4])
    #     g.append(value_7 - self.bounds[5])


    #     L_new = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1]\
    #               + self.algo.alpha[2]*g[2] + self.algo.alpha[3]*g[3] + self.algo.alpha[4]*g[4] + self.algo.alpha[5]*g[5]

    #     while True:

    #         L_prev = L_new

    #         if k==0:
    #             direction = g
    #         else:
    #             direction = [-g_minus[1]+g_plus[1], -g_plus[0]+g_minus[0]]
    #         alpha_bound_1, alpha_bound_2 = self.compute_alpha_bounds(direction, initial_alpha_set)

    #         line_vector = [alpha_bound_2[0] - alpha_bound_1[0], alpha_bound_2[1] - alpha_bound_1[1]]

    #         #### TODO: if prev g_minus* and g_plus* in terms of this new line vector has diff sign (or at least one is zero), then dual opt is found.


    #         # if k==0 and alpha_bound_1 == [alpha_set[1] for alpha_set in initial_alpha_set]:
    #         #     f_1 = f
    #         #     g_1 = g

    #         # else:
    #         self.algo.alpha = alpha_bound_1
    #         self.resolve_LAOStar()

    #         value_1 = self.algo.graph.root.value_1
    #         value_2 = self.algo.graph.root.value_2
    #         value_3 = self.algo.graph.root.value_3
    #         value_4 = self.algo.graph.root.value_4
    #         value_5 = self.algo.graph.root.value_5
    #         value_6 = self.algo.graph.root.value_6
    #         value_7 = self.algo.graph.root.value_7

    #         f_1 = value_1
    #         g_1 = list()
    #         g_1.append(value_2 - self.bounds[0])
    #         g_1.append(value_3 - self.bounds[1])
    #         g_1.append(value_4 - self.bounds[2])
    #         g_1.append(value_5 - self.bounds[3])
    #         g_1.append(value_6 - self.bounds[4])
    #         g_1.append(value_7 - self.bounds[5])

    #         gradient_1 = g_1[0]*line_vector[0] + g_1[1]*line_vector[1]

    #         if gradient_1==0:
    #             print("this case not implemented yet: this solution is optimal")


    #         # if k==0 and alpha_bound_2 == [alpha_set[1] for alpha_set in initial_alpha_set]:
    #         #     f_2 = f
    #         #     g_2 = g

    #         # else:
    #         self.algo.alpha = alpha_bound_2
    #         self.resolve_LAOStar()

    #         value_1 = self.algo.graph.root.value_1
    #         value_2 = self.algo.graph.root.value_2
    #         value_3 = self.algo.graph.root.value_3
    #         value_4 = self.algo.graph.root.value_4
    #         value_5 = self.algo.graph.root.value_5
    #         value_6 = self.algo.graph.root.value_6
    #         value_7 = self.algo.graph.root.value_7

    #         f_2 = value_1
    #         g_2 = list()
    #         g_2.append(value_2 - self.bounds[0])
    #         g_2.append(value_3 - self.bounds[1])
    #         g_2.append(value_4 - self.bounds[2])
    #         g_2.append(value_5 - self.bounds[3])
    #         g_2.append(value_6 - self.bounds[4])
    #         g_2.append(value_7 - self.bounds[5])

    #         gradient_2 = g_2[0]*line_vector[0] + g_2[1]*line_vector[1]

    #         if gradient_2==0:
    #             print("this solution is optimal")

    #         if gradient_1 * gradient_2 > 0:
    #             print("this case not implemented yet: solution with higher L is optimal")


    #         if gradient_1 > 0:
    #             f_plus = f_1
    #             g_plus = g_1
                
    #             f_minus = f_2
    #             g_minus = g_2
                
    #         else:
    #             f_plus = f_2
    #             g_plus = g_2
                
    #             f_minus = f_1
    #             g_minus = g_1


    #         while True:

    #             self.algo.alpha = self.compute_new_alpha(f_minus, g_minus, f_plus, g_plus, line_vector)

    #             L = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1]
    #             UB = float('inf')

    #             self.resolve_LAOStar()

    #             value_1 = self.algo.graph.root.value_1
    #             value_2 = self.algo.graph.root.value_2
    #             value_3 = self.algo.graph.root.value_3
    #             value_4 = self.algo.graph.root.value_4
    #             value_5 = self.algo.graph.root.value_5
    #             value_6 = self.algo.graph.root.value_6
    #             value_7 = self.algo.graph.root.value_7

    #             f = value_1
    #             g = list()
    #             g.append(value_2 - self.bounds[0])
    #             g.append(value_3 - self.bounds[1])
    #             g.append(value_4 - self.bounds[2])
    #             g.append(value_5 - self.bounds[3])
    #             g.append(value_6 - self.bounds[4])
    #             g.append(value_7 - self.bounds[5])

    #             gradient = g[0]*line_vector[0] + g[1]*line_vector[1]

    #             L_u = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1]

    #             print("-"*50)
    #             print("time elapsed: "+str(time.time() - self.t_start))
    #             print("nodes expanded: "+str(len(self.algo.graph.nodes)))
    #             print("     f: "+str(f))
    #             print("     g: "+str(g))
    #             print("     L: "+str(L_u))


    #             # cases
    #             if abs(L_u - L)<0.1**3 and gradient < 0:    ## this should be modified
    #                 f_minus = f
    #                 g_minus = g
    #                 LB = L_u
    #                 UB = min(f, UB)   ## this should be modified
    #                 break

    #             elif abs(L_u - L)<0.1**3 and gradient > 0:    ## this should be modified
    #                 f_plus = f
    #                 g_plus = g
    #                 LB = L_u
    #                 UB = f_minus   ## this should be modified
    #                 break

    #             elif L_u < L and gradient > 0:
    #                 f_plus = f
    #                 g_plus = g

    #             elif L_u < L and gradient < 0:
    #                 f_minus = f
    #                 g_minus = g
    #                 UB = min(UB, f)

    #             elif gradient==0:
    #                 raise ValueError("opt solution found during phase 1. not implented for this case yet.")

    #             elif L_u > L :
    #                 print(L_u)
    #                 print(L)
    #                 raise ValueError("impossible case. Something must be wrong")


    #         k += 1

    #         L_new = L_u

    #         if abs(L_new - L_prev) < 1e-5:
    #             print(alpha_1_list)
    #             print(alpha_2_list)
    #             print(weighted_value_list)
    #             break
            




            

                    

    def compute_alpha_bounds(self, direction, initial_alpha_set):

        if direction[0]==0:
            return [[self.algo.alpha[0], initial_alpha_set[1][0]],[self.algo.alpha[0], initial_alpha_set[1][1]]]

        if direction[1]==0:
            return [[initial_alpha_set[0][0], self.algo.alpha[1]],[initial_alpha_set[0][1], self.algo.alpha[1]]]
        
        alpha_bounds = []

        ## alpha_1 = 0 case
        alpha_1 = initial_alpha_set[0][0]
        t = (alpha_1 - self.algo.alpha[0]) / direction[0]
        alpha_2 = self.algo.alpha[1] + t*direction[1]

        if alpha_2 >= initial_alpha_set[1][0] and alpha_2 <= initial_alpha_set[1][1]:
            alpha_bounds.append([alpha_1,alpha_2])

        ## alpha_1 = infty case
        alpha_1 = initial_alpha_set[0][1]
        t = (alpha_1 - self.algo.alpha[0]) / direction[0]
        alpha_2 = self.algo.alpha[1] + t*direction[1]

        is_exist = False

        if alpha_2 >= initial_alpha_set[1][0] and alpha_2 <= initial_alpha_set[1][1]:
            for alpha in alpha_bounds:
                if alpha[0]==alpha_1 and alpha[1]==alpha_2:
                    is_exist = True
                    break
            if is_exist==False:
                alpha_bounds.append([alpha_1,alpha_2])


        alpha_bound_2 = []

        ## alpha_2 = 0 case
        alpha_2 = initial_alpha_set[1][0]
        t = (alpha_2 - self.algo.alpha[1]) / direction[1]
        alpha_1 = self.algo.alpha[0] + t*direction[0]

        is_exist = False

        if alpha_1 >= initial_alpha_set[0][0] and alpha_1 <= initial_alpha_set[0][1]:
            for alpha in alpha_bounds:
                if alpha[0]==alpha_1 and alpha[1]==alpha_2:
                    is_exist = True
                    break
            if is_exist==False:
                alpha_bounds.append([alpha_1,alpha_2])

        ## alpha_2 = infty case
        alpha_2 = initial_alpha_set[1][1]
        t = (alpha_2 - self.algo.alpha[1]) / direction[1]
        alpha_1 = self.algo.alpha[0] + t*direction[0]

        is_exist = False

        if alpha_1 >= initial_alpha_set[0][0] and alpha_1 <= initial_alpha_set[0][1]:
            for alpha in alpha_bounds:
                if alpha[0]==alpha_1 and alpha[1]==alpha_2:
                    is_exist = True
                    break
            if is_exist==False:
                alpha_bounds.append([alpha_1,alpha_2])

        if len(alpha_bounds)!=2:
            print(alpha_bounds)
            raise ValueError("new alpha bounds computed wrong.")


        return alpha_bounds[0], alpha_bounds[1]




    def compute_new_alpha(self, f_minus, g_minus, f_plus, g_plus, line_vector):
        t = ( (f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1]) \
              - (f_minus + self.algo.alpha[0]*g_minus[0] + self.algo.alpha[1]*g_minus[1]) ) \
            / ( (line_vector[0]*g_minus[0] + line_vector[1]*g_minus[1]) \
                - (line_vector[0]*g_plus[0] + line_vector[1]*g_plus[1]))

        new_alpha = [self.algo.alpha[0] + line_vector[0]*t, self.algo.alpha[1] + line_vector[1]*t]

        return new_alpha














        
        


    def resolve_LAOStar(self, new_alpha=None,epsilon=1e-10):

        if new_alpha != None:
            self.algo.alpha = new_alpha
            
        nodes = list(self.algo.graph.nodes.values())
        self.algo.value_iteration(nodes,epsilon=self.resolve_epsilon)
        self.algo.update_fringe()
        return self.algo.solve()






    def incremental_update(self, num_sol):

        self.algo.incremental = True

        current_best_graph = self.copy_best_graph(self.algo.graph)
        current_best_policy = self.algo.extract_policy()

        self.algo.policy_evaluation(current_best_policy, epsilon=1e-100)

        for k in range(num_sol-1):

            # t = 0

            candidate_generating_states = self.prune_candidates(current_best_policy)

            for state,head,blocked_action_set in candidate_generating_states:

                # t += 1

                node = self.algo.graph.nodes[state]
                new_candidate = self.find_candidate(node,head,blocked_action_set)

                if new_candidate:
                    if self.candidate_exists(new_candidate):
                        self.return_to_best_graph(self.algo.graph, current_best_graph)  ## returning to the previous graph.
                        continue
                    else:
                        heappush(self.candidate_set, new_candidate)
                else:
                    continue

                self.return_to_best_graph(self.algo.graph, current_best_graph)  ## returning to the previous graph.

            current_best_graph, current_best_policy = self.find_next_best()
            self.return_to_best_graph(self.algo.graph, current_best_graph)     
            
            # print(t)
            # print(len(self.anytime_solutions))
            # print(self.anytime_solutions)
            # time.sleep(1000)


    def find_candidate(self, node, head, blocked_action_set):
            
        self.algo.head = head
        self.algo.blocked_action_set = blocked_action_set
        self.algo.tweaking_node = node

        ## if all actions are blocked, then no candidate is generated. 
        if len(self.algo.blocked_action_set)==len(self.model.actions(node.state)):
            return False

        
        self.algo.fringe = set([node])
        policy = self.algo.solve()

        value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9)

        self.add_anytime_solution(value_1, value_2 - self.bounds[0])

        self.candidate_idx += 1

        new_best_graph = self.copy_best_graph(self.algo.graph)

        return (weighted_value, self.candidate_idx, (value_1,value_2,value_3), new_best_graph, policy)
        # return (weighted_value, self.candidate_idx, (value_1,value_2,value_3), self.copy_graph(self.algo.graph), policy)



    def get_blocked_action_set(self,node,head):

        blocked_action_set = set()

        ## TODO: current policy need not be inspected in this way. Can be more easily done. Fix it for better efficiency later.
        for solution in self.k_best_solution_set:
            
            prev_policy = solution[2]

            if self.is_head_eq(head, prev_policy):
                blocked_action_set.add(prev_policy[node.state])

        return blocked_action_set




    def get_head(self, node):

        queue = set([self.algo.graph.root])
        head = set()

        while queue:

            popped_node = queue.pop()

            if popped_node in head:
                continue

            if popped_node == node:
                continue

            if popped_node.terminal==True:
                continue

            head.add(popped_node)
            
            if popped_node.best_action!=None:
                children = popped_node.children[popped_node.best_action]

                for child,child_prob in children:
                    queue.add(child)

            else:
                raise ValueError("Policy seems have unexpanded node.")                    

        return head    
    

    
    def get_ancestors(self, node):
        Z = set()

        queue = set([node])

        while queue:
            node = queue.pop()

            if node not in Z:
                
                Z.add(node)
                parents = node.best_parents_set

                queue = queue.union(parents)

        return Z

    

    def is_head_eq(self,head,policy):

        for node in head:

            if node.terminal==True:
                continue
            
            if node.state not in policy:
                return False

            else:
                if node.best_action == policy[node.state]:
                    continue
                else:
                    return False

        return True
    

    def candidate_exists(self,new_candidate):

        for candidate in self.candidate_set:

            # if abs(candidate[0] - new_candidate[0]) > 1e-5:
            #     continue

            # else:
            if self.is_policy_eq(candidate[4], new_candidate[4]):
                return True


        return False


    def is_policy_eq(self, policy_1, policy_2):

        if len(policy_1)!=len(policy_2):
            return False

        else:

            for state, action in policy_1.items():

                if state not in policy_2:
                    return False
                else:
                    if action != policy_2[state]:
                        return False

        return True

        
    def find_next_best(self):


        # print([cand[0] for cand in self.candidate_set])
        
        next_best_candidate = heappop(self.candidate_set)

        # print(next_best_candidate[0])
        # self.model.print_policy(next_best_candidate[3])

        current_best_weighted_value = next_best_candidate[0]
        current_best_values = next_best_candidate[2]
        current_best_graph = next_best_candidate[3]
        current_best_policy = next_best_candidate[4]

        self.k_best_solution_set.append((current_best_weighted_value, current_best_values, current_best_policy))

        return current_best_graph, current_best_policy




    def copy_graph(self,graph):

        new_graph = Graph(name='G')

        for state, node in graph.nodes.items():

            new_graph.add_node(state, node.value_1, node.value_2, node.value_3, node.best_action, node.terminal)

        for state, node in graph.nodes.items():
            for action, children in node.children.items():
                new_graph.nodes[state].children[action] = []
                for child, child_prob in children:
                    new_graph.nodes[state].children[action].append([new_graph.nodes[child.state], child_prob])

            for parents_node in node.best_parents_set:
                new_graph.nodes[state].best_parents_set.add(new_graph.nodes[parents_node.state])
                
        new_graph.root = new_graph.nodes[self.model.init_state]

        return new_graph




    def copy_best_graph(self, graph):

        copied_best_graph = dict()

        queue = set([self.graph.root])

        while queue:

            node = queue.pop()

            if node.state in copied_best_graph:
                continue

            else:
                if node.best_action!=None:
                    copied_best_graph[node.state] = (node.best_action, node.value_1, node.value_2, node.value_3, node.children, node.best_parents_set)
                    children = node.children[node.best_action]

                    for child,child_prob in children:
                        queue.add(child)

                elif node.terminal==True:
                    copied_best_graph[node.state] = (node.best_action, node.value_1, node.value_2, node.value_3, node.children, node.best_parents_set)

                else:
                    raise ValueError("Best partial graph has non-expanded fringe node.")

        return copied_best_graph


    
    def return_to_best_graph(self, graph, copied_best_graph):

        for state, contents in copied_best_graph.items():

            node = graph.nodes[state]
            
            node.best_action = contents[0]
            node.value_1 = contents[1]
            node.value_2 = contents[2]
            node.value_3 = contents[3]
            node.children = contents[4]
            node.best_parents_set = contents[5]






    def prune_candidates(self, current_best_policy):

        
        if self.candidate_pruning==False:
            candidate_generating_states = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':

                    node = self.algo.graph.nodes[state]
                    head = self.get_head(node)
                    blocked_action_set = self.get_blocked_action_set(node,head)
                    
                    candidate_generating_states.append((state,head,blocked_action_set))

            return candidate_generating_states
        
        
        elif (self.algo.graph.root.value_2 - self.bounds[0]) <= 0:

            epsilon = 0.01
            initial_epsilon = 1e-4

            policy_value = self.algo.graph.root.value_1

            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)


            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)
            N = np.linalg.inv(I - Q)
            N_vector = N[root_idx]
            R = np.ones(num_states)    ## TODO: need to be changed. Now it assumes unit cost.

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]

            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            # V = np.dot(N_new, R)
            # new_value = V[root_idx]

            new_value = np.dot(N_new[root_idx,:], R)


            epsilon -= (policy_value - new_value) / policy_value

            prev_value = new_value

            if epsilon < 0:
                raise ValueError("initial pruning was too aggressive!")
            else:
                N = N_new


            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    u = np.zeros(num_states)
                    u[idx] = 1
                    v = Q[idx,:]
                    N_new = self.SM_update(N,u,v)
                    R[idx] = 0
                    # V = np.dot(N_new, R)
                    # new_value = V[root_idx]

                    new_value = np.dot(N_new[root_idx,:], R)


                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            print(prev_value)
                            print(new_value)
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) / policy_value < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value) / policy_value
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = 1
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states

        

        elif (self.algo.graph.root.value_2 - self.bounds[0]) > 0:

            policy_value = self.algo.graph.root.value_2
            
            epsilon = policy_value - self.bounds[0]
            initial_epsilon = 1e-4

            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)


            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)
            N = np.linalg.inv(I - Q)
            N_vector = N[root_idx]

            R = self.compute_R(state_list)

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]

            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            new_value = np.dot(N_new[root_idx,:], R)


            epsilon -= (policy_value - new_value)

            prev_value = new_value

            if epsilon < 0:
                raise ValueError("initial pruning was too aggressive!")
            else:
                N = N_new


            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    u = np.zeros(num_states)
                    u[idx] = 1
                    v = Q[idx,:]
                    N_new = self.SM_update(N,u,v)
                    R_prev = R[idx]
                    R[idx] = 0
                    # V = np.dot(N_new, R)
                    # new_value = V[root_idx]

                    new_value = np.dot(N_new[root_idx,:], R)

                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            print(prev_value)
                            print(new_value)
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value)
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = R_prev
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states





    def prune_candidates_2(self, current_best_policy):

        
        if self.candidate_pruning==False:
            candidate_generating_states = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':

                    node = self.algo.graph.nodes[state]
                    head = self.get_head(node)
                    blocked_action_set = self.get_blocked_action_set(node,head)
                    
                    candidate_generating_states.append((state,head,blocked_action_set))

            return candidate_generating_states
        
        
        elif (self.algo.graph.root.value_2 - self.bounds[0]) <= 0:

            epsilon = 0.01
            initial_epsilon = 1e-5
            num_per_prune = 50

            policy_value = self.algo.graph.root.value_1

            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)


            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)
            N = np.linalg.inv(I - Q)
            N_vector = N[root_idx]
            R = np.ones(num_states)    ## TODO: need to be changed. Now it assumes unit cost.

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]

            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            # V = np.dot(N_new, R)
            # new_value = V[root_idx]

            new_value = np.dot(N_new[root_idx,:], R)


            epsilon -= (policy_value - new_value) / policy_value

            prev_value = new_value

            if epsilon < 0:
                raise ValueError("initial pruning was too aggressive!")
            else:
                N = N_new


            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    u = np.zeros(num_states)
                    u[idx] = 1
                    v = Q[idx,:]
                    N_new = self.SM_update(N,u,v)
                    R[idx] = 0
                    # V = np.dot(N_new, R)
                    # new_value = V[root_idx]

                    new_value = np.dot(N_new[root_idx,:], R)


                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            print(prev_value)
                            print(new_value)
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) / policy_value < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value) / policy_value
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = 1
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states

        

        elif (self.algo.graph.root.value_2 - self.bounds[0]) > 0:

            policy_value = self.algo.graph.root.value_2
            
            epsilon = policy_value - self.bounds[0]
            initial_epsilon = 1e-5

            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)


            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)
            N = np.linalg.inv(I - Q)
            N_vector = N[root_idx]

            R = self.compute_R(state_list)

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]

            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            print(len(sorted_states))
            print(len(initial_pruned_states))

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            t = time.time()
            N_new = np.linalg.inv(I - Q)
            new_value = np.dot(N_new[root_idx,:], R)
            print(time.time() - t)


            t = time.time()
            V_ = np.linalg.solve(I-Q, R)
            new_value_2 = V_[root_idx]
            print(time.time() - t)
            
            print(new_value)
            print(new_value_2)
            time.sleep(10000)

            epsilon -= (policy_value - new_value)

            prev_value = new_value

            if epsilon < 0:
                raise ValueError("initial pruning was too aggressive!")
            else:
                N = N_new


            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    u = np.zeros(num_states)
                    u[idx] = 1
                    v = Q[idx,:]
                    N_new = self.SM_update(N,u,v)
                    R_prev = R[idx]
                    R[idx] = 0
                    # V = np.dot(N_new, R)
                    # new_value = V[root_idx]

                    new_value = np.dot(N_new[root_idx,:], R)

                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            print(prev_value)
                            print(new_value)
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value)
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = R_prev
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states
        




        

    def compute_transition_matrix(self, state_list):

        num_states = len(state_list)

        Q = np.empty((num_states, num_states))
        root_idx = None

        i = 0
        state_idx_dict = dict()
        for state in state_list:
            state_idx_dict[state] = i

            if state==self.model.init_state:
                root_idx = state_list.index(state)

            Q_vector = np.zeros(num_states)

            node = self.algo.graph.nodes[state]
            children = node.children[node.best_action]

            for child, child_prob in children:
                if child.terminal != True:
                    idx = state_list.index(child.state)
                    if Q_vector[idx] > 0:
                        Q_vector[idx] += child_prob
                    else:
                        Q_vector[idx] = child_prob

            Q[i,:] = Q_vector

            i += 1

        return Q, root_idx, state_idx_dict
   



    def SM_update(self,B, u, v):
        return B - np.outer(B @ u, v @ B) / (1 + v.T @ B @ u)





    def compute_R(self,state_list):
        
        R = np.zeros(len(state_list))

        i = 0
        for state in state_list:
            cost1, cost2 = self.model.cost(state, None)
            R[i] = cost2
            i += 1

        return R


















            






    def prune_candidates_old(self, current_best_policy):

        if self.candidate_pruning==False or (self.algo.graph.root.value_2 - self.bounds[0]) > 0:
            candidate_generating_states = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':

                    node = self.algo.graph.nodes[state]
                    head = self.get_head(node)
                    blocked_action_set = self.get_blocked_action_set(node,head)
                    
                    candidate_generating_states.append((state,head,blocked_action_set))

            return candidate_generating_states
        
        
        else:

            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)

            L = self.compute_likelihoods(state_list)

            
            weighted_value_diff_list = []
            blocked_action_set_list = []
            head_list = []
            

            ## pruning when current best solution is feasible soluiton. 
            if (self.algo.graph.root.value_2 - self.bounds[0]) <= 0:

                policy_value = self.algo.graph.root.value_1
                
                for state in state_list:
                    node = self.algo.graph.nodes[state]
                    head = self.get_head(node)
                    blocked_action_set = self.get_blocked_action_set(node,head)

                    blocked_action_set_list.append(blocked_action_set)
                    head_list.append(head)

                    prev_value = node.value_1
                    possible_actions = self.model.actions(state)
                    value_diff_list = []

                    for action in possible_actions:
                        if action in blocked_action_set:
                            continue

                        new_value_1, new_value_2 = self.model.cost(state,action)

                        children = node.children[action]
                        for child, child_prob in children:
                            heuristic_1, heuristic_2 = self.model.heuristic(child.state)
                            new_value_1 += child_prob * heuristic_1

                        value_diff_list.append(prev_value - new_value_1)

                    max_value_diff = max(value_diff_list)

                    idx = state_list.index(state)
                    weighted_value_diff = L[idx]*max_value_diff

                    weighted_value_diff_list.append(weighted_value_diff / policy_value)

                sorted_states = [(state,head,blocked_action_set) for weighted_value_diff, state, head, blocked_action_set in \
                                 sorted(zip(weighted_value_diff_list, state_list, head_list, blocked_action_set_list))]

                weighted_value_diff_list.sort()


                prob = 0
                k = 0
                for i in weighted_value_diff_list:
                    prob += i
                    if prob >= 0.001:
                        break
                    k += 1

                candidate_generating_states = sorted_states[k:]

                return candidate_generating_states


            
            # ## pruning when current best solution is infeasible soluiton. 
            # elif (self.algo.graph.root.value_2 - self.bounds[0]) > 0:

            #     policy_value = self.algo.graph.root.value_2
                
            #     for state in state_list:
            #         node = self.algo.graph.nodes[state]
            #         head = self.get_head(node)
            #         blocked_action_set = self.get_blocked_action_set(node,head)

            #         blocked_action_set_list.append(blocked_action_set)
            #         head_list.append(head)

            #         prev_value = node.value_2
            #         possible_actions = self.model.actions(state)
            #         value_diff_list = []

            #         for action in possible_actions:
            #             if action in blocked_action_set:
            #                 continue

            #             new_value_1, new_value_2 = self.model.cost(state,action)

            #             children = node.children[action]
            #             for child, child_prob in children:
            #                 heuristic_1, heuristic_2 = self.model.heuristic(child.state)
            #                 new_value_2 += child_prob * heuristic_2

            #             value_diff_list.append(prev_value - new_value_2)

            #         max_value_diff = max(value_diff_list)

            #         idx = state_list.index(state)
            #         weighted_value_diff = L[idx]*max_value_diff

            #         weighted_value_diff_list.append(weighted_value_diff)

            #     sorted_states = [(state,head,blocked_action_set) for weighted_value_diff, state, head, blocked_action_set in \
            #                      sorted(zip(weighted_value_diff_list, state_list, head_list, blocked_action_set_list))]

            #     weighted_value_diff_list.sort()

            #     print(weighted_value_diff_list)
            #     time.sleep(1000)

            #     feasible_gap = policy_value - self.bounds[0]

            #     prob = 0
            #     k = 0
            #     for i in weighted_value_diff_list:
            #         prob += i
            #         if prob >= feasible_gap:
            #             break
            #         k += 1

            #     candidate_generating_states = sorted_states[k:]

            #     return candidate_generating_states


            

    def compute_likelihoods(self, state_list):

        num_states = len(state_list)
        
        Q = np.empty((num_states, num_states))
        root_idx = None

        i = 0
        for state in state_list:

            if state==self.model.init_state:
                root_idx = state_list.index(state)

            Q_vector = np.zeros(num_states)

            node = self.algo.graph.nodes[state]
            children = node.children[node.best_action]

            for child, child_prob in children:
                if child.terminal != True:
                    idx = state_list.index(child.state)
                    Q_vector[idx] = child_prob

            Q[i,:] = Q_vector
            
            i += 1


        t = time.time()
        I = np.identity(num_states)
        o = np.zeros(num_states)
        o[root_idx] = 1
        L = np.linalg.solve(np.transpose(I-Q), o)
        L = L / L[root_idx]

        return L  
    



    def add_anytime_solution(self, f, g):

        if type(g)==list:
            feasible=True
            for g_val in g:
                if g_val>0:
                    feasible=False
                    break
                
            if feasible==True:
                if len(self.anytime_solutions)==0:
                    self.anytime_solutions.append((f, time.time() - self.t_start))

                else:

                    if self.anytime_solutions[-1][0] > f:
                        self.anytime_solutions.append((f, time.time() - self.t_start))

        else:
            if g<0:
                if len(self.anytime_solutions)==0:
                    self.anytime_solutions.append((f, time.time() - self.t_start))

                else:

                    if self.anytime_solutions[-1][0] > f:
                        self.anytime_solutions.append((f, time.time() - self.t_start))


















###############################################################################################



    # def prune_candidates(self, current_best_policy):

    #     if self.candidate_pruning==False:
    #         candidate_generating_states = []
    #         for state, action in current_best_policy.items():
    #             if action != 'Terminal':

    #                 node = self.algo.graph.nodes[state]
    #                 head = self.get_head(node)
    #                 blocked_action_set = self.get_blocked_action_set(node,head)
                    
    #                 candidate_generating_states.append((state,head,blocked_action_set))

    #         return candidate_generating_states
        
        
    #     else:

    #         state_list = []
    #         for state, action in current_best_policy.items():
    #             if action != 'Terminal':
    #                 state_list.append(state)

    #         L = self.compute_likelihoods(state_list)

            
    #         weighted_value_diff_list = []
    #         blocked_action_set_list = []
    #         head_list = []
            

    #         ## pruning when current best solution is feasible soluiton. 
    #         if (self.algo.graph.root.value_2 - self.bounds[0]) <= 0:

    #             policy_value = self.algo.graph.root.value_1
                
    #             for state in state_list:
    #                 node = self.algo.graph.nodes[state]
    #                 head = self.get_head(node)
    #                 blocked_action_set = self.get_blocked_action_set(node,head)

    #                 blocked_action_set_list.append(blocked_action_set)
    #                 head_list.append(head)

    #                 prev_value = node.value_1
    #                 possible_actions = self.model.actions(state)
    #                 value_diff_list = []

    #                 for action in possible_actions:
    #                     if action in blocked_action_set:
    #                         continue

    #                     new_value_1, new_value_2 = self.model.cost(state,action)

    #                     children = node.children[action]
    #                     for child, child_prob in children:
    #                         heuristic_1, heuristic_2 = self.model.heuristic(child.state)
    #                         new_value_1 += child_prob * heuristic_1

    #                     value_diff_list.append(prev_value - new_value_1)

    #                 max_value_diff = max(value_diff_list)

    #                 idx = state_list.index(state)
    #                 weighted_value_diff = L[idx]*max_value_diff

    #                 weighted_value_diff_list.append(weighted_value_diff / policy_value)

    #             sorted_states = [(state,head,blocked_action_set) for weighted_value_diff, state, head, blocked_action_set in \
    #                              sorted(zip(weighted_value_diff_list, state_list, head_list, blocked_action_set_list))]

    #             weighted_value_diff_list.sort()


    #             prob = 0
    #             k = 0
    #             for i in weighted_value_diff_list:
    #                 prob += i
    #                 if prob >= 0.001:
    #                     break
    #                 k += 1

    #             candidate_generating_states = sorted_states[k:]

    #             return candidate_generating_states


            
    #         ## pruning when current best solution is infeasible soluiton. 
    #         elif (self.algo.graph.root.value_2 - self.bounds[0]) > 0:

    #             policy_value = self.algo.graph.root.value_2

    #             sorted_state_list = [state for prob, state in sorted(zip(L, state_list))]

    #             pruned = []

    #             weighted_value_diff = 0

    #             feasible_gap = policy_value - self.bounds[0]

    #             k=0

    #             for state in sorted_state_list:
    #                 node = self.algo.graph.nodes[state]
    #                 head = self.get_head(node)
    #                 blocked_action_set = self.get_blocked_action_set(node,head)

    #                 blocked_action_set_list.append(blocked_action_set)
    #                 head_list.append(head)

    #                 prev_value = node.value_2
    #                 possible_actions = self.model.actions(state)
    #                 value_diff_list = []

    #                 for action in possible_actions:
    #                     if action in blocked_action_set:
    #                         continue

    #                     new_value_1, new_value_2 = self.model.cost(state,action)

    #                     children = node.children[action]
    #                     for child, child_prob in children:
    #                         heuristic_1, heuristic_2 = self.model.heuristic(child.state)
    #                         new_value_2 += child_prob * heuristic_2

    #                     value_diff_list.append(prev_value - new_value_2)

    #                 max_value_diff = max(value_diff_list)

    #                 idx = state_list.index(state)

    #                 pruned.append(idx)

    #                 L2 = self.compute_likelihoods_2(state_list, pruned)

    #                 weighted_value_diff += L2[idx]*max_value_diff

    #                 if weighted_value_diff >= feasible_gap:
    #                     break

    #                 k += 1


    #             sorted_state_pruned = sorted_state_list[k:]

    #             candidate_generating_states = []

    #             for state in sorted_state_pruned:
    #                 node = self.algo.graph.nodes[state]
    #                 head = self.get_head(node)
    #                 blocked_action_set = self.get_blocked_action_set(node,head)
    #                 candidate_generating_states.append((state,head,blocked_action_set))

    #             return candidate_generating_states


            

    # def compute_likelihoods(self, state_list):

    #     num_states = len(state_list)
        
    #     Q = np.empty((num_states, num_states))
    #     root_idx = None

    #     i = 0
    #     for state in state_list:

    #         if state==self.model.init_state:
    #             root_idx = state_list.index(state)

    #         Q_vector = np.zeros(num_states)

    #         node = self.algo.graph.nodes[state]
    #         children = node.children[node.best_action]

    #         for child, child_prob in children:
    #             if child.terminal != True:
    #                 idx = state_list.index(child.state)
    #                 Q_vector[idx] = child_prob

    #         Q[i,:] = Q_vector
            
    #         i += 1


    #     t = time.time()
    #     I = np.identity(num_states)
    #     o = np.zeros(num_states)
    #     o[root_idx] = 1
    #     L = np.linalg.solve(np.transpose(I-Q), o)
    #     L = L / L[root_idx]

    #     return L  



    # def compute_likelihoods_2(self, state_list, pruned):

    #     num_states = len(state_list)
        
    #     Q = np.empty((num_states, num_states))
    #     root_idx = None

    #     i = 0
    #     for state in state_list:

    #         if state==self.model.init_state:
    #             root_idx = state_list.index(state)

    #         Q_vector = np.zeros(num_states)

    #         node = self.algo.graph.nodes[state]
    #         children = node.children[node.best_action]

    #         for child, child_prob in children:
    #             if child.terminal != True:
    #                 idx = state_list.index(child.state)
    #                 Q_vector[idx] = child_prob

    #         Q[i,:] = Q_vector
            
    #         i += 1

    #     for j in pruned:
    #         Q[j,:] = np.zeros(num_states)


    #     t = time.time()
    #     I = np.identity(num_states)
    #     o = np.zeros(num_states)
    #     o[root_idx] = 1
    #     L = np.linalg.solve(np.transpose(I-Q), o)

    #     return L      
