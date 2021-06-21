#!/usr/bin/env python

import sys
import time
from utils import *
from graph import Node, Graph
from LAOStar import LAOStar
from ILAOStar import ILAOStar


class CSSPSolver(object):

    def __init__(self, model, VI_epsilon=1e-50, VI_max_iter=100000, convergence_epsilon=1e-50, bounds=[]):

        self.model = model
        self.bounds = bounds

        # self.algo = LAOStar(self.model,constrained=True,VI_epsilon=VI_epsilon, VI_max_iter=VI_max_iter, \
        #                     convergence_epsilon=convergence_epsilon,\
        #                     bounds=self.bounds,Lagrangian=True)
        
        self.algo = ILAOStar(self.model,constrained=True,VI_epsilon=VI_epsilon, convergence_epsilon=convergence_epsilon,\
                             bounds=self.bounds,alpha=[0.0],Lagrangian=True)
        self.graph = self.algo.graph


    def solve(self, initial_alpha_set):

        self.find_dual(initial_alpha_set)
        # self.find_dual_multiple_bounds(initial_alpha_set)
        # self.anytime_update()

 
    def find_dual(self, initial_alpha_set):

        start_time = time.time()

        ##### phase 1
        # zero case
        self.algo.alpha = [initial_alpha_set[0][0]]

        policy = self.algo.solve()
        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        
        f_plus = value_1
        g_plus = value_2 - self.bounds[0]

        print(time.time() - start_time)

        # print("-------------------------------")
        # print(f_plus + self.algo.alpha[0]*g_plus)
        # print("-------------------------------")
        
        # infinite case
        self.resolve_LAOStar([initial_alpha_set[0][1]])

        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        
        f_minus = value_1
        g_minus = value_2 - self.bounds[0]

        print(time.time() - start_time)


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
            self.resolve_LAOStar([alpha])

            print(time.time() - start_time)

            value_1 = self.algo.graph.root.value_1
            value_2 = self.algo.graph.root.value_2

            L_u = value_1 + alpha*(value_2 - self.bounds[0])
            f = value_1
            g = value_2 - self.bounds[0]

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
           
            print("elapsed time: "+str(time.time()-start_time))



    def find_dual_multiple_bounds(self, initial_alpha_set):

        ###### TODO: functionize "solve_LAOStar with 'resolve' as an option"


        # overall zero case
        self.algo.alpha = [alpha_set[0] for alpha_set in initial_alpha_set]

        policy = self.algo.solve()
        value = self.algo.graph.root.value
        primary_value = value[0]
        secondary_value = value[1:]

        f_plus = primary_value
        g_plus = ptw_sub(secondary_value, self.bounds)

        print("---------- zero case --------")
        for g_temp in g_plus:
            print(g_temp)
        print(f_plus + dot(self.algo.alpha, g_plus))


        ###### TODO: Need to check whether this solution is feasible, which is the case that we already found optima.


        # overall infinite case
        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]
        self.resolve_LAOStar()

        value = self.algo.graph.root.value
        primary_value = value[0]
        secondary_value = value[1:]

        f_minus = primary_value
        g_minus = ptw_sub(secondary_value, self.bounds)

        print("---------- infinite case --------")
        print(self.algo.alpha)
        for g_temp in g_minus:
            print(g_temp)
        print(f_minus + dot(self.algo.alpha, g_minus))

        ###### TODO: Need to check whether this solution is infeasible, which is the case that the problem is infeasible, or alpha_max is not large enough.


        
        self.algo.alpha = [alpha_set[0] for alpha_set in initial_alpha_set]

        L_new = f_plus + dot(self.algo.alpha, g_plus)
        while True:

            
            L_prev = L_new
            
            for bound_idx in range(len(initial_alpha_set)):  # looping over each bound (coorindate)

                print("*"*20)

                # zero case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][0]
                self.resolve_LAOStar()

                value = self.algo.graph.root.value
                primary_value = value[0]
                secondary_value = value[1:]

                f_plus = primary_value
                g_plus = ptw_sub(secondary_value, self.bounds)

                print(self.algo.alpha)
                print(f_plus + dot(self.algo.alpha, g_plus))

                # infinite case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][1]
                self.resolve_LAOStar()

                value = self.algo.graph.root.value
                primary_value = value[0]
                secondary_value = value[1:]

                f_minus = primary_value
                g_minus = ptw_sub(secondary_value, self.bounds)


                print(self.algo.alpha)
                print(f_minus + dot(self.algo.alpha, g_minus))


                while True:

                    new_alpha_comp = (f_plus - f_minus)

                    for bound_idx_inner in range(len(initial_alpha_set)):
                        if bound_idx_inner==bound_idx:
                            continue

                        # print(self.algo.alpha)
                        # print(g_plus)
                        # print(g_minus)
                        new_alpha_comp += self.algo.alpha[bound_idx_inner] * (g_plus[bound_idx_inner] - g_minus[bound_idx_inner])

                    new_alpha_comp = new_alpha_comp / (g_minus[bound_idx]- g_plus[bound_idx])

                    self.algo.alpha[bound_idx] = new_alpha_comp

                    print(self.algo.alpha)

                    L = f_plus + dot(self.algo.alpha, g_plus)
                    UB = float('inf')


                    print(L)
                    # time.sleep(10000)
                    
                    # evaluate L(u), f, g
                    self.resolve_LAOStar()

                    value = self.algo.graph.root.value
                    primary_value = value[0]
                    secondary_value = value[1:]

                    f = primary_value
                    g = ptw_sub(secondary_value, self.bounds)
                    L_u = f + dot(self.algo.alpha, g)


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



                ## optimality check for this entire envelop    

            L_new = L_u

            if abs(L_new - L_prev) < 0.1**300:
                break

            
        print("optimal solution found during phase 1!")
        print("dual optima with the following values:")
        print(" alpha:"+str(self.algo.alpha))
        print("     L: "+str(L))
        print("     f: "+str(f))
        print("     g: "+str(g))


        


    def resolve_LAOStar(self, new_alpha=None):

        if new_alpha != None:
            self.algo.alpha = new_alpha
            
        nodes = list(self.algo.graph.nodes.values())
        self.algo.value_iteration(nodes)
        self.algo.update_fringe()
        self.algo.solve()
