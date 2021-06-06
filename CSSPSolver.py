#!/usr/bin/env python

import sys
import time
from utils import *
from graph import Node, Graph
from LAOStar import LAOStar


class CSSPSolver(object):

    def __init__(self, model, bounds=[]):

        self.model = model
        self.bounds = bounds

        self.algo = LAOStar(self.model,constrained=True,bounds=self.bounds,Lagrangian=True)
        self.graph = self.algo.graph


    def solve(self, LAOStar_method = 'VI'):

        self.find_dual(LAOStar_method)
        # self.anytime_update()

 
    def find_dual(self, LAOStar_method):

        start_time = time.time()

        ##### phase 1
        # zero case
        self.algo.alpha = [0.0]
        
        policy = self.algo.solve()
        value = self.algo.graph.root.value
        
        f_plus = value[0]
        g_plus = value[1] - self.bounds[0]

        print(time.time() - start_time)

        
        # infinite case
        self.resolve_LAOStar([10**5])

        value = self.algo.graph.root.value
        
        f_minus = value[0]
        g_minus = value[1] - self.bounds[0]

        print(time.time() - start_time)

        
        # phase 1 interation to compute alpha
        while True:

            # update alpha
            alpha = (f_minus - f_plus) / (g_plus - g_minus)
            L = f_plus + alpha*g_plus
            UB = float('inf')

           
            # evaluate L(u), f, g
            self.resolve_LAOStar([alpha])

            print(time.time() - start_time)

            value = self.algo.graph.root.value

            L_u = value[0] + alpha*(value[1] - self.bounds[0])
            f = value[0]
            g = value[1] - self.bounds[0]


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


    def resolve_LAOStar(self, new_alpha):
        self.algo.alpha = new_alpha
        nodes = list(self.algo.graph.nodes.values())
        self.algo.value_iteration(nodes)
        self.algo.update_fringe()
        self.algo.solve()
