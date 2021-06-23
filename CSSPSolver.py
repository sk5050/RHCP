#!/usr/bin/env python

import sys
import time
from utils import *
from graph import Node, Graph
from LAOStar import LAOStar
from ILAOStar import ILAOStar

import copy
from heapq import *

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
        value_1,value_2,value_3 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3)
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))
        
        f_plus = value_1
        g_plus = value_2 - self.bounds[0]

        print("-"*50)
        print("time elapsed: "+str(time.time() - start_time))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))

        # print("-------------------------------")
        # print(f_plus + self.algo.alpha[0]*g_plus)
        # print("-------------------------------")
        
        # infinite case
        policy = self.resolve_LAOStar([initial_alpha_set[0][1]])
        value_1,value_2,value_3 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3)
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

        # self.algo = ILAOStar(self.model,constrained=True,bounds=self.bounds,alpha=[initial_alpha_set[0][1]],Lagrangian=True)
        # self.graph = self.algo.graph
        # self.algo.solve()

        f_minus = value_1
        g_minus = value_2 - self.bounds[0]

        print("-"*50)
        print("time elapsed: "+str(time.time() - start_time))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))


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
            value_1,value_2,value_3 = self.algo.get_values(self.algo.graph.root)
            weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3)

            del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
            self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

            # self.algo = ILAOStar(self.model,constrained=True,bounds=self.bounds,alpha=[alpha],Lagrangian=True)
            # self.graph = self.algo.graph
            # self.algo.solve()

            print("-"*50)
            print("time elapsed: "+str(time.time() - start_time))
            print("nodes expanded: "+str(len(self.algo.graph.nodes)))

            L_u = value_1 + alpha*(value_2 - self.bounds[0])
            f = value_1
            g = value_2 - self.bounds[0]

            # print("-------------------------------")
            # print(L_u)
            # print("-------------------------------")
            
            # cases
            if abs(L_u - L)<0.1**5 and g < 0:
                LB = L_u
                UB = min(f, UB)
                break
            
            elif abs(L_u - L)<0.1**5 and g > 0:
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

            



    # def find_dual_multiple_bounds(self, initial_alpha_set):

    #     ###### TODO: functionize "solve_LAOStar with 'resolve' as an option"


    #     # overall zero case
    #     self.algo.alpha = [alpha_set[0] for alpha_set in initial_alpha_set]

    #     policy = self.algo.solve()
    #     value = self.algo.graph.root.value
    #     primary_value = value[0]
    #     secondary_value = value[1:]

    #     f_plus = primary_value
    #     g_plus = ptw_sub(secondary_value, self.bounds)

    #     print("---------- zero case --------")
    #     for g_temp in g_plus:
    #         print(g_temp)
    #     print(f_plus + dot(self.algo.alpha, g_plus))


    #     ###### TODO: Need to check whether this solution is feasible, which is the case that we already found optima.


    #     # overall infinite case
    #     self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]
    #     self.resolve_LAOStar()

    #     value = self.algo.graph.root.value
    #     primary_value = value[0]
    #     secondary_value = value[1:]

    #     f_minus = primary_value
    #     g_minus = ptw_sub(secondary_value, self.bounds)

    #     print("---------- infinite case --------")
    #     print(self.algo.alpha)
    #     for g_temp in g_minus:
    #         print(g_temp)
    #     print(f_minus + dot(self.algo.alpha, g_minus))

    #     ###### TODO: Need to check whether this solution is infeasible, which is the case that the problem is infeasible, or alpha_max is not large enough.


        
    #     self.algo.alpha = [alpha_set[0] for alpha_set in initial_alpha_set]

    #     L_new = f_plus + dot(self.algo.alpha, g_plus)
    #     while True:

            
    #         L_prev = L_new
            
    #         for bound_idx in range(len(initial_alpha_set)):  # looping over each bound (coorindate)

    #             print("*"*20)

    #             # zero case for this coordinate
    #             self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][0]
    #             self.resolve_LAOStar()

    #             value = self.algo.graph.root.value
    #             primary_value = value[0]
    #             secondary_value = value[1:]

    #             f_plus = primary_value
    #             g_plus = ptw_sub(secondary_value, self.bounds)

    #             print(self.algo.alpha)
    #             print(f_plus + dot(self.algo.alpha, g_plus))

    #             # infinite case for this coordinate
    #             self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][1]
    #             self.resolve_LAOStar()

    #             value = self.algo.graph.root.value
    #             primary_value = value[0]
    #             secondary_value = value[1:]

    #             f_minus = primary_value
    #             g_minus = ptw_sub(secondary_value, self.bounds)


    #             print(self.algo.alpha)
    #             print(f_minus + dot(self.algo.alpha, g_minus))


    #             while True:

    #                 new_alpha_comp = (f_plus - f_minus)

    #                 for bound_idx_inner in range(len(initial_alpha_set)):
    #                     if bound_idx_inner==bound_idx:
    #                         continue

    #                     # print(self.algo.alpha)
    #                     # print(g_plus)
    #                     # print(g_minus)
    #                     new_alpha_comp += self.algo.alpha[bound_idx_inner] * (g_plus[bound_idx_inner] - g_minus[bound_idx_inner])

    #                 new_alpha_comp = new_alpha_comp / (g_minus[bound_idx]- g_plus[bound_idx])

    #                 self.algo.alpha[bound_idx] = new_alpha_comp

    #                 print(self.algo.alpha)

    #                 L = f_plus + dot(self.algo.alpha, g_plus)
    #                 UB = float('inf')


    #                 print(L)
    #                 # time.sleep(10000)
                    
    #                 # evaluate L(u), f, g
    #                 self.resolve_LAOStar()

    #                 value = self.algo.graph.root.value
    #                 primary_value = value[0]
    #                 secondary_value = value[1:]

    #                 f = primary_value
    #                 g = ptw_sub(secondary_value, self.bounds)
    #                 L_u = f + dot(self.algo.alpha, g)


    #                 # cases
    #                 if abs(L_u - L)<0.1**10 and g[bound_idx] < 0:
    #                     LB = L_u
    #                     UB = min(f, UB)
    #                     break

    #                 elif abs(L_u - L)<0.1**10 and g[bound_idx] > 0:
    #                     LB = L_u
    #                     UB = f_minus
    #                     break

    #                 elif L_u < L and g[bound_idx] > 0:
    #                     f_plus = f
    #                     g_plus = g

    #                 elif L_u < L and g[bound_idx] < 0:
    #                     f_minus = f
    #                     g_minus = g
    #                     UB = min(UB, f)

    #                 elif g[bound_idx]==0:
    #                     raise ValueError("opt solution found during phase 1. not implented for this case yet.")

    #                 elif L_u > L :
    #                     print(L_u)
    #                     print(L)
    #                     raise ValueError("impossible case. Something must be wrong")



    #             ## optimality check for this entire envelop    

    #         L_new = L_u

    #         if abs(L_new - L_prev) < 0.1**300:
    #             break

            
    #     print("optimal solution found during phase 1!")
    #     print("dual optima with the following values:")
    #     print(" alpha:"+str(self.algo.alpha))
    #     print("     L: "+str(L))
    #     print("     f: "+str(f))
    #     print("     g: "+str(g))


        


    def resolve_LAOStar(self, new_alpha=None,epsilon=1e-10):

        if new_alpha != None:
            self.algo.alpha = new_alpha
            
        nodes = list(self.algo.graph.nodes.values())
        self.algo.value_iteration(nodes,epsilon=self.resolve_epsilon)
        self.algo.update_fringe()
        return self.algo.solve()





        
    def incremental_update(self, num_sol):

        self.algo.incremental = True



        current_best_graph = self.copy_graph(self.algo.graph)
        current_best_policy = self.algo.extract_policy()
        
        for k in range(num_sol-2):
            for state,best_action in current_best_policy.items():

                if state=="Terminal":
                    continue

                node = current_best_graph.nodes[state]
                new_candidate = self.find_candidate(node)

                if new_candidate:
                    heappush(self.candidate_set, new_candidate)
                else:
                    continue

                self.algo.graph = current_best_graph  ## returning to the previous graph.

            current_best_graph, current_best_policy = self.find_next_best()
            self.algo.graph = current_best_graph



    def find_candidate(self, node):
        
        self.algo.head = self.get_head(node)
        self.algo.blocked_action_set = self.get_blocked_action_set(node,self.algo.head)
        self.algo.tweaking_node = node

        ## if all actions are blocked, then no candidate is generated. 
        if len(self.algo.blocked_action_set)==len(self.model.actions(node.state)):
            return False
        
        self.algo.fringe = set([node])
        policy = self.algo.solve()

        value_1,value_2,value_3 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3)

        return (weighted_value, (value_1,value_2,value_3), self.copy_graph(self.algo.graph), policy)



    def get_blocked_action_set(self,node,head):

        blocked_action_set = set()

        ## TODO: current policy need not be inspected in this way. Can be more easily done. Fix it for better efficiency later.
        for solution in self.k_best_solution_set:
            
            prev_policy = solution[2]

            if self.is_head_eq(head, prev_policy):
                blocked_action_set.add(prev_policy[node.state])

        return blocked_action_set

    

    def get_head(self,node):

        ancestors = self.get_ancestors(node)
        queue = ancestors.copy()
        head = ancestors.copy()

        while queue:
            popped_node = queue.pop()

            if popped_node == node:
                continue

            if popped_node.best_action!=None:
                children = popped_node.children[popped_node.best_action]

                for child,child_prob in children:
                    if child in head:
                        continue
                    else:
                        queue.add(child)
                        head.add(child)

            elif popped_node.terminal==True:
                continue

            else:
                raise ValueError("Policy seems have unexpanded node.")                    

        head.remove(node)
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



        
    def find_next_best(self,node):

        next_best_candidate = heappop(self.candidate_set)

        current_best_weighted_value = next_best_candidate[0]
        current_best_values = next_best_candidate[1]
        current_best_graph = next_best_candidate[2]
        current_best_policy = next_best_candidate[3]

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
