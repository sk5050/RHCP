#!/usr/bin/env python

import sys
import time
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
        print("     f: "+str(f_plus))
        print("     g: "+str(g_plus))

        self.add_anytime_solution(f_plus,g_plus)

        # print("-------------------------------")
        # print(f_plus + self.algo.alpha[0]*g_plus)
        # print("-------------------------------")
        
        # infinite case
        policy = self.resolve_LAOStar([initial_alpha_set[0][1]])
        value_1,value_2,value_3 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3)

        del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

        # self.algo = ILAOStar(self.model,constrained=True,bounds=self.bounds,alpha=[initial_alpha_set[0][1]],Lagrangian=True)
        # self.graph = self.algo.graph
        # self.algo.solve()

        f_minus = value_1
        g_minus = value_2 - self.bounds[0]

        self.add_anytime_solution(f_plus,g_plus)

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
            value_1,value_2,value_3 = self.algo.get_values(self.algo.graph.root)
            weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3)

            del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
            self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

            # self.algo = ILAOStar(self.model,constrained=True,bounds=self.bounds,alpha=[alpha],Lagrangian=True)
            # self.graph = self.algo.graph
            # self.algo.solve()

            L_u = value_1 + alpha*(value_2 - self.bounds[0])
            f = value_1
            g = value_2 - self.bounds[0]

            self.add_anytime_solution(f_plus,g_plus)

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





        
    # def incremental_update(self, num_sol):

    #     self.algo.incremental = True



    #     current_best_graph = self.copy_graph(self.algo.graph)
    #     current_best_policy = self.algo.extract_policy()

    #     for k in range(num_sol-1):

    #         for state,best_action in current_best_policy.items():

    #             if best_action=="Terminal":
    #                 continue

    #             node = self.algo.graph.nodes[state]
    #             new_candidate = self.find_candidate(node)

    #             if new_candidate:
    #                 if self.candidate_exists(new_candidate):
    #                     self.algo.graph = self.copy_graph(current_best_graph)  ## returning to the previous graph.
    #                     continue
    #                 else:
    #                     heappush(self.candidate_set, new_candidate)
    #             else:
    #                 continue

    #             self.algo.graph = self.copy_graph(current_best_graph)  ## returning to the previous graph.


    #        # time.sleep(1000)
                
    #         current_best_graph, current_best_policy = self.find_next_best()
    #         self.algo.graph = self.copy_graph(current_best_graph)


    def incremental_update(self, num_sol):

        self.algo.incremental = True

        current_best_graph = self.copy_best_graph(self.algo.graph)
        current_best_policy = self.algo.extract_policy()

        for k in range(num_sol-1):

            t = 0

            candidate_generating_states = self.prune_candidates(current_best_policy)

            for state,head,blocked_action_set in candidate_generating_states:

                t += 1

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
            
            print(t)


    def find_candidate(self, node, head, blocked_action_set):
        
        # self.algo.head = self.get_head(node)
        # self.algo.blocked_action_set = self.get_blocked_action_set(node,self.algo.head)

        # if self.algo.blocked_action_set != blocked_action_set:
        #     print("-----------------")
        #     print(self.algo.blocked_action_set)
        #     print(blocked_action_set)
            
        self.algo.head = head
        self.algo.blocked_action_set = blocked_action_set
        self.algo.tweaking_node = node

        ## if all actions are blocked, then no candidate is generated. 
        if len(self.algo.blocked_action_set)==len(self.model.actions(node.state)):
            return False

        
        self.algo.fringe = set([node])
        policy = self.algo.solve()

        value_1,value_2,value_3 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3)

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
    

    # def get_head(self,node):

    #     head = self.get_ancestors(node)
    #     queue = head.copy()

    #     while queue:
    #         popped_node = queue.pop()

    #         if popped_node == node:
    #             continue

    #         if popped_node.best_action!=None:
    #             children = popped_node.children[popped_node.best_action]

    #             for child,child_prob in children:
    #                 if child in head:
    #                     continue
    #                 else:
    #                     queue.add(child)
    #                     head.add(child)

    #         elif popped_node.terminal==True:
    #             continue

    #         else:
    #             raise ValueError("Policy seems have unexpanded node.")                    

    #     head.remove(node)
    #     return head


    
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

            if abs(candidate[0] - new_candidate[0]) > 1e-5:
                continue

            else:
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

            policy_value = self.algo.graph.root.value_1
            weighted_value_diff_list = []
            blocked_action_set_list = []
            head_list = []

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
                k += 1
                prob += i

                if prob >= 0.01:
                    break


            candidate_generating_states = sorted_states[k:]

            return candidate_generating_states



    # def compute_likelihoods(self, state_list):

    #     Q = []
    #     root_idx = None

    #     for state in state_list:

    #         if state==self.model.init_state:
    #             root_idx = state_list.index(state)

    #         Q_vector = [0]*len(state_list)

    #         node = self.algo.graph.nodes[state]
    #         children = node.children[node.best_action]

    #         for child, child_prob in children:
    #             if child.terminal != True:
    #                 idx = state_list.index(child.state)
    #                 Q_vector[idx] = child_prob

    #         Q.append(Q_vector)

    #     Q = np.matrix(Q)
    #     N = np.linalg.inv(np.eye(len(Q_vector)) - Q)

    #     L = N[0] / N[0,root_idx]

    #     return L


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

        # N = np.linalg.inv(np.eye(num_states) - Q)
        # L = N[root_idx] / N[root_idx, root_idx]

        # L = N[0] / N[0,root_idx]

        return L    



    def add_anytime_solution(self, f, g):

        if g<0:

            if len(self.anytime_solutions)==0:
                self.anytime_solutions.append((f, time.time() - self.t_start))

            else:

                if self.anytime_solutions[-1][0] > f:
                    self.anytime_solutions.append((f, time.time() - self.t_start))
