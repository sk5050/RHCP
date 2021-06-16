#!/usr/bin/env python

import sys
import time
from utils import *
from graph import Node, Graph


class VI(object):

    def __init__(self, model, VI_epsilon=1e-50, VI_max_iter=100000):

        self.model = model
        self.VI_epsilon = VI_epsilon
        self.VI_max_iter = VI_max_iter
        
        
        self.graph = Graph(name='G')
        self.graph.add_root(model.init_state, value=self.model.heuristic(model.init_state))

        self.debug_k = 0



    def solve(self):

        self.expand_all()
        self.value_iteration(list(self.graph.nodes.values()))
        self.update_best_partial_graph()
        return self.extract_policy()
    

    def expand_all(self):

        visited = set()
        queue = set([self.graph.root])

        while queue:

            node = queue.pop()

            if node in visited:
                continue

            else:
                children_nodes = self.expand(node)

                for child in children_nodes:
                    if child.terminal==False and child not in visited:
                        queue.add(child)
                
            visited.add(node)



    def expand(self,expanded_node):

        state = expanded_node.state
        actions = self.model.actions(state)

        children_nodes = []
        for action in actions:
            children = self.model.state_transitions(state,action)

            children_list = []
            for child_state, child_prob in children:
                if child_state in self.graph.nodes:
                    child = self.graph.nodes[child_state]
                else:
                    self.graph.add_node(child_state, self.model.heuristic(child_state))
                    child = self.graph.nodes[child_state]

                    if self.model.is_terminal(child.state):
                        child.set_terminal()
                
                child.parents_set.add(expanded_node)
                children_list.append([child,child_prob])

                children_nodes.append(child)

            expanded_node.children[action] = children_list


        return children_nodes


            

    def compute_value(self,node,action):

        cost_vector = self.model.cost(node.state,action)

        value = cost_vector

        for child, child_prob in node.children[action]:

            value = ptw_add(value, scalar_mul(child.value,child_prob))

        return value





    def value_iteration(self, Z, epsilon=1e-50, max_iter=100000, return_on_policy_change=False):

        iter=0

        V_prev = dict()
        V_new = dict()
        for node in Z:
            if node.terminal==False:
                V_prev[node.state] = node.value
                V_new[node.state] = [float('inf')]*2


        while not self.VI_convergence_test(V_prev,V_new,epsilon):
            for node in Z:
                if node.terminal==False:
                    V_prev[node.state] = node.value

                    actions = self.model.actions(node.state)
                    min_value = [float('inf')]*2
                    weighted_min_value = float('inf')

                    prev_best_action = node.best_action
                    best_action = None

                    for action in actions:

                        new_value = self.compute_value(node,action)

                        if new_value[0] < min_value[0]:
                            min_value = new_value
                            best_action = action



                    V_new[node.state] = min_value
                    if return_on_policy_change==True:
                        if prev_best_action != best_action:
                            return False

            for node in Z:
                if node.terminal==False:
                    node.value = V_new[node.state]

            iter += 1
                    
            if iter > max_iter:
                print("Maximun number of iteration reached.")
                break

        return V_new



    def VI_convergence_test(self,V_prev,V_new,epsilon):
        # might need more fast implementation. Numpy is better, but I am considering using pypy

        error = max([abs(V_prev[state][0]-V_new.get(state,0)[0]) for state in V_prev])

        
        if error < epsilon:
            return True
        else:
            return False



    def update_best_partial_graph(self):

        for state,node in self.graph.nodes.items():
            node.best_parents_set = set()
        
        visited = set()
        queue = set([self.graph.root])
        self.graph.root.color = 'w'

        while queue:

            node = queue.pop()

            if node in visited:
                continue

            else:
                if node.children!=dict():

                    actions = self.model.actions(node.state)
                    min_value = [float('inf')]*2
                    weighted_min_value = float('inf')
                    
                    for action in actions:
                        new_value = self.compute_value(node,action)

                        if new_value[0] < min_value[0]:
                            node.best_action = action
                            min_value = new_value


                    node.value = min_value
                    children = node.children[node.best_action]

                    for child,child_prob in children:
                        queue.add(child)
                        child.best_parents_set.add(node)
                        child.color = 'w'

            visited.add(node)


        

    def extract_policy(self):

        queue = set([self.graph.root])
        policy = dict()

        while queue:

            node = queue.pop()

            if node.state in policy:
                continue

            else:
                if node.best_action!=None:
                    policy[node.state] = node.best_action
                    children = node.children[node.best_action]

                    for child,child_prob in children:
                        queue.add(child)

                elif node.terminal==True:
                    policy[node.state] = "Terminal"

                else:
                    raise ValueError("Best partial graph has non-expanded fringe node.")

        return policy
        




    
