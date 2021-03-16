#!/usr/bin/env python

import sys
from graph import Node, Graph


class LAOStar(object):

    def __init__(self, model, method='VI', lb=None, ub=None, alpha=0.0):

        self.model = model
        self.lb = lb
        self.ub = ub
        self.alpha = alpha
        
        self.graph = Graph(name='G')
        self.graph.add_root(model.init_state, value=None)

        self.fringe = {self.graph.root}


    def solve(self):

        while not is_termination():

            expanded_node = self.expand()
            self.update_values_and_graph(expanded_node)
            self.get_fringe()



    def is_termination(self):

        if self.fringe:
            return False
        else:
            if self.method=='VI':
                if self.convergence_test():
                    return True
                else:
                    self.get_fringe()

            return True


    def update_values_and_graph():

        Z = self.get_ancestors(expanded_node)

        if self.method=='VI':
            self.value_iteration(Z)
        elif self.method=='PI':
            raise ValueError("Not yet implemented.")
        else:
            raise ValueError("Error in method choice.")

        self.update_best_partial_graph()



    def get_ancestors(self, expanded_node):
        Z = []

        queue = set(expanded_node)

        while queue:
            node = queue.pop(0)

            if node not in Z:
                Z.append(node)
                parents = node.best_parents_set
                queue
            



    def value_iteration(self, updating_nodes):
        print("hi")



    def update_best_partial_graph(self):
        print("hi")



    def get_fringe(self):
        print("hi")
        
