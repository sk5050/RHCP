#!/usr/bin/env python

import sys
from graph import Node, Graph


class LAOStar(object):

    def __init__(self, model, method='VI', lb=None, ub=None, alpha=0.0):

        self.model = model
        self.method = method
        self.lb = lb
        self.ub = ub
        self.alpha = alpha
        
        self.graph = Graph(name='G')
        self.graph.add_root(model.init_state, value=self.model.heuristic(model.init_state))

        self.fringe = {self.graph.root}


    def solve(self):

        while not self.is_termination():

            expanded_node = self.expand()
            self.update_values_and_graph(expanded_node)
            self.update_fringe()

        print(len(self.graph.nodes))
        self.print_policy()
        print(self.graph.root.value)

    def is_termination(self):

        if self.fringe:
            return False
        else:
            # if self.method=='VI':
            #     if self.convergence_test():
            #         return True
            #     else:
            #         self.update_fringe()

            return True



    def expand(self):
        expanded_node = self.fringe.pop()
        state = expanded_node.state
        actions = self.model.actions(state)

        for action in actions:
            children = self.model.state_transitions(state,action)

            children_list = []
            for child_state, child_prob in children:
                if child_state in self.graph.nodes:
                    child = self.graph.nodes[child_state]
                else:
                    self.graph.add_node(child_state, self.model.heuristic(child_state))
                    child = self.graph.nodes[child_state]

                    if child.state==self.model.goal:
                        child.set_terminal()
                
                child.parents_set.add(expanded_node)
                children_list.append([child,child_prob])

            expanded_node.children[action] = children_list

        return expanded_node

                    

    def update_values_and_graph(self, expanded_node):

        Z = self.get_ancestors(expanded_node)


        if self.method=='VI':
            V_new = self.value_iteration(Z)

        elif self.method=='PI':
            raise ValueError("Not yet implemented.")
        else:
            raise ValueError("Error in method choice.")

        self.update_best_partial_graph(Z, V_new)




    def get_ancestors(self, expanded_node):
        Z = []

        queue = set([expanded_node])

        while queue:
            node = queue.pop()

            if node not in Z:
                
                Z.append(node)
                parents = node.best_parents_set

                queue = queue.union(parents)

        return Z
            



    def value_iteration(self, Z, epsilon=1e-3, max_iter=10000):#float('inf')):

        print(len(Z))
        iter=0

        V_prev = dict()
        V_new = dict()
        for node in Z:
            if node.terminal==False:
                V_prev[node.state] = node.value
                V_new[node.state] = float('inf')


        while not self.VI_convergence_test(V_prev,V_new,epsilon):
            for node in Z:
                if node.terminal==False:
                    V_prev[node.state] = node.value

                    actions = self.model.actions(node.state)
                    min_value = float('inf')

                    for action in actions:
                        new_value = self.model.cost(node.state,action) + \
                            sum(child.value*child_prob for child,child_prob \
                                        in node.children[action])

                        if new_value < min_value:
                            min_value = new_value

                    V_new[node.state] = min_value

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

        error = max([abs(V_prev[state]-V_new.get(state,0)) for state in V_prev])

        if error < epsilon:
            return True
        else:
            return False

    def update_best_partial_graph(self, Z, V_new):

        visited = set()
        queue = set([self.graph.root])

        while queue:

            node = queue.pop()

            if node in visited:
                continue

            else:
                if node.children!=dict():

                    actions = self.model.actions(node.state)
                    min_value = float('inf')
                    for action in actions:
                        new_value = self.model.cost(node.state,action) + \
                            sum(child.value*child_prob for child,child_prob \
                                        in node.children[action])
                        if new_value < min_value:
                            node.best_action = action
                            min_value = new_value

                    node.value = min_value
                    children = node.children[node.best_action]

                    for child,child_prob in children:
                        queue.add(child)
                        child.best_parents_set.add(node)

            visited.add(node)

        

        # for state,node in self.graph.nodes.items():

        #     if node.best_action!=None and node.terminal==False:

        #         actions = self.model.actions(node.state)
        #         min_value = float('inf')
        #         for action in actions:
        #             new_value = self.model.cost(node.state,action) + \
        #                 sum(child.value*child_prob for child,child_prob \
        #                             in node.children[action])
        #             if new_value < min_value:
        #                 node.best_action = action
        #                 min_value = new_value

        #         best_children = node.children[node.best_action]

        #         for child,child_prob in best_children:
        #             if child in visited:
        #                 child.best_parents_set.add(node)
        #             else:
        #                 child.best_parents_set = set([node])



    def update_fringe(self):

        fringe = set()
        queue = set([self.graph.root])
        visited = set()

        while queue:

            node = queue.pop()

            if node in visited:
                continue

            else:
                if node.best_action!=None:  # if this node has not been expanded
                    children = node.children[node.best_action]

                    for child,child_prob in children:
                        queue.add(child)

                elif node.terminal != True:
                    fringe.add(node)

            visited.add(node)

        self.fringe = fringe

        

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


    def print_policy(self):

        policy = self.extract_policy()

        for state, action in policy.items():
            print(state, ' : ', action, self.graph.nodes[state].value,self.graph.nodes[state].terminal )
