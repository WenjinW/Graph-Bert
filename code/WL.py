'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.method import method
import hashlib

class MethodWLNodeColoring(method):
    data = None
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    def __init__(self, node_list, edge_list):
        self.node_color_dict = {}
        self.node_neighbor_dict = {}

        # convert the Tensor to list
        node_list = node_list.tolist()
        edge_list_1, edge_list_2 = edge_list[0].tolist(), edge_list[1].tolist()
        
        for node in node_list:
            self.node_color_dict[node] = 1
            self.node_neighbor_dict[node] = {}
        
        for index in range(len(edge_list_1)):
            u1, u2 = edge_list_1[index], edge_list_2[index]
            if u1 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u1] = {}
            if u2 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u2] = {}
            self.node_neighbor_dict[u1][u2] = 1
            self.node_neighbor_dict[u2][u1] = 1
        
        self.WL_recursion(node_list)
    
    def get_WL(self):
        
        return self.node_color_dict

    def WL_recursion(self, node_list):
        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in node_list:
                neighbors = self.node_neighbor_dict[node]
                neighbor_color_list = [self.node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(self.node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if self.node_color_dict == new_color_dict or iteration_count == self.max_iter:
                return
            else:
                self.node_color_dict = new_color_dict
            iteration_count += 1
