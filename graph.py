import numpy as np
import torch
from torch.autograd import backward
class HeteroGraph():
    '''
    the connects should be (h,t,r)
    '''
    def __init__(self, nodes, edges, connects, ent2idx) -> None:
        self.nodes = nodes
        self.edges = edges
        self.connects = connects
        self.ent2idx = ent2idx

    def get_nodes_phrase(self):
        p_list = [node['name'] for node in self.nodes]
        return  p_list

    def get_rela_phrase(self):
        p_list = []
        for edge in self.edges:
            if isinstance(edge, dict):
                p = edge['name']
            else:
                p = edge
            p = p.replace('_', ' ')
            p_list.append(p)
        return p_list

    def get_simple_graph(self):
        all_phrases = [node['name'] for node in self.nodes]
        simple_nodes = [i for i in range(len(all_phrases))]
        simple_edges = [[con[0], con[1]] for con in self.connects]

        return LeviGraph(all_phrases, simple_nodes, simple_edges, len(simple_nodes))

    def get_levi_graph(self):
        nodes_phrase = self.get_nodes_phrase()
        rela_phrase = self.get_rela_phrase()
        all_phrase = nodes_phrase + rela_phrase
        node_num = len(self.nodes)

        levi_nodes = [i for i in range(node_num)]
        levi_edge = []
        for connect in self.connects:
            origin_idx = connect[2] + node_num #TODO VERIFY THE DATA ORDER IN CONNECT
            cur_idx = len(levi_nodes)
            levi_nodes.append(origin_idx)
            levi_edge.append([connect[0], cur_idx])
            levi_edge.append([cur_idx, connect[1]])

        return LeviGraph(all_phrase, levi_nodes, levi_edge, len(self.nodes))

class LeviGraph():
    def __init__(self, phrases, nodes, edges, c_num) -> None:
        self.allphrase = phrases
        self.levi_nodes = nodes
        self.levi_edges = edges
        self.c_num = c_num

    def get_adj(self, ):
        sz = len(self.levi_nodes)
        edges = self.levi_edges
        back_edges = []

        for edge in edges: 
            back_edges.append([edge[1], edge[0]])

        edges = torch.LongTensor(edges)
        back_edges = torch.LongTensor(back_edges)
        values = torch.ones((edges.shape[0]))

        adj = torch.sparse_coo_tensor(edges.t(), values, size=(sz, sz))
        adj_back = torch.sparse_coo_tensor(back_edges.t(), values, size=(sz, sz))

        result = [adj, adj_back]
        return result