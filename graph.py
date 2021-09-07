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
        p_list = []
        for node in self.nodes:
            p_list.append(node['name'])
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

class LeviGraph():
    def __init__(self, hgraph) -> None:
        self.allphrase, self.levi_nodes, self.levi_edges, self.c_num = self.transform(hgraph)

    def transform(self, hgraph):
        nodes = hgraph.nodes
        edges = hgraph.edges
        connects = hgraph.connects
        nodes_phrase = self.get_nodes_phrase(nodes)
        rela_phrase = self.get_rela_phrase(edges)
        all_phrase = nodes_phrase + rela_phrase
        node_num = len(nodes)

        levi_nodes = [i for i in range(len(nodes))]
        levi_edge = []
        for connect in connects:
            origin_idx = connect[2] + node_num #TODO VERIFY THE DATA ORDER IN CONNECT
            cur_idx = len(levi_nodes)
            levi_nodes.append(origin_idx)
            levi_edge.append([connect[0], cur_idx])
            levi_edge.append([cur_idx, connect[1]])

        return all_phrase, levi_nodes, levi_edge, len(hgraph.nodes)

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

    def get_nodes_phrase(self, nodes):
        p_list = []
        for node in nodes:
            p_list.append(node['name'])
        return  p_list

    def get_rela_phrase(self, edges):
        p_list = []
        for edge in edges:
            if isinstance(edge, dict):
                p = edge['name']
            else:
                p = edge
            p = p.replace('_', ' ')
            p_list.append(p)
        return p_list  