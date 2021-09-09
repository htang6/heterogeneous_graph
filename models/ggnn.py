import torch
import torch.nn as nn
import numpy as np
from utils import mrr_mr_hitk
from .trainable import Trainable

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        # A_in = A[:, :, :self.n_node*self.n_edge_types]
        # A_out = A[:, :, self.n_node*self.n_edge_types:]
        state_in = torch.squeeze(state_in, dim=0)
        state_out = torch.squeeze(state_out, dim=0)
        A_in = A[0]
        A_out = A[1]

        a_in = torch.sparse.mm(A_in, state_in) # Here it gets an embeddings by merging embeddings from different edges
        a_out = torch.sparse.mm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 1)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 1) #change from 2 to one because reduce dimensions
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, **opt):
        super(GGNN, self).__init__()

        self.state_dim = opt['state_dim']
        self.n_edge_types = 1 #Must be one for model to run properly
        self.n_node = opt['n_node']
        self.n_steps = opt['n_steps']

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, A):
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            # Here it does a different mapping for different edges
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)

        return prop_state

class HGGNN(nn.Module):
    def __init__(self, ggnn, g_emb, adj) -> None:
        super().__init__()
        self.ggnn = ggnn
        self.register_buffer('g_emb', g_emb)
        self.register_buffer('forward_adj', adj[0])
        self.register_buffer('backward_adj', adj[1])

    def forward(self, idx):
        g_emb = self.ggnn(self.g_emb, [self.forward_adj, self.backward_adj])
        return g_emb[idx]

    # the first column is the phrase id and second column is the concept id