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


class GGNN(Trainable):
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

class HGGNN(Trainable):
    def __init__(self, ggnn, p_emb, g_emb, adj, sampler, c_num) -> None:
        super().__init__()
        self.ggnn = ggnn
        self.c_num = c_num
        self.sampler = sampler
        self.test_g_emb = None
        self.register_buffer('p_emb', p_emb)
        self.register_buffer('g_emb', g_emb)
        self.register_buffer('forward_adj', adj[0])
        self.register_buffer('backward_adj', adj[1])

    def forward(self, idx):
        g_emb = self.ggnn(self.g_emb, [self.forward_adj, self.backward_adj])
        return g_emb[idx]

    # the first column is the phrase id and second column is the concept id
    def train_step(self, idx, pos_pairs):
        device = pos_pairs.device
        neg_pairs = self.sampler.sample(pos_pairs)
        neg_pairs = torch.LongTensor(neg_pairs).to(device)
        pos_label = torch.ones(pos_pairs.shape[0], device=device)
        neg_label = torch.ones(neg_pairs.shape[0], device=device) * (-1)
        pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        labels = torch.cat([pos_label, neg_label], dim=0)
        
        
        g_emb = self(pairs[:,1])
        p_emb = self.p_emb[pairs[:,0]]
        score = self.dist_score(g_emb, p_emb)
        loss = torch.matmul(score, labels)
        return loss

    def dist_score(self, g_emb, p_emb):
        return torch.norm(g_emb-p_emb, dim=-1)

    def eval_pre(self):
        self.test_g_emb = self.ggnn(self.g_emb, [self.forward_adj, self.backward_adj])

    def eval_step(self, idx, batch):
        p_idx = torch.unsqueeze(batch[:, 0], dim=1)
        labels = batch[:, 1]
        p_exp = p_idx.expand((batch.shape[0], self.c_num))
        c_exp = torch.arange(0, self.c_num).expand((batch.shape[0], self.c_num))
        g_emb = self.test_g_emb[c_exp]
        p_emb = self.p_emb[p_exp]
        scores = self.dist_score(g_emb, p_emb)

        mrr_tot = 0
        mr_tot = 0
        hit_tot = np.zeros((3,))
        count = 0
        for score, label in zip(scores, labels):
            mrr, mr, hit = mrr_mr_hitk(score, label)
            mrr_tot += mrr
            mr_tot += mr
            hit_tot += hit
            count += 1

        return {'mrr':float(mrr_tot),
                'mr':mr_tot,
                'hit':hit_tot,
                'count':count
                }

    def eval_sum(self, result_list):
        mrr_tot = 0.
        mr_tot = 0
        hit_tot = np.zeros((3,))
        count = 0
        for result in result_list:
            mrr_tot += result['mrr']
            mr_tot += result['mr']
            hit_tot += result['hit']
            count += result['count']
            
        return {'mrr':float(mrr_tot)/count,
                'mr':mr_tot/count,
                'hit@1':hit_tot[0]/count,
                'hit@3':hit_tot[1]/count,
                'hit@10':hit_tot[2]/count,
                }
        
        