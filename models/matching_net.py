import torch
import numpy as np
from torch._C import dtype
from utils import mrr_mr_hitk
from .trainable import Trainable

class MatchingNet(Trainable):
    def __init__(self, scorer, l_model, r_model, sampler, c_num) -> None:
        super().__init__()
        self.scorer = scorer
        self.l_model = l_model
        self.r_model = r_model
        self.loss_func = torch.nn.NLLLoss()
        self.c_num = c_num
        self.sampler = sampler
        self.test_g_emb = None

    def forward(self, l_idx, r_idx):
        l_emb = self.l_model(l_idx)
        r_emb = self.r_model(r_idx)
        emb = torch.cat([l_emb, r_emb], dim=-1)
        score = self.scorer(emb)
        return score

    # the first column is the phrase id and second column is the concept id
    def train_step(self, idx, batch):
        pos_pairs = batch
        device = pos_pairs.device
        neg_pairs = self.sampler.sample(pos_pairs)
        neg_pairs = torch.LongTensor(neg_pairs).to(device)
        pos_label = torch.ones(pos_pairs.shape[0], device=device, dtype = torch.long)
        neg_label = torch.zeros(neg_pairs.shape[0], device=device, dtype = torch.long)
        pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        labels = torch.cat([pos_label, neg_label], dim=0)
        
        score = self(pairs[:,0], pairs[:,1])
        loss = self.loss_func(score, labels)
        return loss

    def dist_score(self, g_emb, p_emb):
        return torch.norm(g_emb-p_emb, dim=-1)

    def eval_step(self, idx, batch):
        p_idx = torch.unsqueeze(batch[:, 0], dim=1)
        labels = batch[:, 1]
        p_exp = p_idx.expand((batch.shape[0], self.c_num))
        c_exp = torch.arange(0, self.c_num).expand((batch.shape[0], self.c_num))
        scores = self(p_exp, c_exp)
        scores = scores[:, :, 1]

        mrr_tot = 0
        mr_tot = 0
        hit_tot = np.zeros((3,))
        count = 0
        for score, label in zip(scores, labels):
            mrr, mr, hit = mrr_mr_hitk(score, label, descend=True)
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