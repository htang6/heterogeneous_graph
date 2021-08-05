import torch
from torch import nn
import numpy as np
from torch._C import dtype
from transformers.models import bert

from utils import mrr_mr_hitk

class SynoPred(nn.Module):
    def __init__(self, c_emb, bert, sampler, **args) -> None:
        super().__init__()
        self.c_emb = c_emb
        self.bert = bert
        self.sampler = sampler
        self.loss = nn.NLLLoss()

        graph_dim =  self.c_emb.embedding_dim
        phrase_dim = 768
        self.fc = nn.Sequential(
                                nn.Dropout(args['dropout']),
                                nn.Linear(graph_dim + phrase_dim, args['hid_sz']),      
                                nn.ReLU(),
                                nn.Dropout(args['dropout']),
                                nn.Linear(args['hid_sz'], 2),
                                nn.LogSoftmax(1)
                            )

    # input should be tensor of pairs with phrase in the left and concept
    # in the right
    def forward(self, pid, cid):
        emb_right = self.c_emb(cid)
        emb_left = self.bert(pid)
        emb = torch.cat([emb_left, emb_right], dim=1)
        return self.fc(emb)

    def train_step(self, idx, batch):
        # first column of batch is the phrase, second is the concept
        neg_sample = self.sampler.sample(batch)
        neg_sample = torch.from_numpy(neg_sample).to(batch.device)
        pos_label = torch.ones(batch.shape[0], dtype=torch.long, device=batch.device)
        neg_label = torch.zeros(neg_sample.shape[0], dtype=torch.long, device=batch.device)
        full_batch = torch.cat((batch, neg_sample), dim=0)
        full_label = torch.cat([pos_label, neg_label], dim=0)

        score = self.forward(full_batch[:,0], full_batch[:,1])
        loss = self.loss(score, full_label)
        return loss

    def eval_step(self, idx, batch):
        batch_pid = batch[:,0]
        batch_cid = batch[:,1]
        batch_score = self.retr_score(batch_pid)
        batch_score = batch_score[:,:,1]
        return (batch_cid, batch_score)
            
    
    def eval_sum(self, result_list):
        mrr_tot = 0.
        mr_tot = 0
        hit_tot = np.zeros((3,))
        count = 0
        for result in result_list:
             for target, score in zip(*result):
                # Since higher score means higher rank here, so we need to pass in True
                mrr, mr, hit = mrr_mr_hitk(score, target, descend=True)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                count+=1
        return {'mrr':float(mrr_tot)/count,
                'mr':mr_tot/count,
                'hit@1':hit_tot[0]/count,
                'hit@3':hit_tot[1]/count,
                'hit@10':hit_tot[2]/count,
                }

    def retr_score(self, pid):
        n_ent = self.graph_model.ent_emb.num_embeddings
        all_cid = torch.arange(0, n_ent).unsqueeze(0).expand(pid.shape[0], n_ent).to(pid.device)
        phrase_emb = self.bert(pid)
        phrase_emb = phrase_emb.unsqueeze(1).expand(phrase_emb.shape[0],\
         n_ent, phrase_emb.shape[1])
        c_emb = self.graph_model.embed(all_cid)
        
        emb = torch.cat([phrase_emb, c_emb], dim=2)
        result = self.fc(emb)
        return result
        









    