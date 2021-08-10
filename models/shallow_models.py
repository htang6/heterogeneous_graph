import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mrr_mr_hitk


class TransEModule(nn.Module):
    '''copy from NSCachine code'''
    def __init__(self, n_ent, n_rela, sampler, corrupter, filter_data, args):
        super(TransEModule, self).__init__()
        self.ent_emb = nn.Embedding(n_ent, args['hid_sz'])
        self.rela_emb = nn.Embedding(n_rela, args['hid_sz'])
        self.sampler = sampler
        self.init_weight()
        self.corrupter = corrupter
        self.p = args['p']

        self.filter_data = filter_data

    def forward(self, head, tail, rela):
        shape = head.size()
        head = head.contiguous().view(-1)
        tail = tail.contiguous().view(-1)
        rela = rela.contiguous().view(-1)
        head_embed = F.normalize(self.ent_emb(head),2,-1)
        tail_embed = F.normalize(self.ent_emb(tail),2,-1)
        rela_embed = self.rela_emb(rela)
        return torch.norm(tail_embed - head_embed - rela_embed, p=self.p, dim=-1).view(shape)

    def dist(self, head, tail, rela):
        return self.forward(head, tail, rela)

    def score(self, head, tail, rela):
        return self.forward(head, tail, rela)

    def prob_logit(self, head, tail, rela):
        return -self.forward(head, tail, rela) / self.temp

    def embed(self, idx):
        return self.ent_emb(idx)

    def point_loss(self, head, tail, rela, label):
        softplus = torch.nn.Softplus().cuda()
        score = self.forward(head, tail, rela)
        score = torch.sum(softplus(-1*label*score))
        return score

    def init_weight(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param.data)

    def train_step(self, idx, batch):
        h = batch[:,0]
        r = batch[:,1]
        t = batch[:,2]
        h_rand, t_rand = self.sampler.neg_sample(h.shape[0])
        prob = self.corrupter.bern_prob[r]
        selection = torch.bernoulli(prob).type(torch.BoolTensor)
        n_h = torch.LongTensor(h.cpu().numpy())
        n_t = torch.LongTensor(t.cpu().numpy())
        n_r = torch.LongTensor(r.cpu().numpy())
        if n_h.size() != h_rand.size():
            n_h = n_h.unsqueeze(1).expand_as(h_rand).clone()
            n_t = n_t.unsqueeze(1).expand_as(h_rand).clone()
            n_r = n_r.unsqueeze(1).expand_as(h_rand).clone()
            h = h.unsqueeze(1)
            r = r.unsqueeze(1)
            t = t.unsqueeze(1)
            
        n_h[selection] = h_rand[selection]
        n_t[~selection] = t_rand[~selection]

        n_h = n_h.to(h.device)
        n_t = n_t.to(h.device)
        n_r = n_r.to(h.device)
        
        p_loss = self.point_loss(h, t, r, 1)
        n_loss = self.point_loss(n_h, n_t, n_r, -1)

        loss = p_loss + n_loss
        return loss

    # Here we assume the first column is head, the second is relationship, the third is tail
    def eval_step(self, idx, batch, filt=True):
        mrr_tot = 0.
        mr_tot = 0
        hit_tot = np.zeros((3,))
        count = 0
        heads = self.filter_data[0]
        tails = self.filter_data[1]

        n_ent = self.ent_emb.weight.shape[0]
        batch_h = batch[:,0]
        batch_r = batch[:,1]
        batch_t = batch[:,2]
        batch_size = batch_h.size(0)

        head_val = batch_h.unsqueeze(1).expand(batch_size, n_ent)
        tail_val = batch_t.unsqueeze(1).expand(batch_size, n_ent)
        rela_val = batch_r.unsqueeze(1).expand(batch_size, n_ent)
        all_val = torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)\
        .type(torch.LongTensor).to(batch.device)

        batch_head_scores = self.score(all_val, tail_val, rela_val).data
        batch_tail_scores = self.score(head_val, all_val, rela_val).data
        for h, t, r, head_score, tail_score in zip(batch_h, batch_t, batch_r, batch_head_scores, batch_tail_scores):
            h_idx = int(h.data.cpu().numpy())
            t_idx = int(t.data.cpu().numpy())
            r_idx = int(r.data.cpu().numpy())
            if filt:
                # filter out triplets that are valid and may be precede the target triplet
                # by setting a very large score, smaller means more true
                if tails[(h_idx,r_idx)]._nnz() > 1:
                    tmp = tail_score[t_idx].data.cpu().numpy()
                    idx = tails[(h_idx, r_idx)]._indices()
                    tail_score[idx] = 1e20
                    tail_score[t_idx] = torch.from_numpy(tmp)
                if heads[(t_idx, r_idx)]._nnz() > 1:
                    tmp = head_score[h_idx].data.cpu().numpy()
                    idx = heads[(t_idx, r_idx)]._indices()
                    head_score[idx] = 1e20
                    head_score[h_idx] = torch.from_numpy(tmp)
            mrr, mr, hit = mrr_mr_hitk(tail_score, t_idx)
            mrr_tot += mrr
            mr_tot += mr
            hit_tot += hit
            mrr, mr, hit = mrr_mr_hitk(head_score, h_idx)
            mrr_tot += mrr
            mr_tot += mr
            hit_tot += hit
            count += 2

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

# class ComplExModule(nn.Module):
#     '''copy from NSCachine code'''
#     def __init__(self, n_ent, n_rel, args):
#         super(ComplExModule, self).__init__()

#         self.ent_re_embed = nn.Embedding(n_ent, args.hidden_dim)
#         self.ent_im_embed = nn.Embedding(n_ent, args.hidden_dim)

#         self.rel_re_embed = nn.Embedding(n_rel, args.hidden_dim)
#         self.rel_im_embed = nn.Embedding(n_rel, args.hidden_dim)
#         self.init_weight()

#     def forward(self, head, tail, rela):
#         shapes = head.size()
#         head = head.contiguous().view(-1)
#         tail = tail.contiguous().view(-1)
#         rela = rela.contiguous().view(-1)

#         head_re_embed = self.ent_re_embed(head)
#         tail_re_embed = self.ent_re_embed(tail)
#         rela_re_embed = self.rel_re_embed(rela)
#         head_im_embed = self.ent_im_embed(head)
#         tail_im_embed = self.ent_im_embed(tail)
#         rela_im_embed = self.rel_im_embed(rela)

#         score = torch.sum(rela_re_embed * head_re_embed * tail_re_embed, dim=-1) \
#                 + torch.sum(rela_re_embed * head_im_embed * tail_im_embed, dim=-1) \
#                 + torch.sum(rela_im_embed * head_re_embed * tail_im_embed, dim=-1) \
#                 - torch.sum(rela_im_embed * head_im_embed * tail_re_embed, dim=-1)
#         return score.view(shapes)

#     def dist(self, head, tail, rela):
#         return -self.forward(head, tail, rela)

#     def score(self, head, tail, rela):
#         return -self.forward(head, tail, rela)

#     def prob_logit(self, head, tail, rela):
#         return self.forward(head, tail, rela)/self.temp