import numpy as np
import torch

class TriRandomSampler():
    def __init__(self, ent_num, sample_num) -> None:
        self.ent_num = ent_num
        self.sample_num = sample_num

    def neg_sample(self, batch_sz, *extra):
        '''
        batch_sz: groups of samples, like the size of batch during training
        '''
        h_idx = np.random.randint(low=0, high=self.ent_num, size=(batch_sz, self.sample_num))
        t_idx = np.random.randint(low=0, high=self.ent_num, size=(batch_sz, self.sample_num))
        h_rand = torch.LongTensor(h_idx)
        t_rand = torch.LongTensor(t_idx)
        return h_rand, t_rand

class TriCacheSampler(): 
    def __init__(self, head_cache, tail_cache, N_1, N_2, n_ent, update) -> None:
        self.head_cache = head_cache
        self.tail_cache = tail_cache
        self.N_1 = N_1
        self.N_2 = N_2
        self.n_ent = n_ent
        self.update = update


    def neg_sample(self, batch_sz, extra):
        head_idx=extra[0].cpu()
        tail_idx = extra[1].cpu()
        randint = np.random.randint(low=0, high=self.N_1, size=batch_sz)
        h_idx = self.head_cache[head_idx, randint]
        t_idx = self.tail_cache[tail_idx, randint]
        h_rand = torch.LongTensor(h_idx)
        t_rand = torch.LongTensor(t_idx)
        self.update_cache(*extra)
        return h_rand, t_rand

    def update_cache(self, head, tail, rela, head_idx, tail_idx, model):
        ''' update the cache with different schemes '''
        head_idx, head_uniq = np.unique(head_idx.cpu(), return_index=True)
        tail_idx, tail_uniq = np.unique(tail_idx.cpu(), return_index=True)

        tail_h = tail[head_uniq]
        rela_h = rela[head_uniq]

        rela_t = rela[tail_uniq]
        head_t = head[tail_uniq]

        # get candidate for updating the cache
        h_cache = self.head_cache[head_idx]
        t_cache = self.tail_cache[tail_idx]
        h_cand = np.concatenate([h_cache, np.random.choice(self.n_ent, (len(head_idx), self.N_2))], 1)
        t_cand = np.concatenate([t_cache, np.random.choice(self.n_ent, (len(tail_idx), self.N_2))], 1)
        h_cand = torch.from_numpy(h_cand).type(torch.LongTensor)
        t_cand = torch.from_numpy(t_cand).type(torch.LongTensor)
        t_cand = t_cand.to(head.device)
        h_cand = h_cand.to(head.device)

        # expand for computing scores/probs
        rela_h = rela_h.unsqueeze(1).expand(-1, self.N_1 + self.N_2)
        tail_h = tail_h.unsqueeze(1).expand(-1, self.N_1 + self.N_2)
        head_t = head_t.unsqueeze(1).expand(-1, self.N_1 + self.N_2)
        rela_t = rela_t.unsqueeze(1).expand(-1, self.N_1 + self.N_2)

        h_probs = model.prob(h_cand, tail_h, rela_h)
        t_probs = model.prob(head_t, t_cand, rela_t)

        if self.update == 'IS':
            h_new = torch.multinomial(h_probs, self.N_1, replacement=False)
            t_new = torch.multinomial(t_probs, self.N_1, replacement=False)
        elif self.update == 'top':
            _, h_new = torch.topk(h_probs,  k=self.N_1, dim=-1)
            _, t_new = torch.topk(t_probs,  k=self.N_1, dim=-1)

        h_idx = torch.arange(0, len(head_idx)).type(torch.LongTensor).unsqueeze(1).expand(-1, self.N_1)
        t_idx = torch.arange(0, len(tail_idx)).type(torch.LongTensor).unsqueeze(1).expand(-1, self.N_1)
        h_rep = h_cand[h_idx, h_new]
        t_rep = t_cand[t_idx, t_new]

        self.head_cache[head_idx] = h_rep.cpu().numpy()
        self.tail_cache[tail_idx] = t_rep.cpu().numpy()
