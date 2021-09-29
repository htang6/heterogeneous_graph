import random
import numpy as np
import torch

class PairRandomSampler():
    def __init__(self, c_num, sample_sz) -> None:
        self.c_num = c_num
        self.sample_sz = sample_sz

    def sample(self, data):
        negative_list = []
        for pair in data:
            pid = pair[0].cpu().item()
            cid = pair[1].cpu().item()
            candidates = list(range(cid)) + list(range(cid+1, self.c_num))
            sample_idx_list = random.sample(candidates, self.sample_sz)
            for sample_idx in sample_idx_list:
                negative_list.append([pid, sample_idx])
        return np.asarray(negative_list)

    # def sample_text(self, data):
    #     negative_list = []
    #     for pair in data:
    #         cid = pair[1].cpu().item()
    #         candidates = list(range(cid)) + list(range(cid+1, self.c_num))
    #         sample_idx_list = random.sample(candidates, self.sample_sz)
    #         negative_list.append(sample_idx_list)
    #     return np.asarray(negative_list)

class PairCacheSampler():
    def __init__(self, sample_sz, p_num, c_num, pos_pairs) -> None:
        self.cache = None
        self.sample_sz = sample_sz
        self.cache_sz = sample_sz*2
        pos_pairs = pos_pairs.cpu().numpy()
        self.p2c = {pair[0]:pair[1] for pair in pos_pairs}
        self.c_num = c_num
        self.p_num = p_num
        self.model = None
        self._init_cache()

    def _init_cache(self):
        cache = [None] * self.p_num
        for pid in range(self.p_num):
            if pid in self.p2c:
                cid = self.p2c[pid]
                candidates = list(range(cid)) + list(range(cid+1, self.c_num))
                sample_idx_list = random.sample(candidates, self.cache_sz)
            else:
                sample_idx_list = [0]*self.cache_sz
            cache[pid] = sample_idx_list
        self.cache = np.asarray(cache)

    #Whether I should call update_cache after or before the sample
    def sample(self, data):
        neg_list = []
        for pair in data:
            pid = pair[0].cpu().item()
            sample_idx_list = np.random.choice(self.cache[pid], self.sample_sz, replace=False)
            for sample_idx in sample_idx_list:
                neg_list.append([pid, sample_idx])
        self._updata_cache(data)
        return np.asarray(neg_list)

    # def sample_test(self, data):
    #     neg_list = []
    #     for pair in data:
    #         pid = pair[0].cpu().item()
    #         sample_idx_list = np.random.choice(self.cache[pid], self.sample_sz, replace=False)
    #         neg_list.append(sample_idx_list)
    #     self._updata_cache(data)
    #     return np.asarray(neg_list)

    def _updata_cache(self, data):
        pid_mat = data[:, 0]
        cache_to_update = self.cache[pid_mat.cpu()]

        all_cands = []
        for pair in data:
            cid = pair[1].cpu().item()
            candidates = list(range(cid)) + list(range(cid+1, self.c_num))
            sample_idx_list = random.sample(candidates, self.cache_sz)
            all_cands.append(sample_idx_list)
        all_cands = np.asarray(all_cands)
        expanded_cache = np.concatenate([cache_to_update, all_cands], axis=1)
        expanded_cache = torch.from_numpy(expanded_cache).to(pid_mat.device)

        expanded_pid = pid_mat.unsqueeze(1).expand(pid_mat.shape[0], self.cache_sz*2)
        with torch.no_grad():
            scores = self.model(expanded_pid, expanded_cache)
            probs = torch.exp(scores[:, :, 1])
            sampled_idx = torch.multinomial(probs, self.cache_sz, replacement=False)
            cache = torch.gather(expanded_cache, dim=1, index=sampled_idx)
            self.cache[pid_mat.cpu()] = cache.cpu().numpy()






