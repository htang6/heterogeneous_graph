import random
import torch
import numpy as np

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
