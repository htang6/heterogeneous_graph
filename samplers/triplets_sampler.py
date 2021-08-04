import numpy as np
import torch

class TriRandomSampler:
    def __init__(self, ent_num, sample_num) -> None:
        self.ent_num = ent_num
        self.sample_num = sample_num

    def neg_sample(self, entry_num):
        '''
        entry_num: groups of samples, like the size of batch during training
        '''
        h_idx = np.random.randint(low=0, high=self.ent_num, size=(entry_num, self.sample_num))
        t_idx = np.random.randint(low=0, high=self.ent_num, size=(entry_num, self.sample_num))
        h_rand = torch.LongTensor(h_idx)
        t_rand = torch.LongTensor(t_idx)
        return h_rand, t_rand
