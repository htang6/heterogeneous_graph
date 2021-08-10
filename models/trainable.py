import torch

class Trainable(torch.nn.module):
    def train_step(self, idx, batch):
        pass

    def eval_step(self, idx, batch):
        pass

    def eval_sum(self, result_list):
        pass