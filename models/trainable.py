import torch

class Trainable(torch.nn.Module):
    def train_step(self, idx, batch):
        pass

    def eval_pre(self):
        pass

    def eval_step(self, idx, batch):
        pass

    def eval_sum(self, result_list):
        pass