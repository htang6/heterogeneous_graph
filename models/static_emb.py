import torch
class StaticEmb(torch.nn.Module):
    def __init__(self, emb) -> None:
        super().__init__()
        self.register_buffer('emb', emb)
    
    def forward(self, idx):
        return self.emb[idx]
    