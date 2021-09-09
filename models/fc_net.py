import torch.nn as nn
import torch

class FCNet(nn.Module):
    def __init__(self, *size_list, dropout=0.5):
        super().__init__()
        sz = len(size_list)
        assert sz > 2, "should have at least on hiddne layers!"
        layers = []

        #build hidden layers
        for i in range(sz-3):
            layers.append(nn.Linear(size_list[i], size_list[i+1]))
            layers.append(nn.ReLU())

        #build the last two layers
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(size_list[sz-3], size_list[sz-2]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(size_list[sz-2], size_list[sz-1]))
        self.model = nn.Sequential(*layers)

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        out = self.model(embeds)
        probs = torch.log_softmax(out, -1)
        return probs