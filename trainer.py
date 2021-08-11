
from pprint import pformat
import torch

class Trainer():
    def __init__(self, model, conf, device, logger) -> None:
        self.model = model.to(device)
        self.conf = conf
        self.device = device
        self.logger = logger

    def fit(self, train_dl, eval_dl, lr):
        self.logger.info('Start training...')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.zero_grad()
        self.model.train()
        for epoch in range(self.conf['epoch_num']):
            for idx, batch in enumerate(train_dl):
                batch = batch.to(self.device)
                loss = self.model.train_step(idx, batch)
                if (idx + 1) % self.conf['print_interval'] == 0:
                    self.logger.info('epoch %d batch %d loss %.3f', epoch, idx, loss.item())
                    metrics = self.eval(eval_dl)
                    self.logger.info('Evaluation result: \n' + pformat(metrics))
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
        self.logger.info('End training...')

    def eval(self, eval_dl):
        eval_list = []
        self.model.eval()
        for idx, batch in enumerate(eval_dl):
            batch = batch.to(self.device)
            result  = self.model.eval_step(idx, batch)
            eval_list.append(result)
        metrics = self.model.eval_sum(eval_list)
        self.model.train()
        return metrics
