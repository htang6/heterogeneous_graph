
from pprint import pformat
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, model, conf, device, logger) -> None:
        self.model = model.to(device)
        self.conf = conf
        self.device = device
        self.logger = logger
        self.tb_writer = SummaryWriter('default')

    def fit(self, train_dl, eval_dl, lr):
        self.logger.info('Trainer: start training...')
        b_sz = train_dl.batch_size
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.zero_grad()
        self.model.train()
        for epoch in range(self.conf['epoch_num']):
            for idx, batch in enumerate(train_dl):
                batch = batch.to(self.device)
                loss = self.model.train_step(idx, batch)
                if (idx + 1) % self.conf['print_interval'] == 0:
                    self.logger.info('epoch %d batch %d loss %.3f', epoch, idx, loss.item())
                    total_idx = epoch*b_sz+idx
                    self.tb_writer.add_scalar('loss', total_idx, loss.item())
                    metrics = self.eval(eval_dl)
                    self.logger.info('Evaluation result: \n' + pformat(metrics))
                    for k,v in metrics.items():
                        self.tb_writer.add_scalar(k, total_idx, v)
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
        self.logger.info('Trainer: end training...')

    def eval(self, eval_dl):
        eval_list = []
        self.model.eval()
        with torch.no_grad():
            self.model.eval_pre()
            for idx, batch in enumerate(eval_dl):
                batch = batch.to(self.device)
                result  = self.model.eval_step(idx, batch)
                eval_list.append(result)
            metrics = self.model.eval_sum(eval_list)
        self.model.train()
        return metrics
