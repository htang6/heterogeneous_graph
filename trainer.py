import time
from pprint import pformat
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, model, conf, device, logger, acc_num = 1) -> None:
        self.model = model.to(device)
        self.conf = conf
        self.device = device
        self.logger = logger
        self.tb_writer = SummaryWriter('default')
        self.acc_num = acc_num

    def fit(self, train_dl, eval_dl, lr):
        self.logger.info('Trainer: start training...')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.zero_grad()
        self.model.train()
        epoch_num = self.conf['epoch_num']
        for epoch in range(epoch_num):
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epoch_num))
            loss_list = []
            t_start = time.time()
            for idx, batch in enumerate(train_dl):
                batch = batch.to(self.device)
                loss = self.model.train_step(idx, batch)
                self.model.zero_grad()
                loss.backward()
                # Accumulate gradient
                if (idx + 1) % self.acc_num == 0:
                    optimizer.step()
                loss_list.append(loss.detach())

            train_time = time.time() - t_start
            avg_loss = torch.mean(torch.stack(loss_list))
            self.logger.info('[Train] Average loss: %.3f', avg_loss.item())
            self.logger.info('[Train] Time: %d', train_time)

            t_start = time.time()
            metrics = self.eval(eval_dl)
            eval_time = time.time() - t_start
            self.logger.info('[Evaluation] time: %d', eval_time)
            self.logger.info('[Evaluation] result: \n' + pformat(metrics))
            for k,v in metrics.items():
                self.tb_writer.add_scalar(k, v)
                
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
