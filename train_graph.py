import argparse

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import yaml

from bern_corrupter import BernCorrupter
from models import TransEModule, ComplExModule, ShallowModule
from samplers import TriRandomSampler, TriCacheSampler
from trainer import Trainer
from utils import heads_tails, get_logger, load_data, build_graph, split_data, get_cache_list
from test_utils import get_test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
    parser.add_argument('--config', type=str, default='config/shallow.yaml', help='name of config file')
    parser.add_argument('--model', type=str, default='ComplEx')
    parser.add_argument('--sampler', type=str, default='cache')
    parser.add_argument('--test', type=str, default='no')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('DEVICE: ', device)

    logger = get_logger('train')

    if args.test == 'no':
        clist = load_data()
        graph = build_graph(clist)
        n_ent = len(graph.nodes)
        n_rela = len(graph.edges)
        all_data = torch.LongTensor(graph.connects)
        train_data, eval_data, test_data = split_data(all_data)
    elif args.test == 'yes':
        train_data, eval_data, test_data, n_ent, n_rela = get_test_data()
        all_data = torch.cat((train_data, eval_data, test_data), dim=0)

    h_idx, t_idx, h_cache, t_cache, h_pos, t_pos = \
        get_cache_list(train_data[:,0], train_data[:,1], train_data[:,2], n_ent, 30)
    cache_idx = torch.from_numpy(np.stack((h_idx, t_idx), axis = 1))
    train_data = torch.cat((train_data, cache_idx), dim=1)
    if args.sampler == 'cache':
        sampler = TriCacheSampler(h_cache, t_cache, 30, 30, n_ent, 'IS')
    elif args.sampler == 'rand':
        sampler = TriRandomSampler(n_ent, 2)

    head = train_data[:,0]
    tail = train_data[:,1]
    rela = train_data[:,2]
    corrupter = BernCorrupter([head, tail, rela], n_ent, n_rela)

    occurrence = heads_tails(all_data, n_ent)

    if args.model == 'TransE':
        emb_model = TransEModule(n_ent, n_rela, config['arch'])
    elif args.model == 'ComplEx':
        enb_model = ComplExModule(n_ent, n_rela, config['arch'])
    trainable = ShallowModule(n_ent, enb_model, sampler, corrupter, occurrence)

    train_dl = DataLoader(train_data, batch_size=config['train_conf']['batch_sz'])
    eval_dl = DataLoader(eval_data, batch_size=128)

    trainer = Trainer(trainable, config['train_conf'], device=device, logger=logger)
    trainer.fit(train_dl, eval_dl, 0.001)

    torch.save(trainable.model.get_state(), './results/c_emb.pt')


    
    
