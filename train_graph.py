from collections import defaultdict
import argparse
from os import getloadavg
from platform import node

import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer, AutoModel
import yaml

from bern_corrupter import BernCorrupter
from graph import HeteroGraph
from data_process import parse_block
from models import TransEModule, BertEmb, SynoPred, ComplExModule, ShallowModule
from samplers import TriRandomSampler, PairRandomSampler
from trainer import Trainer
from utils import split_idx, heads_tails, get_logger, load_data, build_graph, get_data





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
    parser.add_argument('--config', type=str, default='config/shallow.yaml', help='name of config file')
    parser.add_argument('--model', type=str, default='ComplEx')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('DEVICE: %s', device)

    logger = get_logger('train')

    clist = load_data()
    graph = build_graph(clist)
    n_ent = len(graph.nodes)
    n_rela = len(graph.edges)

    sampler = TriRandomSampler(n_ent, 2)

    connects = torch.LongTensor(graph.connects)
    train_idx, eval_idx, test_idx = split_idx(connects.shape[0])
    train_data = connects[train_idx]
    eval_data = connects[eval_idx]
    test_data = connects[test_idx]

    head = train_data[:,0]
    rela = train_data[:,1]
    tail = train_data[:,2]
    corrupter = BernCorrupter([head, tail, rela], n_ent, n_rela)

    occurrence = heads_tails(graph.connects, len(graph.nodes))

    sampler = TriRandomSampler(n_ent, 2)
    if args.model == 'TransE':
        emb_model = TransEModule(n_ent, n_rela, config['arch'])
    elif args.model == 'ComplEx':
        enb_model = ComplExModule(n_ent, n_rela, config['arch'])
    trainable = ShallowModule(n_ent, enb_model, sampler, corrupter, occurrence)

    train_dl = DataLoader(train_data, batch_size=config['train_conf']['batch_sz'])
    eval_dl = DataLoader(eval_data, batch_size=50)

    trainer = Trainer(trainable, config['train_conf'], device=device, logger=logger)
    trainer.fit(train_dl, eval_dl, 0.001)

    torch.save(trainable.model.get_state(), './results/c_emb.pt')
    # TODO: need to find a new way to save 


    
    
