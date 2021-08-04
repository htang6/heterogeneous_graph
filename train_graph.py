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
from models import TransEModule, BertEmb, SynoPred
from samplers import TriRandomSampler, PairRandomSampler
from trainer import Trainer
from utils import split_idx, heads_tails, get_logger, load_data, build_graph, get_data





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
    parser.add_argument('--config', type=str, default='default.yaml', help='name of config file')
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

    sampler = TriRandomSampler(len(graph.nodes), 2)
    connects = torch.LongTensor(graph.connects)

    head = connects[:,0]
    rela = connects[:,1]
    tail = connects[:,2]

    corrupter = BernCorrupter([head, tail, rela], len(graph.nodes), len(graph.edges))

    occurrence = heads_tails(graph.connects, len(graph.nodes))
    transE = TransEModule(len(graph.nodes), len(graph.edges), sampler, corrupter, \
    occurrence, config['graph_model']['arch'])

    train_idx, eval_idx, test_idx = split_idx(connects.shape[0])
    train_data = connects[train_idx]
    eval_data = connects[eval_idx]
    test_data = connects[test_idx]

    train_dl = DataLoader(train_data, batch_size=32)
    eval_dl = DataLoader(eval_data, batch_size=32)

    trainer = Trainer(transE, config['graph_model']['train_conf'], device=device, logger=logger)
    trainer.fit(train_dl, eval_dl, 0.001)

    torch.save(transE.ent_emb.weight, './results/c_emb.pt')



    
    
