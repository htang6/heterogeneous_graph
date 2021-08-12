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
from test_utils import get_test_data

def dummy_graph():
    nodes = range(0,10)
    edge = [0]
    connects = []
    for i in range(0,5):
        connects.append([0, 0, i])
    
    return HeteroGraph(nodes, edge, connects, None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
    parser.add_argument('--config', type=str, default='config/shallow_test.yaml', help='name of config file')
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

    train_data, val_data, test_data, n_ent, n_rela = get_test_data()
    all_data = torch.cat((train_data, val_data, test_data), dim=0)

    sampler = TriRandomSampler(n_ent, 1)

    head = train_data[:,0]
    rela = train_data[:,1]
    tail = train_data[:,2]

    corrupter = BernCorrupter([head, tail, rela], n_ent, n_rela)

    occurrence = heads_tails(all_data, n_ent)
    if args.model == 'TransE':
        model = TransEModule(n_ent, n_rela, sampler, corrupter, \
        occurrence, config['arch'])
    elif args.model == 'ComplEx':
        enb_model = ComplExModule(n_ent, n_rela, config['arch'])
        model = ShallowModule(n_ent, enb_model, sampler, corrupter, occurrence)

    train_dl = DataLoader(train_data, batch_size=4096)
    eval_dl = DataLoader(val_data, batch_size=128)

    trainer = Trainer(model, config['train_conf'], device=device, logger=logger)
    trainer.fit(train_dl, eval_dl, 0.001) 

    # torch.save(model.ent_emb.weight, './results/c_emb.pt')