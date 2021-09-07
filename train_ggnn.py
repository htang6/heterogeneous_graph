import argparse
import pickle

import torch
from torch.utils.data.dataloader import DataLoader
import yaml

from models import HGGNN, GGNN
from samplers import PairRandomSampler
from trainer import Trainer
from utils import split_idx, get_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
    parser.add_argument('--config', type=str, default='config/ggnn.yaml', help='name of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('DEVICE: %s', device)

    logger = get_logger('train')

    graph_emb = torch.load('data/graph_emb.pt')
    phrase_emb = torch.load('data/phrase_emb.pt')
    with open('data/levi_graph', 'rb') as f:
        levi_graph = pickle.load(f)
    with open('data/pc_pairs', 'rb') as f:
        pairs = pickle.load(f)
    adj_mat = levi_graph.get_adj()
    pair_sampler = PairRandomSampler(levi_graph.c_num, 2)

    ggnn = GGNN(state_dim = graph_emb.shape[1], n_edge_types=1, n_node=graph_emb.shape[0], n_steps=5)
    hggnn = HGGNN(ggnn, phrase_emb, graph_emb, adj_mat, pair_sampler, levi_graph.c_num)

    pairs = torch.LongTensor(pairs)
    train_idx, eval_idx, test_idx = split_idx(pairs.shape[0])
    train_pairs = pairs[train_idx]
    eval_pairs = pairs[eval_idx]
    test_pairs = pairs[test_idx]

    train_dl = DataLoader(train_pairs, batch_size=32)
    eval_dl = DataLoader(eval_pairs[:32], batch_size=32)

    trainer = Trainer(hggnn, config['train_conf'], device=device, logger = logger)
    trainer.fit(train_dl, eval_dl, 0.001)
