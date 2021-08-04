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

    # We then train binary classification model to match concept and phrase
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    pair_sampler = PairRandomSampler(len(graph.nodes), 2)

    phrases, pairs = get_data(graph)
    bert_emb = BertEmb(tokenizer, model, phrases)

    embeddings = torch.load('./results/c_emb.pt')
    c_emb = torch.nn.Embedding.from_pretrained(embeddings)
    # Fixing the parameters
    for parameter in c_emb.parameters():
        parameter.requires_grad = False
    
    pred_model = SynoPred(c_emb, bert_emb, pair_sampler, **config['pred_model']['arch'])

    pairs = torch.LongTensor(pairs)
    train_idx, eval_idx, test_idx = split_idx(pairs.shape[0])
    train_pairs = pairs[train_idx]
    eval_pairs = pairs[eval_idx]
    test_pairs = pairs[test_idx]

    train_dl = DataLoader(train_pairs, batch_size=32)
    eval_dl = DataLoader(eval_pairs[:32], batch_size=32)

    trainer = Trainer(pred_model, config['pred_model']['train_conf'], device=device, logger = logger)
    trainer.fit(train_dl, eval_dl, 0.001)





    
    
