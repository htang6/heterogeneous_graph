import argparse
import pickle

import torch
from torch.utils.data.dataloader import DataLoader
import yaml
from transformers import BertTokenizer, AutoModel

from models import HGGNN, GGNN, MatchingNet, StaticEmb, FCNet, SageWrapper, BertEmb
from samplers import PairRandomSampler, PairCacheSampler
from trainer import Trainer
from utils import split_idx, get_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
    parser.add_argument('--config', type=str, default='config/ggnn.yaml', help='name of config file')
    parser.add_argument('--arch', type=int, default=1, help='the architecture of the model')
    parser.add_argument('--sampler', type=str, default='cache', help='which sampelr to use')
    parser.add_argument('--data', type=str, default='simple_graph', help = 'levi_graph, simple_graph')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('DEVICE: %s', device)

    logger = get_logger('train')

    DATA_FOLDER = args.data
    FULL_DIR = 'data/'+ DATA_FOLDER + '/'
    graph_emb = torch.load(FULL_DIR + 'graph_emb.pt')
    phrase_emb = torch.load(FULL_DIR + 'phrase_emb.pt')
    with open(FULL_DIR + 'graph', 'rb') as f:
        levi_graph = pickle.load(f)
    with open(FULL_DIR + 'pc_pairs', 'rb') as f:
        pairs = pickle.load(f)
    adj_mat = levi_graph.get_adj()

    graph_phrases = [levi_graph.allphrase[idx] for idx in levi_graph.levi_nodes]
    with open(FULL_DIR + 'phrases', 'rb') as f:
        phrases = pickle.load(f)

    print('***************Data Info**************')
    print('Total phrases: ', phrase_emb.shape[0])
    print('Total graph nodes: ', len(levi_graph.levi_nodes))
    print('Total graph edges: ', len(levi_graph.levi_edges))
    print('**************************************')

    pairs = torch.LongTensor(pairs)
    train_idx, eval_idx, test_idx = split_idx(pairs.shape[0], 5, 20)
    train_pairs = pairs[train_idx]
    eval_pairs = pairs[eval_idx]
    test_pairs = pairs[test_idx]

    if args.sampler == 'random':
        pair_sampler = PairRandomSampler(levi_graph.c_num, 4)
    elif args.sampler == 'cache':
        pair_sampler = PairCacheSampler(4, phrase_emb.shape[0], levi_graph.c_num, train_pairs)

    acc_num = 1
    if args.arch == 1:
        print("Arch: Static Bert embedding + GGNN")
        l_model = StaticEmb(phrase_emb)
        ggnn = GGNN(state_dim = graph_emb.shape[1], n_edge_types=1, n_node=graph_emb.shape[0], n_steps=5)
        r_model = HGGNN(ggnn, graph_emb, adj_mat)
    elif args.arch == 2:
        print("Arch: Static Bert embedding * 2")
        l_model = StaticEmb(phrase_emb)
        r_model = StaticEmb(graph_emb)
    elif args.arch == 3:
        print('Arch: Static Bert embedding + graphSage')
        l_model = StaticEmb(phrase_emb)
        feat_dim = graph_emb.shape[1]
        r_model = SageWrapper(2, feat_dim, feat_dim, graph_emb, levi_graph.levi_edges)
    elif args.arch == 4:
        print('Arch: Bert Model * 2')
        tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        bert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        l_model = BertEmb(tokenizer, bert, phrases)
        r_model = BertEmb(tokenizer, bert, graph_phrases)
        acc_num = 8

    emb_sz = config['arch']['emb_sz']
    input_sz = graph_emb.shape[1] + phrase_emb.shape[1]
    scorer = FCNet(*[input_sz, emb_sz, emb_sz, 2])
    model = MatchingNet(scorer, l_model, r_model, pair_sampler, levi_graph.c_num)

    if args.sampler == 'cache':
        pair_sampler.model = model

    b_sz = config['train_conf']['batch_sz']
    lr = config['train_conf']['lr']
    train_dl = DataLoader(train_pairs, batch_size=b_sz)
    eval_dl = DataLoader(eval_pairs, batch_size=32)

    trainer = Trainer(model, config['train_conf'], device=device, logger=logger, acc_num=acc_num)
    trainer.fit(train_dl, eval_dl, lr)
