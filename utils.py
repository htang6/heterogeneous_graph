from collections import defaultdict
import logging

import torch
import numpy as np

from graph import HeteroGraph
from data_process import parse_block

# Here the scores actually means distance, the smaller is regarded higher ranked.
def mrr_mr_hitk(scores, target, k=[1,3,10], descend = False):
    _, sorted_idx = torch.sort(scores, descending=descend)
    find_target = sorted_idx == target
    target_rank = torch.nonzero(find_target)[0, 0] + 1
    hits = np.zeros((len(k),))
    for i, r in enumerate(k):
        hits[i] = int(target_rank <= r)
    return 1. / float(target_rank), target_rank, hits

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    start = 0
    while(start < n_sample):
        end = min(n_sample, start + batch_size)
        ret = [ls[start:end] for ls in lists]
        start += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def split_idx(num_nodes, test_split = 5, val_split = 10):
    rand_indices = np.random.permutation(num_nodes)

    test_size = num_nodes // test_split
    val_size = num_nodes // val_split

    test_indexs = rand_indices[:test_size]
    val_indexs = rand_indices[test_size:(test_size+val_size)]
    train_indexs = rand_indices[(test_size+val_size):]
    
    return train_indexs, val_indexs, test_indexs

# Please test this funciton
def heads_tails(conn_list, n_ent):
    all_heads = [conn[0] for conn in conn_list]
    all_relas = [conn[1] for conn in conn_list]
    all_tails = [conn[2] for conn in conn_list]

    heads = defaultdict(lambda: set())
    tails = defaultdict(lambda: set())
    for h, t, r in zip(all_heads, all_tails, all_relas):
        tails[(h, r)].add(t)
        heads[(t, r)].add(h)


    heads_sp = {}
    tails_sp = {}
    for k in heads.keys():
        heads_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(heads[k])]),
                                                torch.ones(len(heads[k])), torch.Size([n_ent]))

    for k in tails.keys():
        tails_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(tails[k])]),
                                                torch.ones(len(tails[k])), torch.Size([n_ent]))
    print("heads/tails size:", len(tails), len(heads))

    return heads_sp, tails_sp

def get_logger(name):
    logger = logging.getLogger('name')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fh = logging.FileHandler('logs/'+name+'.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def load_data():
    f = open('data/cl.obo.txt', 'r')
    content = f.read()
    term_blocks = content.split('\n\n')
    cinfo_list = []
    for term_block in term_blocks:
        cinfo = parse_block(term_block)
        if cinfo != None:
            cinfo_list.append(parse_block(term_block))

    return cinfo_list

def build_graph(clist):
    ent2idx = {}
    rela2idx = {'is_a':0, 'disjoint_from':1}

    for idx, c in enumerate(clist):
        ent2idx[c['cid']] = idx

    for c in clist:
        for relation in c['main_relations']:
            if relation[0] not in rela2idx:
                rela2idx[relation[0]] = len(rela2idx)

    ent_list = []
    rela_list = ['is_a', 'disjoint_from']
    connections = []

    for c in clist:
        if c['cid'] in rela2idx:
            rela_list.append(c)
        elif c['cid'] in ent2idx:
            ent_list.append(c)
        for relation in c['main_relations']:
            h = ent2idx[c['cid']]
            r = rela2idx[relation[0]]
            t = ent2idx[relation[1]]
            connections.append([h, r, t])
    
    return HeteroGraph(ent_list, rela_list, connections, ent2idx)

def get_data(graph):
    plist = []
    for ent in graph.nodes:
        for phrase in ent['synonyms'][0]:
            plist.append(phrase)
        for phrase in ent['synonyms'][1]:
            plist.append(phrase)

    p2idx = {}
    for i, p in enumerate(plist):
        p2idx[p] = i

    c2idx = graph.node2idx

    pos_pairs = []
    for ent in graph.nodes:
        for phrase in ent['synonyms'][0]:
            pos_pairs.append([p2idx[phrase] ,c2idx[ent['cid']]])

    return plist, pos_pairs