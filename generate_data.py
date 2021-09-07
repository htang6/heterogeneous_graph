import pickle
from tqdm import tqdm

import torch
from transformers import BertTokenizer, AutoModel

from graph import LeviGraph
from utils import get_logger, load_data, build_graph, get_data

def bert_emb(tokenizer, bert, phrase_list, device):
    all_emb = []
    batch_size = 256
    batch_num = int(len(phrase_list)/batch_size) + 1
    '''
    model.eval() will not disable the gradient calculation, but only change the forward
    behavior of some layers. torch.no_grad() will disable the gradient calculation and
    should be call during testing time
    '''
    with torch.no_grad():
        for i in tqdm(range(batch_num)):
            batch = phrase_list[i*batch_size:(i+1)*batch_size]
            token = tokenizer(batch, padding = True)
            id = torch.LongTensor(token['input_ids']).to(device)
            mask = torch.LongTensor(token['attention_mask']).to(device)
            result = bert(id, mask)
            all_emb.append(result['pooler_output'].cpu())
        
        result = torch.cat(all_emb, dim=0)
    return result

if __name__ == '__main__':
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
    bert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

    phrases, pairs = get_data(graph)
    levi_graph = LeviGraph(graph)
    graph_phrase = levi_graph.allphrase

    adj_mat = levi_graph.get_adj()
    print('start generate bert embeddings...')
    bert.to(device)
    phrase_emb = bert_emb(tokenizer, bert, phrases, device)
    graph_emb = bert_emb(tokenizer, bert, graph_phrase, device)
    graph_emb = graph_emb[levi_graph.levi_nodes]
    print('end generate bert embeddings')

    torch.save(graph_emb.cpu(), 'data/graph_emb.pt')
    torch.save(phrase_emb.cpu(), 'data/phrase_emb.pt')
    with open('data/levi_graph', 'wb') as f:
        pickle.dump(levi_graph, f)
    with open('data/pc_pairs', 'wb') as f:
        pickle.dump(pairs, f)


