import torch


class BertEmb(torch.nn.Module):
    def __init__(self, tokenizer, model, phrases) -> None:
        super().__init__()
        self.model = model
        result = tokenizer(phrases, padding = True)
        self.register_buffer('all_ids', torch.LongTensor(result['input_ids']))
        self.register_buffer('all_masks', torch.LongTensor(result['attention_mask']))

    def forward(self, batch):
        # FIXME Why the uniq idx are 1024 when sampling?
        uniq_idx, reverse_idx = torch.unique(batch, return_inverse=True)
        sz = 512
        emb_list = []
        for i in range(0, uniq_idx.shape[0], sz):
            idx_batch = uniq_idx[i:i+sz]
            ids = self.all_ids[idx_batch]
            mask = self.all_masks[idx_batch]
            out = self.model(input_ids=ids, attention_mask=mask)
            emb_list.append(out['pooler_output'])
        uniq_emb = torch.cat(emb_list, dim=0)
        all_emb = uniq_emb[reverse_idx]
        return all_emb

