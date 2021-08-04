import torch


class BertEmb(torch.nn.Module):
    def __init__(self, tokenizer, model, phrases) -> None:
        super().__init__()
        self.model = model
        result = tokenizer(phrases, padding = True)
        self.register_buffer('all_ids', torch.LongTensor(result['input_ids']))
        self.register_buffer('all_masks', torch.LongTensor(result['attention_mask']))

    def forward(self, input):
        id = self.all_ids[input]
        mask = self.all_masks[input]
        out = self.model(input_ids=id, attention_mask=mask)
        return out['pooler_output']

