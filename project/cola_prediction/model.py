import torch.nn as nn
from transformers import AutoModel


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(cfg.model.pretrained.model)
        self.W = nn.Linear(self.model.config.hidden_size, cfg.model.output_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits
