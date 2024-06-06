import torch.nn as nn
from transformers import AutoModel


class Model(nn.Module):
    def __init__(self, model_name):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.model.config.hidden_size, 2)
        self.num_classes = 2

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits
