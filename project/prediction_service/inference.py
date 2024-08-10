import torch
import os 
from cola_prediction.model import Model
from cola_prediction.data import Data  


class Inference:
    def __init__(self, cfg):
        self.tokenizer = Data(cfg)
        self.model = Model(cfg)
        state_dict = torch.load(os.path.join(os.getcwd(), "models/model.pth"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, inference_sample):
        tokenized_sample = self.tokenizer.tokenize_data(inference_sample)
        
        input_ids = torch.tensor(tokenized_sample["input_ids"])
        attention_mask = torch.tensor(tokenized_sample["attention_mask"])

        with torch.no_grad():
            logits = self.model(
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0)
            )
            predicted_label = torch.argmax(logits, dim=1)

        return predicted_label