import torch

from cola_prediction.model import Model
from cola_prediction.data import Data  

class Inference:
    def __init__(self, cfg):
        self.model_dvc_file = cfg.inference.model_dvc_file
        self.local_model_path = cfg.inference.local_model_path
        self.initialize_model(cfg)

    def initialize_model(self, cfg):
        self.tokenizer = Data(cfg)
        self.model = Model(cfg)

    def load_model(self):
        state_dict = torch.load(self.local_model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, inference_sample):
        tokenized_sample = self.tokenizer.tokenize_data(inference_sample)
        logits = self.model(
            tokenized_sample["input_ids"], 
            tokenized_sample["attention_mask"]
        )
        predicted_label = torch.argmax(logits, dim=1)
        return predicted_label