import torch
import mlflow

from cola_prediction.model import Model
from cola_prediction.data import Data  

mlflow.set_tracking_uri("http://localhost:5000")

class Inference:
    def __init__(self, cfg, model_uri):
        self.tokenizer = Data(cfg)
        self.model = mlflow.pytorch.load_model(model_uri)
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