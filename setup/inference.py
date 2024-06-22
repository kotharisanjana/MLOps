import torch
import mlflow

class Predictor:
    def __init__(self, model_uri, test_dataloader):
        self.loaded_model = mlflow.pytorch.load_model(model_uri)
        self.test_dataloader = test_dataloader

    def predict(self):
        for batch in self.test_dataloader:
            logits = self.loaded_model(batch["input_ids"], batch["attention_mask"])
            preds = torch.argmax(logits, dim=1)
            return preds
