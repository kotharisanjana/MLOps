import torch
import mlflow

class Predictor:
    def __init__(self, model_uri):
        self.loaded_model = mlflow.pytorch.load_model(model_uri)

    def predict(self, test_dataloader):
        for batch in test_dataloader:
            logits = self.loaded_model(batch["input_ids"], batch["attention_mask"])
            preds = torch.argmax(logits, dim=1)
            return preds
