import torch
import mlflow

class Predictor:
    def __init__(self, model_uri, test_dataloader):
        self.loaded_model = mlflow.pytorch.load_model(model_uri)
        self.test_dataloader = test_dataloader

    def predict(self):
        predictions = []

        for batch in self.test_dataloader:
            logits = self.loaded_model(
                batch["input_ids"], 
                batch["attention_mask"]
            )
            predicted_label = torch.argmax(logits, dim=1)
            scores = torch.nn.Softmax(dim=0)(logits[0]).tolist()
        
            for score, label in zip(scores, predicted_label):
                predictions.append({"label": label, "score": score})
        return predictions