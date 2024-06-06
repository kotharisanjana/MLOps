import torch

class Predictor:
    def __init__(self, model, model_path):
        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, test_dataloader):
        for batch in test_dataloader:
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            preds = torch.argmax(logits, dim=1)
            return preds
