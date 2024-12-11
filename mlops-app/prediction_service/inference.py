import torch
import os 
from model_service.model import Model

class Inference:
    def __init__(self, cfg):
        self.model = Model(cfg)
        state_dict = torch.load(os.path.join(os.getcwd(), "models/model.pth"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, test_dataloader):
        predictions = []

        for batch in test_dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
        
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                predicted_labels = torch.argmax(logits, dim=1)  

            predictions.extend(predicted_labels)