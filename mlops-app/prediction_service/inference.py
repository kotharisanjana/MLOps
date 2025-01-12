import torch
import os 
import time
from prometheus_client import Gauge

from model_service.model import Model

ACCURACY = Gauge("prediction_accuracy", "Accuracy of the model")

class Inference:
    def __init__(self, cfg):
        self.model = Model(cfg)
        state_dict = torch.load(os.path.join(os.getcwd(), "models/model.pth"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, test_dataloader):
        for i, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            y_true = batch["label"]

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                y_pred = torch.argmax(logits, dim=1) 

                batch_acc = (y_pred == y_true).float().mean()

                print(i, batch_acc)

                ACCURACY.set(batch_acc)
            
            time.sleep(2)