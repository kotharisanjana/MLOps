import torch
import onnxruntime as ort
import numpy as np

class OnnxPredictor:
    def __init__(self, test_dataloader, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.test_dataloader = test_dataloader

    def predict(self):
        predictions = []

        for batch in self.test_dataloader:
            ort_inputs = {
            "input_ids": np.expand_dims(batch["input_ids"], axis=0),
            "attention_mask": np.expand_dims(batch["attention_mask"], axis=0),
            }
            ort_outs = self.ort_session.run(None, ort_inputs) 
            scores = torch.nn.Softmax(dim=1)(torch.tensor(ort_outs[0]))[0].tolist()     

            predicted_label = torch.argmax(torch.tensor(scores)).item()     
            predicted_score = scores[predicted_label]

            predictions.append({"label": predicted_label, "score": predicted_score})

        return predictions