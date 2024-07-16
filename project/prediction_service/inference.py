import torch
import dvc.api
from transformers import AutoTokenizer, AutoModel

class Inference:
    def __init__(self, cfg):
        self.dvc_repo = cfg.inference.dvc_repo
        self.model_dvc_file = cfg.inference.model_dvc_file
        self.local_model_path = cfg.inference.local_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
        self.model = AutoModel.from_pretrained(cfg.model.model)
        self.max_length = cfg.data.max_length

    def download_latest_model(self):
        with dvc.api.open(self.model_dvc_file, self.dvc_repo) as f:
            with open(self.local_model_path, 'wb') as model_file:
                model_file.write(f.read())
        state_dict = torch.load(self.local_model_path, torch.device("cpu"))
        self.model.load_state_dict(state_dict)  
        self.model.eval()  

    def predict(self, inference_sample):
        tokenized_sample = self.tokenizer(
            inference_sample,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        logits = self.model(
            tokenized_sample["input_ids"], 
            tokenized_sample["attention_mask"]
        )
        logits = outputs.last_hidden_state
        predicted_label = torch.argmax(logits, dim=1)
        return predicted_label