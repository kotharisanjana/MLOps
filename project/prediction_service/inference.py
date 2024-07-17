import torch
import yaml
import boto3
from transformers import AutoTokenizer

from cola_prediction.model import Model

class Inference:
    def __init__(self, cfg):
        self.dvc_repo = cfg.inference.dvc_repo
        self.model_dvc_file = cfg.inference.model_dvc_file
        self.local_model_path = cfg.inference.local_model_path
        self.max_length = cfg.data.max_length
        self.s3_bucket = cfg.inference.s3_bucket
        self.initialize_model(cfg)

    def initialize_model(self, cfg):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
        self.model = Model(cfg)

    def get_model_md5(self):
        with open(self.model_dvc_file, 'r') as f:
            dvc_content = yaml.safe_load(f)
        md5_hash = dvc_content['outs'][0]['md5']
        return md5_hash

    def download_latest_model_from_s3(self):
        md5_hash = self.get_model_md5()
        s3_key = f"files/{md5_hash[:2]}/{md5_hash[2:]}/model.pth"
        s3 = boto3.client('s3')
        s3.download_file(self.s3_bucket, s3_key, self.local_model_path)
        state_dict = torch.load(self.local_model_path, map_location=torch.device('cpu'))
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
        predicted_label = torch.argmax(logits, dim=1)
        return predicted_label