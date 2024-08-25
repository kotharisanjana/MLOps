import torch
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer
import pandas as pd

# https://huggingface.co/datasets/nyu-mll/glue/viewer/cola

class Data(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.dataset = cfg.data.dataset
        self.batch_size = cfg.data.configuration.batch_size
        self.max_length = cfg.data.configuration.max_length
        self.train_size = cfg.data.size.train_size
        self.val_size = cfg.data.size.val_size
        self.load_tokenizer(cfg)

    def load_tokenizer(self, cfg):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained.tokenizer, cache_dir="/tmp/transformers_cache")

    def load_data(self):
        dataset = datasets.load_dataset("glue", self.dataset)
        self.train_data = dataset["train"].select(range(self.train_size))
        self.val_data = dataset["validation"].select(range(self.val_size))
        
    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def prepare_data(self):
        self.train_data = self.train_data.map(self.tokenize_data, batched=True)
        self.train_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        self.val_data = self.val_data.map(self.tokenize_data, batched=True)
        self.val_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

    def setup_dataloaders(self):
        train_dataloader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

        val_dataloader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=True
        )

        return train_dataloader, val_dataloader
    
    def convert_to_csv(self):
        train_data_pandas = pd.DataFrame(self.train_data)
        train_data_pandas.to_csv("train_data.csv", index=False)

        val_data_pandas = pd.DataFrame(self.val_data)
        val_data_pandas.to_csv("val_data.csv", index=False)