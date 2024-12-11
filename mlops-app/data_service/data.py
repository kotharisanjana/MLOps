import torch
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer
import pandas as pd

class Data(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.dataset = datasets.load_dataset("glue", cfg.data.dataset)
        self.train_batch_size = cfg.data.configuration.train_batch_size
        self.test_batch_size = cfg.data.configuration.test_batch_size
        self.max_length = cfg.data.configuration.max_length
        self.train_size = cfg.data.size.train_size
        self.load_tokenizer(cfg)

    def load_tokenizer(self, cfg):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained.tokenizer, cache_dir="/tmp/transformers_cache")

    def load_training_data(self):
        self.train_data = self.dataset["train"].select(range(self.train_size))

    def load_testing_data(self):
        self.test_data = self.dataset["validation"]
        
    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def prepare_training_data(self):
        self.train_data = self.train_data.map(self.tokenize_data, batched=True)
        self.train_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

    def prepare_testing_data(self):
        self.test_data = self.test_data.map(self.tokenize_data, batched=True)
        self.test_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

    def setup_training_dataloader(self):
        train_dataloader = DataLoader(
            self.train_data, batch_size=self.train_batch_size, shuffle=True
        )

        return train_dataloader
    
    def setup_testing_dataloader(self):
        test_dataloader = DataLoader(
            self.test_data, batch_size=self.test_batch_size
        )
        return test_dataloader
    
    def convert_to_csv(self):
        train_data_pandas = pd.DataFrame(self.train_data)
        train_data_pandas.to_csv("train_data.csv", index=False)