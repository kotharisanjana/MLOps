import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class Data(torch.utils.data.Dataset):
    def __init__(self, params):
        self.batch_size = params["batch_size"]
        self.tokenizer = AutoTokenizer.from_pretrained(params["model_name"])

    def load_data(self):
        dataset = load_dataset("glue", "cola")
        self.train_data = dataset["train"].select(range(32))
        self.val_data = dataset["validation"].select(range(32))
        self.test_data = dataset["test"].select(range(32))
        
    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
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

        self.test_data = self.test_data.map(self.tokenize_data, batched=True)
        self.test_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

    def setup_train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def setup_val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )
    
    def setup_test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False
        )