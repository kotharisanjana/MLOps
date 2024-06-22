import torch
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer
import mlflow

# https://huggingface.co/datasets/nyu-mll/glue/viewer/cola

class Data(torch.utils.data.Dataset):
    def __init__(self, params):
        self.batch_size = params["batch_size"]
        self.tokenizer = AutoTokenizer.from_pretrained(params["model_name"])

    def load_data(self):
        dataset = datasets.load_dataset("glue", "cola")
        self.train_data = dataset["train"].select(range(128))
        self.val_data = dataset["validation"].select(range(32))
        self.test_data = dataset["test"].select(range(32))
        
    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def prepare_logging_data(self):
        train_dataset = mlflow.data.huggingface_dataset.from_huggingface(self.train_data, "train_data")
        val_dataset = mlflow.data.huggingface_dataset.from_huggingface(self.val_data, "val_data")
        test_dataset = mlflow.data.huggingface_dataset.from_huggingface(self.test_data, "test_data")
        return train_dataset, val_dataset, test_dataset

    def prepare_modeling_data(self):
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

    def setup_dataloaders(self):
        train_dataloader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

        val_dataloader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=True
        )
    
        test_dataloader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False
        )

        return train_dataloader, val_dataloader, test_dataloader