import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import mlflow

from tracking import logging

class Trainer():
    def __init__(self, model, params, train_dataloader, val_dataloader, train_dataset, val_dataset):
        self.model = model
        self.params = params
        self.num_epochs = self.params["num_epochs"]  
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")

    def train_model(self):
        if mlflow.active_run() is not None:
            mlflow.end_run()

        with mlflow.start_run():
            logging.log_experiment(self.params, self.train_dataset, self.val_dataset)
            return self.run_training_loop(self.train_dataloader, self.val_dataloader)

    def run_training_loop(self, train_dataloader, val_dataloader):
        for epoch in tqdm(range(self.num_epochs)):
            all_preds, all_labels = [], []
            train_loss = 0

            self.model.train()

            for batch in train_dataloader:
                self.optimizer.zero_grad()

                logits = self.model(batch["input_ids"], batch["attention_mask"])
                loss = self.loss_fn(logits, batch["label"])
                train_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["label"].cpu().numpy())

                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / len(train_dataloader)
            train_acc = accuracy_score(all_labels, all_preds)
            val_loss, val_acc = self.evaluate_model(val_dataloader)

            logging.log_training_metrics(train_loss, train_acc, val_loss, val_acc, epoch)
            
            if val_loss < self.best_val_loss:
                print(epoch)
                self.best_val_loss = val_loss
                model_info = logging.log_model(self.model)

        return model_info.model_uri

    def evaluate_model(self, val_dataloader):
        val_loss = 0
        all_preds = []
        all_labels = []

        self.model.eval()

        with torch.no_grad():
            for batch in val_dataloader:
                logits = self.model(batch["input_ids"], batch["attention_mask"])
                loss = self.loss_fn(logits, batch["label"])

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["label"].cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        val_acc = accuracy_score(all_labels, all_preds)
        return avg_val_loss, val_acc
    
    

