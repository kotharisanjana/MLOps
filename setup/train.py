import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class Trainer():
    def __init__(self, model, lr):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")

    def train_model(self, train_dataloader, val_dataloader, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            train_loss = 0

            for batch in train_dataloader:
                self.optimizer.zero_grad()

                logits = self.model(batch["input_ids"], batch["attention_mask"])
                loss = self.loss_fn(logits, batch["label"])

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_dataloader)

            val_loss, val_acc = self.evaluate_model(val_dataloader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            self.save_best_model(self.model, val_loss)

    def evaluate_model(self, val_dataloader):
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

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
    
    def save_best_model(self, model, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(model.state_dict(), "././models/best_model.pth")

