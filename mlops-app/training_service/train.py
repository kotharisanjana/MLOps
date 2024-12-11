import torch
from tqdm import tqdm
import mlflow
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tracking_service import logging

class Trainer():
    def __init__(self, cfg, model, train_dataloader, val_dataloader):
        self.model = model
        self.params_to_log = dict(cfg)
        self.num_epochs = cfg.training.num_epochs 

        optimizer_config = OmegaConf.to_container(cfg.training.optimizer, resolve=True)
        optimizer_config['params'] = self.model.parameters()
        self.optimizer = instantiate(optimizer_config)
        self.loss_fn = instantiate(cfg.training.loss)
        self.metric = instantiate(cfg.training.metric.accuracy)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
       
    def train_model(self, exp_id):
        with mlflow.start_run(experiment_id=exp_id):
            logging.log_parameters(self.params_to_log) 
            logging.log_dataset()
            return self.training_loop(self.train_dataloader, self.val_dataloader)

    def training_loop(self, train_dataloader):
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
            train_acc = self.metric(torch.tensor(all_labels), torch.tensor(all_preds))
            
            logging.log_training_metrics(train_loss, train_acc, epoch)

        model_info = logging.log_model(self.model)

        return model_info.model_uri
    
    

