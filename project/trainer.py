import os 
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from cola_prediction.data import Data
from cola_prediction.model import Model
from cola_prediction.train import Trainer

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlflow_experiment")

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    data = Data(cfg)
    data.load_data()
    train_dataset, val_dataset = data.prepare_logging_data()
    data.prepare_modeling_data()
    train_dataloader, val_dataloader = data.setup_dataloaders()

    model = Model(cfg)

    trainer = Trainer(cfg, model, train_dataloader, val_dataloader, train_dataset, val_dataset)
    model_uri = trainer.train_model()
    
    config_path = os.path.join(hydra.utils.get_original_cwd(), "configs/model/default.yaml")
    custom_cfg = OmegaConf.load(config_path)
    custom_cfg.trained = DictConfig({"model_uri": model_uri})
    OmegaConf.save(custom_cfg, config_path)


if __name__ == "__main__":
    main()

    
