import os 
import mlflow
import dagshub
import hydra
from omegaconf import DictConfig, OmegaConf

from cola_prediction.data import Data
from cola_prediction.model import Model
from cola_prediction.train import Trainer

dagshub.init(repo_owner='sanjanak98', repo_name='MLOps', mlflow=True)

def get_or_create_experiment_id(exp_name):
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
        return exp_id
    return exp.experiment_id

@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    exp_name = "mlops"
    exp_id = get_or_create_experiment_id(exp_name)

    data = Data(cfg)
    data.load_data()
    data.convert_to_csv()
    data.prepare_data()
    train_dataloader, val_dataloader = data.setup_dataloaders()

    model = Model(cfg)

    trainer = Trainer(cfg, model, train_dataloader, val_dataloader)
    model_uri = trainer.train_model(exp_id)
    
    config_path = os.path.join(hydra.utils.get_original_cwd(), "configs/model/default.yaml")
    custom_cfg = OmegaConf.load(config_path)
    custom_cfg.trained = DictConfig({"model_uri": model_uri})
    OmegaConf.save(custom_cfg, config_path)


if __name__ == "__main__":
    main()
