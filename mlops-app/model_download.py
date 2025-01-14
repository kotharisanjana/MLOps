import mlflow
import torch
import os
from omegaconf import OmegaConf

mlflow.set_tracking_uri("") # set Dagshub MLflow URL

def load_model_uri_from_config():
    config_path = os.path.join(os.getcwd(), "configs/model/default.yaml")
    cfg = OmegaConf.load(config_path)
    return cfg.trained.model_uri

def download_model():
    model_uri = load_model_uri_from_config()
    model = mlflow.pytorch.load_model(model_uri)
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "models/model.pth"))