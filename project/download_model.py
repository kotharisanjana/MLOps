import mlflow
import torch
import os
from omegaconf import OmegaConf

# bridge network IP so that docker container can access the host machine
# mlflow.set_tracking_uri("http://172.17.0.1:5000")
# mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_tracking_uri("https://dagshub.com/sanjanak98/MLOps.mlflow")

def load_model_uri_from_config():
    config_path = os.path.join(os.getcwd(), "configs/model/default.yaml")
    cfg = OmegaConf.load(config_path)
    return cfg.trained.model_uri

def download_model(model_uri):
    model = mlflow.pytorch.load_model(model_uri)
    torch.save(model.state_dict(), os.path.join(os.getcwd(),  "models/model.pth"))

if __name__ == "__main__":
    model_uri = load_model_uri_from_config()
    download_model(model_uri)