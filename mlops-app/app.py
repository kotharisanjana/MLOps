from flask import Flask
import os 
import mlflow
import dagshub
import hydra
from omegaconf import DictConfig, OmegaConf

from data_service.data import Data
from model_service.model import Model
from training_service.train import Trainer
from prediction_service.inference import Inference
from model_download import download_model

app = Flask(__name__)

cfg = None

dagshub.init(repo_owner='sanjanak98', repo_name='MLOps', mlflow=True)

@app.route('/')
def hello():
    return "MLOps pipeline project"

def initialize_hydra():
    global cfg
    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name="config")
        if cfg is None:
            raise ValueError("Hydra configuration loading failed, 'cfg' is None")
        
initialize_hydra()

def get_or_create_experiment_id(exp_name):
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
        return exp_id
    return exp.experiment_id

@app.route('/model-training')
def train_model():
    exp_name = "mlops"
    exp_id = get_or_create_experiment_id(exp_name)

    global cfg

    data = Data(cfg)
    data.load_training_data()
    data.convert_to_csv()
    data.prepare_training_data()
    train_dataloader = data.setup_training_dataloader()

    model = Model(cfg)

    trainer = Trainer(cfg, model, train_dataloader)
    model_uri = trainer.train_model(exp_id)
           
    config_path = os.path.join(os.getcwd(), "configs/model/default.yaml")
    custom_cfg = OmegaConf.load(config_path)
    custom_cfg.trained = DictConfig({"model_uri": model_uri})
    OmegaConf.save(custom_cfg, config_path)

    download_model()

    return "Model training completed!"

@app.route('/inference')
def inference():
    global cfg

    inferencing_instance = Inference(cfg)
    data = Data(cfg)
    data.load_testing_data()
    data.prepare_testing_data()
    test_dataloader = data.setup_testing_dataloader()

    inferencing_instance.predict(test_dataloader)

    return "Inference completed" 



