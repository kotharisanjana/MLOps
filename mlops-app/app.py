from flask import Flask
import threading
import os 
import mlflow
import dagshub
import hydra
import redis
from datetime import datetime, timedelta
from omegaconf import DictConfig, OmegaConf

from data_service.data import Data
from model_service.model import Model
from training_service.train import Trainer
from prediction_service.prediction import Prediction
from model_download import download_model

app = Flask(__name__)

cfg = None
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
dagshub.init(repo_owner="", repo_name="", mlflow=True) # Enter dagshub repo_owner and repo_name
training_complete_event = threading.Event() 

def create_app():
    app = Flask(__name__)

    redis_client.delete("last_train_time")

    @app.route("/")
    def start():
        return "MLOps pipeline project"

    return app

def initialize_hydra():
    global cfg
    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name="config")
        if cfg is None:
            raise ValueError("Hydra configuration loading failed, cfg is None")
        
app = create_app()
initialize_hydra()

def get_or_create_experiment_id(exp_name):
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
        return exp_id
    return exp.experiment_id

def check_if_retrain():
    global redis_client

    last_train_timestamp = redis_client.get("last_train_time")
    current_timestamp = datetime.now()

    if last_train_timestamp is not None:
        last_train_time = datetime.fromisoformat(last_train_timestamp)
        if current_timestamp - last_train_time < timedelta(minutes=10):
            return False
        else:
            redis_client.set("last_train_time", current_timestamp.isoformat())
            return True
    else:
        redis_client.set("last_train_time", current_timestamp.isoformat())
        return True
        
def train_task():
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

    training_complete_event.set()

@app.route("/model-training")
def train_model():
    if check_if_retrain():
        training_thread = threading.Thread(target=train_task)
        training_thread.start()
        
        training_complete_event.wait()

        return "Model training completed!"
    else:
        return "Training not triggered as it occurred less than 10 minutes ago."

@app.route("/inference")
def inference():
    global cfg

    pred_instance = Prediction(cfg)
    data = Data(cfg)
    data.load_testing_data()
    data.prepare_testing_data()
    test_dataloader = data.setup_testing_dataloader()

    pred_instance.predict(test_dataloader)

    return "Inference completed" 



