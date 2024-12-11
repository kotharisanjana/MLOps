from flask import Flask
import os 
import mlflow
import dagshub
import hydra
import random
from prometheus_client import Gauge, Counter
from omegaconf import DictConfig, OmegaConf

from data_service.data import Data
from model_service.model import Model
from training_service.train import Trainer
from prediction_service.inference import Inference

app = Flask(__name__)

dagshub.init(repo_owner='sanjanak98', repo_name='MLOps', mlflow=True)
ACCURACY = Gauge("prediction_accuracy", "Accuracy of the model")
INVOCATIONS = Counter("invocation_count", "Number of invocations")

@app.route('/')
def hello():
    return 'Hello, World!'

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

    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name="config")

        data = Data(cfg)
        data.load_training_data()
        data.convert_to_csv()
        data.prepare_training_data()
        train_dataloader = data.setup_training_dataloader()

        model = Model(cfg)

        trainer = Trainer(cfg, model, train_dataloader)
        model_uri = trainer.train_model(exp_id)
        
        config_path = os.path.join(hydra.utils.get_original_cwd(), "configs/model/default.yaml")
        custom_cfg = OmegaConf.load(config_path)
        custom_cfg.trained = DictConfig({"model_uri": model_uri})
        OmegaConf.save(custom_cfg, config_path)

@app.route('/inference')
def inference():
    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name="config")
        
        inferencing_instance = Inference(cfg)
        data = Data(cfg)
        data.load_testing_data()
        data.prepare_testing_data()
        test_dataloader = data.setup_testing_dataloader()

        predictions = inferencing_instance.predict(test_dataloader)

        ACCURACY.set(random.random())
        INVOCATIONS.inc()



