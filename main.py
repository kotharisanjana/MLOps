import mlflow
import hydra
from omegaconf import DictConfig

from classification_model.data import Data
from classification_model.model import Model
from classification_model.train import Trainer
from classification_model.inference import Predictor
# from onnx.conversion import convert_to_onnx
# from onnx.inference import OnnxPredictor

mlflow_tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("mlflow_experiment")

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    data = Data(cfg)
    data.load_data()
    train_dataset, val_dataset, test_dataset = data.prepare_logging_data()
    data.prepare_modeling_data()
    train_dataloader, val_dataloader, test_dataloader = data.setup_dataloaders()

    model = Model(cfg)

    trainer = Trainer(cfg, model, train_dataloader, val_dataloader, train_dataset, val_dataset)
    model_uri = trainer.train_model()

    predictor = Predictor(model_uri, test_dataloader)
    predictions = predictor.predict()

    # model_path = "./model/model.onnx"
    # convert_to_onnx(model_uri, train_dataloader, model_path)
    # onnx_predictor = OnnxPredictor(test_dataloader, model_path)
    # onnx_predictions = onnx_predictor.predict()

if __name__ == "__main__":
    main()

    
