import mlflow

from setup.data import Data
from setup.model import Model
from setup.train import Trainer
from setup.inference import Predictor
from onnx.conversion import convert_to_onnx
from onnx.inference import OnnxPredictor

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlflow_experiment")

if __name__ == "__main__":
    params = {
        "model_name": "google/bert_uncased_L-2_H-128_A-2",
        "batch_size": 4,
        "lr": 1e-3,
        "num_epochs": 2
    }

    data = Data(params)
    data.load_data()
    train_dataset, val_dataset, test_dataset = data.prepare_logging_data()
    data.prepare_modeling_data()
    train_dataloader, val_dataloader, test_dataloader = data.setup_dataloaders()

    model = Model(params["model_name"])

    trainer = Trainer(model, params, train_dataloader, val_dataloader, train_dataset, val_dataset)
    model_uri = trainer.train_model()

    predictor = Predictor(model_uri, test_dataloader)
    predictions = predictor.predict()

    model_path = "./model/model.onnx"
    convert_to_onnx(model_uri, train_dataloader, model_path)
    onnx_predictor = OnnxPredictor(test_dataloader, model_path)
    onnx_predictions = onnx_predictor.predict()


    
