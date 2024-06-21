from data import Data
from model import Model
from train import Trainer
from inference import Predictor
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlflow_experiment")

if __name__ == "__main__":
    params = {
        "model_name": "google/bert_uncased_L-2_H-128_A-2",
        "batch_size": 4,
        "lr": 1e-2,
        "num_epochs": 1
    }

    data = Data(params)
    data.load_data()
    data.prepare_data()
    train_dataloader = data.setup_train_dataloader()
    val_dataloader = data.setup_val_dataloader()
    test_dataloader = data.setup_test_dataloader()

    model = Model(params["model_name"])

    trainer = Trainer(model, params)
    model_uri = trainer.train_model(train_dataloader, val_dataloader)

    predictor = Predictor(model_uri)
    predictions = predictor.predict(test_dataloader)

