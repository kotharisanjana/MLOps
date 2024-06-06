from data import Data
from model import Model
from train import Trainer
from inference import Predictor

if __name__ == "__main__":
    model_name="google/bert_uncased_L-2_H-128_A-2"
    batch_size = 32
    lr = 1e-2
    num_epochs = 1

    data = Data(model_name, batch_size)
    data.load_data()
    data.prepare_data()
    train_dataloader = data.setup_train_dataloader()
    val_dataloader = data.setup_val_dataloader()
    test_dataloader = data.setup_test_dataloader()

    model = Model(model_name)

    trainer = Trainer(model, lr)
    trainer.train_model(train_dataloader, val_dataloader, num_epochs)

    # predictor = Predictor(model, "././models/best_model.pth")
    # predictions = predictor.predict(test_dataloader)

