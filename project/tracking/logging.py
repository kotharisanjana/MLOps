import mlflow
import torch

def log_experiment(params, train_dataset, val_dataset):
    mlflow.log_params(params)
    mlflow.log_input(train_dataset, "train_data")
    mlflow.log_input(val_dataset, "val_data")

def log_training_metrics(train_loss, train_acc, val_loss, val_acc, epoch):
    mlflow.log_metrics(
         {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        step=epoch, 
    )
 
def log_model(model):
    model_info = mlflow.pytorch.log_model(model, "classification_model")
    return model_info

def save_model_to_local(model):
    torch.save(model.state_dict(), "/home/sanjana/Desktop/MLOps/models/model.pth")