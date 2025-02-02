import mlflow

def log_parameters(params):
    mlflow.log_param("model", params["model"]["pretrained"]["model"])
    mlflow.log_param("tokenizer", params["model"]["pretrained"]["tokenizer"])
    mlflow.log_param("train_batch_size", params["data"]["configuration"]["train_batch_size"])
    mlflow.log_param("max_length", params["data"]["configuration"]["max_length"])
    mlflow.log_param("train_size", params["data"]["size"]["train_size"])
    mlflow.log_param("epochs", params["training"]["num_epochs"])
    mlflow.log_param("learning_rate", params["training"]["optimizer"]["lr"])

def log_dataset():
    mlflow.log_artifact("train_data.csv", "data_folder")

def log_training_metrics(train_loss, train_acc, epoch):
    mlflow.log_metrics(
         {
            "train_loss": train_loss,
            "train_acc": train_acc,
        },
        step=epoch, 
    )
 
def log_model(model):
    model_info = mlflow.pytorch.log_model(model, "classification_model")
    return model_info