import torch 
import mlflow

def convert_to_onnx(model_uri, train_dataloader, model_path):
    input_batch = next(iter(train_dataloader))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }

    model = mlflow.pytorch.load_model(model_uri)

    torch.onnx.export(
        model,
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),  
        model_path,
        export_params=True,
        opset_version=10,
        input_names=["input_ids", "attention_mask"], 
        output_names=["output"], 
        dynamic_axes={
            "input_ids": {0: "batch_size"}, 
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )