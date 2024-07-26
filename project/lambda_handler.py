import json
import mlflow
from hydra import compose, initialize

from prediction_service.inference import Inference

# either take the model fromt he last ru or pass a model uri to lambda_handler while creating image???

# def get_latest_model_uri(experiment_name):
#     client = mlflow.tracking.MlflowClient()
#     experiment = client.get_experiment_by_name(experiment_name)
#     if experiment:
#         runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)
#         if runs:
#             latest_run = runs[0]
#             return latest_run.data.params["model_uri"]
#     return None

def lambda_handler(model_uri, event, context):
	with initialize(config_path="configs"):
		cfg = compose(config_name="config")

		inferencing_instance = Inference(cfg, model_uri)
        
	if "resource" in event.keys():
		body = event["body"]
		body = json.loads(body)
		inference_sample = body["sentence"]
	else:
		inference_sample = event["sentence"]

	input = {"sentence": inference_sample}
	response = inferencing_instance.predict(input)
	return {
		"statusCode": 200,
		"headers": {},
		"body": json.dumps(response.cpu().numpy().tolist())
	}
	
if __name__ == "__main__":
	event = {
		"sentence": "This is a test sentence."
	}
	model_uri = "mlflow-artifacts:/605967283737555311/9f1aa47ca4f94264ab1c5f717abec13c/artifacts/classification_model"
	resp = lambda_handler(model_uri, event, None)
	print(resp)