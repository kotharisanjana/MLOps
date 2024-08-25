import json
from hydra import compose, initialize

from prediction_service.inference import Inference
from load_model import load_model_uri_from_config, download_model

def lambda_handler(event, context):
	with initialize(config_path="configs"):
		cfg = compose(config_name="config")

		inferencing_instance = Inference(cfg)
        
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