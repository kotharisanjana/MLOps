import json
from hydra import compose, initialize
import time

from prediction_service.inference import Inference

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

	response = inferencing_instance.predict(inference_sample)
	return {
		"statusCode": 200,
		"headers": {},
		"body": json.dumps(response.cpu().numpy().tolist())
	}
	
# if __name__ == "__main__":
# 	event = {
# 		"sentence": "This is a test sentence."
# 	}
# 	resp = lambda_handler(event, None)
# 	print(resp)