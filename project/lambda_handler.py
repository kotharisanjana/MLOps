import json
from hydra import compose, initialize

from prediction_service.inference import Inference

def lambda_handler(event, context):
	with initialize(config_path="configs"):
		cfg = compose(config_name="config")

		inferencing_instance = Inference(cfg)
        
	if "resource" in event.keys():
		body = event["body"]
		body = json.loads(body)
		inference_sample = body["sentence"]

		response = inferencing_instance.predict(inference_sample)
		return {
			"statusCode": 200,
			"headers": {},
			"body": json.dumps(response)
		}
	else:
		return inferencing_instance.predict(event["sentence"])


# if __name__ == "__main__":
#     test = {"sentence": "this is a sample sentence"}
#     print(lambda_handler(test, None))