import json
import boto3
import random
from hydra import compose, initialize

from prediction_service.inference import Inference

cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")

def lambda_handler(event, context):
	with initialize(config_path="configs"):
		cfg = compose(config_name="config")

		inferencing_instance = Inference(cfg)
              
	inference_sample = None

	if "body" in event:
		try:
			body = json.loads(event["body"])  
			inference_sample = body.get("sentence", None)
		except (json.JSONDecodeError, TypeError):
			return {
				"statusCode": 400,
				"body": json.dumps({"error": "Invalid JSON in request body"})
				}
		
	if inference_sample is None:
		inference_sample = event.get("sentence", None)

	if inference_sample is None:
		return {
			"statusCode": 400,
			"body": json.dumps({"error": "sentence key not found in event"})
			}

	input = {"sentence": inference_sample}
	response = inferencing_instance.predict(input)

	cloudwatch.put_metric_data(
		MetricData = [
            {
                "MetricName": "OnlineDevices",
                "Dimensions": [
                    {
                        "Name": "LastMessages",
                        "Value": "With-in-one-hour"
                    },
                    {
                        "Name": "APP_VERSION",
                        "Value": "1.0"
                    },
                    ],
                    "Unit": "Count",
                    "Value": random.randint(1, 500)
            },
            {
                "MetricName": "TotalDevices",
                "Dimensions": [
                    {
                        "Name": "LastMessages",
                        "Value": "With-in-one-year"
                    },
                    {
                        "Name": "APP_VERSION",
                        "Value": "1.0"
                    },
                    ],
                    "Unit": "Count",
                    "Value": random.randint(1, 500)
            },
        ],
		Namespace = "MLOpsApp"
	)

	return {
		"statusCode": 200,
		"headers": {},
		"body": json.dumps(response.cpu().numpy().tolist())
	}