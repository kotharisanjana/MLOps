# import json
# import random
# from prometheus_client import start_http_server, Gauge
# from hydra import compose, initialize
# import threading

# from prediction_service.inference import Inference

# ACCURACY = Gauge('prediction_accuracy', 'Accuracy of the model')

# def start_server():
#     start_http_server(port=5000, addr='0.0.0.0')
#     while True:
#         pass

# def lambda_handler(event):
# 	with initialize(config_path="configs"):
# 		cfg = compose(config_name="config")

# 		inferencing_instance = Inference(cfg)
              
# 	inference_sample = None

# 	if "body" in event:
# 		try:
# 			body = json.loads(event["body"])  
# 			inference_sample = body.get("sentence", None)
# 		except (json.JSONDecodeError, TypeError):
# 			return {
# 				"statusCode": 400,
# 				"body": json.dumps({"error": "Invalid JSON in request body"})
# 				}
		
# 	if inference_sample is None:
# 		inference_sample = event.get("sentence", None)

# 	if inference_sample is None:
# 		return {
# 			"statusCode": 400,
# 			"body": json.dumps({"error": "sentence key not found in event"})
# 			}

# 	input = {"sentence": inference_sample}
# 	response = inferencing_instance.predict(input)
	
# 	ACCURACY.set(random.random())

# 	return {
# 		"statusCode": 200,
# 		"headers": {},
# 		"body": json.dumps(response.cpu().numpy().tolist())
# 	}



# if __name__ == "__main__":
# 	server_thread = threading.Thread(target=start_server)      
# 	server_thread.start()

# 	event = {
# 		"sentence": "life is beautiful when you find happiness in small things"
# 	}
	
# 	lambda_handler(event)
	
# 	server_thread.join()
