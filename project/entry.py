import random
from prometheus_client import start_http_server, Gauge, Counter
from hydra import compose, initialize
import threading
import time
import sys

from prediction_service.inference import Inference

ACCURACY = Gauge("prediction_accuracy", "Accuracy of the model")
INVOCATIONS = Counter("invocation_count", "Number of invocations")

def start_server():
    start_http_server(port=5001, addr="0.0.0.0")
    while True:
	    time.sleep(1)
    
def inference(event):
	with initialize(config_path="configs"):
		cfg = compose(config_name="config")

		inferencing_instance = Inference(cfg)
              
	response = inferencing_instance.predict(event)
	print(response, flush=True)
	
	ACCURACY.set(random.random())
	INVOCATIONS.inc()


if __name__ == "__main__":
	server_thread = threading.Thread(target=start_server) 
	server_thread.daemon = True 
	server_thread.start()

	try:
		while True:
			event = {
				"sentence": "life is beautiful when you find happiness in small things"
			}
			
			inference(event)
			time.sleep(20) 
	except KeyboardInterrupt:
		print("Shutting down")
		sys.exit(0)
	except Exception as e:  
		print(f"Unexpected error: {e}")
		sys.exit(1)