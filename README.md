# Comprehensive MLOps Infrastructure: From Training to Deployment on AWS 

MLOps is the practice of automating and streamlining the complete lifecycle of ML models, from development and experimentation to deployment and monitoring. This project presents a robust MLOps pipeline that integrates essential tools for experiment tracking, configuration management, containerization, automated CI/CD, and real-time monitoring. Designed to address the challenges of managing models in production, this pipeline simplifies the process of taking ML models from training to deployment and serves as a practical guide for implementing similar workflows.

## Workflow
![Screenshot from 2025-01-12 20-51-09](https://github.com/user-attachments/assets/458ad392-5961-4322-9b1e-5ee100089680)

## Tools
1. Application: Pytorch, Flask, REST, Redis
2. Experiment Tracking and Version Control: MLflow, Dagshub
3. Configuration Management: Hydra
4. Containerization: Docker
5. CI/CD: GitHub Actions
6. Deployment: Amazon ECR, Amazon EC2
7. Real-Time Monitoring: Amazon Cloudwatch, YACE exporter, Prometheus, Grafana

## Setup & Usage
### Local setup
1. Create a virtual environment (python 3.8): ```python -m venv <name_of_virtual_environment>```
2. Clone repository: ```git clone https://github.com/kotharisanjana/MLOps.git```
3. Install requirements: ```pip install requirements.txt```
4. Set environemnt variable:  ```"MLFLOW_TRACKING USERNAME" = <mlflow username>```
5. Navigate to app folder: ```cd mlops-app```
6. Make /models folder: ```mkdir models```
7. Run flask application: ```flask run```
8. For training: ```localhost:5000/model-training```
9. For inference: ```localhost:5000/inference```
10. On push to github repositoy, github actions is trigerred which deploys application to EC2

### Cloud setup 
**(services configuration files in /configs and EC2 policy files in /policies)**
1. Spin up an EC2 instance (Linux)
2. Allow traffic on the following ports:
   - 9090 (Prometheus)
   - 9093 (Prometheus Alertmanager)
   - 3000 (Flask application)
   - 5000 (YACE exporter)
4. Set below 2 policies for the EC2 instance:
   - Cloudwatch policy
   - ECR policy
5. Install following services on EC2: <br>
   a. **Docker**
   
   b. **YACE exporter** (to export operational metrics to prometheus)
      - Installation: [Doc 1](https://dev.to/setevoy/prometheus-yet-another-cloudwatch-exporter-collecting-aws-cloudwatch-metrics-50hd), [Doc 2](https://itnext.io/prometheus-yet-another-cloudwatch-exporter-collecting-aws-cloudwatch-metrics-806bd34818a8)
      - Config file: ```sudo nano /mnt/tig-vol/volumes/tig/exporters/cw-yace/config.yml```
        
   c. **Prometheus**
      - Installation: [Doc 1](https://codewizardly.com/prometheus-on-aws-ec2-part1/)
      - Config file: ```sudo nano /etc/prometheus/prometheus.yml```
      - Alert rules file: ```sudo nano /etc/prometheus/alert_rules.yml``` 
      - Web URL: ```localhost:9090/```
        
   d. **Prometheus Alertmanager**
      - Installation:
          ```
          - cd /etc/prometheus
          - wget https://github.com/prometheus/alertmanager/releases/download/v0.27.0/alertmanager-0.27.0.linux-amd64.tar.gz
          - tar xvf alertmanager-0.27.0.linux-amd64.tar.gz
          ```
      - Config file: ```sudo nano /etc/prometheus/alert_manager/alertmanager-0.27.0.linux-amd64/alertmanager.yml```
      - Web URL: ```localhost:9093/```
      
   e. **Redis-server**
      - Installation:
        ```
        - sudo yum groupinstall "Development Tools" -y
        - sudo yum install -y tcl
        - wget http://download.redis.io/redis-stable.tar.gz
        - tar xzvf redis-stable.tar.gz
        - cd redis-stable
        - make
        - sudo make install
        ```
6. On completing all installations: ```bash start_services.sh ```
7. Once application docker image is available in ECR repository (perform these steps only if github actions do not automate it):
    - Pull image on EC2: ```docker pull <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/<your-repository-name>:<tag>```
    - Run image: ```docker run -it -p 3000:3000 –-network host <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/<your-repository-name>:<tag>```
8. To call APIs:
   - `http://<ec2-public-address>:3000/model-training` (when running the application for the first time - saves trained model in /models)
   - `http://<ec2-public-address>:3000/inference`
9. To shut down all services: ```bash stop_services.sh``` 

## Future scope
1. **Feature Store Integration:** Incorporate a feature store like Feast to manage, version, and serve consistent features for training and inference.
2. **A/B Testing:** Implement A/B testing to compare model variants in production and deploy the best-performing version.
3. **Modular Packaging:** Package the pipeline components into reusable modules that can be easily imported into other projects.
4. **Enhanced Monitoring:** Monitor data drift, concept drift, covariate shift, data quality metrics, etc.
