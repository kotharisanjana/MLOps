FROM python:3.8-slim

WORKDIR /project

COPY requirements_inference.txt requirements_inference.txt

RUN pip install --no-cache-dir torch==2.3.1 -r requirements_inference.txt --extra-index-url https://download.pytorch.org/whl/cpu

RUN rm -rf /root/.cache

COPY . .

RUN mkdir models

RUN python download_model.py

# After deployment to lambda function
CMD ["lambda_handler.lambda_handler"]

# For testing on local
# RUN python lambda_handler.py