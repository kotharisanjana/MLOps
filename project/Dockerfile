# FROM python:3.8-slim
FROM public.ecr.aws/lambda/python:3.8
# FROM amazon/aws-lambda-python:3.8

WORKDIR /project

COPY requirements_inference.txt requirements_inference.txt
# COPY requirements_inference.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir torch==2.3.1 -r requirements_inference.txt --extra-index-url https://download.pytorch.org/whl/cpu

RUN rm -rf /root/.cache

COPY . .
# COPY . ${LAMBDA_TASK_ROOT}
# COPY lambda_handler.py ${LAMBDA_TASK_ROOT}

RUN mkdir models

RUN python download_model.py

# After deployment to lambda function
CMD ["lambda_function.lambda_handler"]

# For testing on local
# RUN python lambda_handler.py