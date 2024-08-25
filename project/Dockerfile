FROM public.ecr.aws/lambda/python:3.8
# FROM amazon/aws-lambda-python:3.8

WORKDIR ${LAMBDA_TASK_ROOT}

COPY . ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir torch==2.3.1 -r requirements_inference.txt --extra-index-url https://download.pytorch.org/whl/cpu

RUN rm -rf /root/.cache

ARG MODEL_DIR=models
RUN mkdir $MODEL_DIR

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

RUN python download_model.py

# After deployment to lambda function
CMD ["lambda_function.lambda_handler"]

# For testing on local
# RUN python lambda_handler.py