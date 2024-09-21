FROM public.ecr.aws/lambda/python:3.8

WORKDIR ${LAMBDA_TASK_ROOT}

COPY . ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir torch==2.3.1 -r requirements_inference.txt --extra-index-url https://download.pytorch.org/whl/cpu

ARG CACHE_DIR=/tmp/transformers_cache
ARG MODEl_DIR=models

RUN mkdir $CACHE_DIR
RUN mkdir $MODEl_DIR

ENV TRANSFORMERS_CACHE=${MODEL_DIR} \
    TRANSFORMERS_VERBOSITY=error

RUN python load_model.py

CMD ["lambda_function.lambda_handler"]