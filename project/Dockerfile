# FROM public.ecr.aws/lambda/python:3.8

# COPY requirements_inference.txt requirements_inference.txt

# RUN pip install -r requirements_inference.txt --no-cache-dir

# COPY . .

# RUN rm -rf /var/lib/apt/lists/*

# CMD ["lambda_handler.lambda_handler"]

FROM public.ecr.aws/lambda/python:3.8

ARG TEMP_DIR=./temp
RUN mkdir $TEMP_DIR

ENV TRANSFORMERS_CACHE=$TEMP_DIR \
    TRANSFORMERS_VERBOSITY=error \
    PYTHONPATH="/var/task:${PYTHONPATH}"
    
COPY requirements_inference.txt requirements_inference.txt
RUN pip install -r requirements_inference.txt --no-cache-dir

COPY . .

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD [ "lambda_handler.lambda_handler"]