FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir torch==2.3.1 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

ARG CACHE_DIR=/tmp/transformers_cache
ARG MODEL_DIR=models

RUN mkdir $CACHE_DIR $MODEL_DIR

ENV TRANSFORMERS_CACHE=${CACHE_DIR} \
    TRANSFORMERS_VERBOSITY=error

CMD ["flask", "run", "--host=0.0.0.0", "--port=3000"]
