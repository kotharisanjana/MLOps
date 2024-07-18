FROM public.ecr.aws/lambda/python:3.8

ARG TEMP_DIR=./tmp
RUN mkdir $TEMP_DIR

ENV TRANSFORMERS_CACHE=$TEMP_DIR \
    TRANSFORMERS_VERBOSITY=error \
    PYTHONPATH="/var/task:${PYTHONPATH}"

COPY requirements_inference.txt requirements_inference.txt
RUN pip install -r requirements_inference.txt --no-cache-dir

COPY . .

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install "dvc[s3]"
RUN dvc init --no-scm
RUN dvc remote add -d model-store s3://mlopsbucket1

RUN dvc pull ./models/model.pth.dvc

RUN chmod -R 0755 $TEMP_DIR

CMD [ "lambda_handler.lambda_handler"]