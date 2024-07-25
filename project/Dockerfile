FROM public.ecr.aws/lambda/python:3.8

ARG TEMP_DIR=./tmp
RUN mkdir $TEMP_DIR

ENV TRANSFORMERS_CACHE=$TEMP_DIR \
    TRANSFORMERS_VERBOSITY=error 

WORKDIR /project

# ENV PYTHONPATH "${PYTHONPATH}:./"

RUN rpm -e --nodeps sqlite

# Ensure a clean environment
RUN yum clean all && \
    yum update -y && \
    yum install -y wget gcc make tar

# Download SQLite source and install
RUN curl --compressed https://www.sqlite.org/2023/sqlite-autoconf-3320300.tar.gz && \
    # tar xzf ./sqlite-autoconf-3320300.tar.gz && \
    cd ./sqlite-autoconf-3320300 && \
    ./configure && \
    make && \
    make install

RUN sqlite3 --version

# COPY requirements_inference.txt requirements_inference.txt
# RUN pip install -r requirements_inference.txt --no-cache-dir --no-deps

COPY . .

# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8

RUN pip install "dvc[s3]"
RUN dvc init --no-scm
RUN dvc remote add -d model-store s3://mlopsbucket1

RUN dvc pull models/model.pth.dvc

RUN chmod -R 0755 $TEMP_DIR

CMD [ "lambda_handler.lambda_handler"]