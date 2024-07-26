FROM python:3.8-slim

WORKDIR /project

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

RUN ls -R

RUN python download_model.py

# CMD ["lambda_handler.lambda_handler"]

RUN python lambda_handler.py