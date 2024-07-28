FROM python:3.8-slim

WORKDIR /project

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir torch==2.3.1 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

RUN rm -rf /root/.cache

COPY . .

RUN python download_model.py

# CMD ["lambda_handler.lambda_handler"]

RUN python lambda_handler.py