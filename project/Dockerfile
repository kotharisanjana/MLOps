FROM amazon/aws-lambda-python
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY . /app
RUN python lambda_handler.py
CMD ["lambda_handler.lambda_handler"]