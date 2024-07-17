FROM amazon/aws-lambda-python
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["lambda_handler.lambda_handler"]