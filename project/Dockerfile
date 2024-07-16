FROM amazon/aws-lambda-python
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY . /app
CMD ["app.lambda_handler"]