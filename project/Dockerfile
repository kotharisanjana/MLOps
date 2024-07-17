FROM public.ecr.aws/lambda/python:3.8

COPY requirements_inference.txt requirements_inference.txt

RUN pip install -r requirements_inference.txt --no-cache-dir --no-deps

COPY . .

RUN rm -rf /var/lib/apt/lists/*

CMD ["lambda_handler.lambda_handler"]