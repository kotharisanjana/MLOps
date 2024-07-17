FROM public.ecr.aws/lambda/python:3.8

COPY requirements_inference.txt requirements_inference.txt

RUN pip install -r requirements_inference.txt

COPY . .

CMD ["lambda_handler.lambda_handler"]