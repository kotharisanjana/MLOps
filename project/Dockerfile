FROM public.ecr.aws/lambda/python:3.8

WORKDIR /project

COPY requirements_inference.txt requirements_inference.txt

RUN pip install -r requirements_inference.txt --no-cache-dir

COPY . .

CMD ["lambda_handler.lambda_handler"]