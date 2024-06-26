FROM python:3.8
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY . /app
# CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 & python main.py"]
CMD ["python", "main.py"]