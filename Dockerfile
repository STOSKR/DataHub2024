FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
COPY main.py ./

RUN pip install -r requirements.txt
RUN pip install google-cloud-bigquery google-cloud-bigquery-storage vertexai pandas google-cloud-aiplatform db-dtypes google-cloud-bigquery-storage

EXPOSE 8080

CMD ["python", "main.py"]