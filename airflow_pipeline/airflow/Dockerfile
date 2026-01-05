FROM python:3.10-slim

ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ARG AIRFLOW_VERSION=2.8.1
ENV AIRFLOW_CONSTRAINTS_URL="https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.10.txt"

RUN pip install "apache-airflow==2.8.1" \
    --constraint "${AIRFLOW_CONSTRAINTS_URL}"

RUN pip install apache-airflow-providers-docker \
    --constraint "${AIRFLOW_CONSTRAINTS_URL}"

COPY requirements-airflow.txt /
RUN pip install -r /requirements-airflow.txt \
    --constraint "${AIRFLOW_CONSTRAINTS_URL}"

WORKDIR $AIRFLOW_HOME
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
CMD ["/entrypoint.sh"]

