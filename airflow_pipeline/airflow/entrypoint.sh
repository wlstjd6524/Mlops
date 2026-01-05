#!/usr/bin/env bash
set -e

rm -f "$AIRFLOW_HOME/airflow-webserver.pid"

airflow db migrate

airflow users create \
  --username "${AIRFLOW_ADMIN_USER:-mlops1}" \
  --password "${AIRFLOW_ADMIN_PASSWORD:-mlops1}" \
  --firstname fast \
  --lastname campus \
  --role Admin \
  --email mlops1@example.com || true

airflow scheduler &
exec airflow webserver -p 8080
