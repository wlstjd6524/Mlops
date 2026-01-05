from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import dotenv_values

# 1. 프로젝트 루트 계산
# airflow/dags/model_train_dag.py 기준
BASE_DIR = Path(__file__).resolve().parents[2]

# 2. 환경변수 로드 (.env 파일)
COMMON_ENV = dotenv_values(BASE_DIR / "airflow/.env.common")
TRAIN_ENV = dotenv_values(BASE_DIR / "train/.env.train")

# 병합 (train이 우선)
ENV = {**COMMON_ENV, **TRAIN_ENV}

# 3. DAG 기본 설정
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2025, 12, 27),
    "retries": 1,
}

# 4. DAG 정의
with DAG(
    dag_id="model_training",
    default_args=default_args,
    description="Train, evaluate, and save ML model",
    schedule="@daily",
    catchup=False,
    tags=["mlops", "train"],
) as dag:

    train_and_evaluate_and_save = DockerOperator(
        task_id="train_and_evaluate_and_save",
        image="mlops-train:v1",
        api_version="auto",
        auto_remove=True,
        command="python src/model/main.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        environment={
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "S3_BUCKET_NAME": os.environ.get("S3_BUCKET_NAME"),
            "S3_PREFIX": "preprocess/train/",
       },  
    )

    train_and_evaluate_and_save

