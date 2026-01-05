from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import Variable
from docker.types import Mount

class CleanDockerOperator(DockerOperator):
    template_fields = ('command', 'environment', 'container_name', 'image') #'mounts' 제외

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'tmdb_daily_collector',
    default_args=default_args,
    description='매일 TMDB 데이터를 수집하고 정제하여 S3에 적재',
    schedule='@daily',
    start_date=datetime(2025, 12, 26),
    catchup=True,
) as dag:
    current_offset = '{{ var.value.get("tmdb_offset", 0) }}'

    # 학습 데이터 수집 태스크 (Train)
    train_task = CleanDockerOperator(
        task_id='collect_train_data',
        image='mlops1:v1',
        auto_remove='success',
        mounts=[Mount(source='/home/ubuntu/mlops-mlops1/.env', target='/opt/.env', type='bind')],
        # 위에서 계산한 offset을 인자로 전달
        command=f'python ./data-prepare/main.py --mode train '
                f'--offset {{{{ (logical_date.replace(tzinfo=None) - macros.datetime(2025, 12, 26)).days }}}} '
                f'--target_date {{{{ ds_nodash }}}}',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge'
    )

    # Airflow variables (tmdb_offset) 변경 -> logical_date 계산식으로 변경!
    # def update_offset(**context):
    #     val = int(Variable.get('tmdb_offset', default=0)) + 1
    #     Variable.set('tmdb_offset', str(val))

    # update_task = PythonOperator(
    #     task_id='update_offset',
    #     python_callable=update_offset
    # )

    # 추론 데이터 수집 태스크 (Inference)
    inference_task = CleanDockerOperator(
        task_id='collect_inference_data',
        image='mlops1:v1',
        auto_remove='success',
        mounts=[Mount(source='/home/ubuntu/mlops-mlops1/.env', target='/opt/.env', type='bind')],
        command=f'python ./data-prepare/main.py --mode inference '
                f'--target_date {{{{ ds_nodash }}}}',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge'
    )

    train_task >> inference_task
