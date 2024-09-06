import sys
from pathlib import Path
from airflow.decorators import dag, task
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(BASE_DIR.as_posix())
from ml.src.versioning import init_mlflow
from ml.src.model import fit

default_args = {
    'owner': 'kazkovas73',
    'retries': 1,
    'time_delay': timedelta(minutes=5)
}

@task
def init_mlflow_task():
    init_mlflow()

@task
def run_pipeline_task():
    fit()


@dag(
    dag_id='dag_pipeline_v01',
    default_args=default_args,
    start_date=datetime(2024, 9, 6, 10),
    schedule_interval='@daily'
)
def pipeline():
    init_mlflow_task()
    run_pipeline_task()