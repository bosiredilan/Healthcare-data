"""
dags/retrain_dag.py
Airflow 3 TaskFlow DAG – AfyaPredict KE weekly retraining pipeline.
Schedule: Every Saturday at 12:00 UTC (15:00 EAT).
"""

import subprocess
from datetime import datetime

from airflow.sdk import dag, task


@dag(
    dag_id="afyapredict_ke_retrain",
    description="Weekly retraining pipeline for AfyaPredict KE – Kenyan clinical ML platform",
    schedule="0 12 * * 6",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["afyapredict", "kenya", "ml", "retrain"],
)
def afyapredict_ke_retrain():

    @task(task_id="ingest_raw_data")
    def ingest() -> str:
        result = subprocess.run(
            ["python", "scripts/ingest.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        return "ingest_complete"

    @task(task_id="clean_data")
    def clean(upstream: str) -> str:
        result = subprocess.run(
            ["python", "scripts/clean.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        return "clean_complete"

    @task(task_id="kenyanize_data")
    def kenyanize(upstream: str) -> str:
        result = subprocess.run(
            ["python", "scripts/kenyanize.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        return "kenyanize_complete"

    @task(task_id="load_to_postgres")
    def load(upstream: str) -> str:
        result = subprocess.run(
            ["python", "scripts/load.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        return "load_complete"

    @task(task_id="train_model")
    def train(upstream: str) -> dict:
        result = subprocess.run(
            ["python", "ml/train.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        return {"status": "trained", "output": result.stdout[-500:]}

    @task(task_id="notify_completion")
    def notify(train_result: dict) -> None:
        print("=" * 60)
        print("🇰🇪 AfyaPredict KE – Weekly Retrain Complete")
        print(f"   Status  : {train_result['status']}")
        print(f"   Schedule: Saturday 12:00 UTC (15:00 EAT)")
        print("=" * 60)
        print(train_result["output"])

    ingest_out = ingest()
    clean_out = clean(ingest_out)
    kenyanize_out = kenyanize(clean_out)
    load_out = load(kenyanize_out)
    train_out = train(load_out)
    notify(train_out)


afyapredict_ke_retrain()