"""Airflow DAG: runs the same incremental update job (validate -> partial_fit -> save bundle)
on a schedule, instead of Task Scheduler / cron. No change to the underlying logic -- this just
calls klia.jobs.update.run(), the same function the CLI uses.

Setup (see README "Apache Airflow" section):
  1. pip install apache-airflow==2.9.3 (in a separate venv/container -- it pins many deps)
  2. export AIRFLOW_HOME=~/airflow
  3. airflow db init
  4. Point Airflow's dags_folder (in airflow.cfg) at this project's `airflow/dags`, or symlink
     this file into $AIRFLOW_HOME/dags
  5. Set DATABASE_URL as an Airflow Variable or Connection, or export it before `airflow webserver`
  6. airflow webserver --port 8080   &   airflow scheduler
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

default_args = {
    "owner": "klia",
    "retries": 2,
    "retry_delay": dt.timedelta(minutes=5),
}


@dag(
    dag_id="klia_incremental_update",
    description="Validate new flights, update the model incrementally, save a new bundle",
    schedule="30 20 * * *",          # 20:30 UTC = 04:30 Malaysia time, daily
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["klia", "mlops"],
)
def klia_incremental_update():

    @task
    def check_connection() -> None:
        """Fail fast if Neon or the table/schema is unreachable, before touching the model."""
        from klia.jobs.check import main as check_main
        if check_main() != 0:
            raise RuntimeError("klia.jobs.check failed -- see task logs for details")

    @task
    def run_update() -> dict:
        """One partial_fit call per batch of new rows, exactly like `python -m klia.jobs.update`."""
        from klia.jobs.update import run
        result = run()
        log.info("update result: %s", result)
        if result.get("status") == "no_new_rows":
            raise AirflowSkipException("no new rows since the last run")
        return result

    check_connection() >> run_update()


klia_incremental_update()
