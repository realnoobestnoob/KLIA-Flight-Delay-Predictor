"""Airflow DAG: runs the same incremental update job (validate -> partial_fit -> save bundle)
on a schedule, instead of Task Scheduler / cron. No change to the underlying logic -- this just
calls klia.jobs.update.run(), the same function the CLI uses.

Setup for Airflow 3.3.2 (see README "Apache Airflow setup" section):
  1. python -m venv .venv-airflow
  2. .venv-airflow\Scripts\activate  (Windows) or source .venv-airflow/bin/activate (macOS/Linux)
  3. pip install apache-airflow==3.3.2
  4. airflow db migrate
  5. airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
  6. Point Airflow's dags_folder (in airflow.cfg) at this project's `airflow/dags`, or symlink
     this file into $AIRFLOW_HOME/dags
  7. Set DATABASE_URL as an environment variable before starting webserver/scheduler
  8. airflow webserver --port 8080   &   airflow scheduler
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
    description="Validate new flights, update the model incrementally with drift detection, save a new bundle",
    schedule="30 20 * * *",          # 20:30 UTC = 04:30 Malaysia time, daily
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["klia", "mlops"],
    max_active_runs=1,               # Prevent concurrent runs (update job uses advisory lock anyway)
)
def klia_incremental_update():
    """Main DAG for KLIA incremental model update with Evidently drift monitoring.
    
    Tasks:
      1. check_connection — verify Neon and table schema are accessible
      2. run_update — fetch new rows, validate, compute features, partial_fit, run Evidently drift check
    """

    @task
    def check_connection() -> None:
        """Fail fast if Neon or the table/schema is unreachable, before touching the model.
        
        Raises RuntimeError if any check fails (connection, table existence, required columns).
        """
        from klia.jobs.check import main as check_main
        try:
            result = check_main()
            if result != 0:
                raise RuntimeError(f"klia.jobs.check returned {result}. See task logs for details.")
        except Exception as exc:
            log.error("Connection check failed: %s", exc, exc_info=True)
            raise

    @task
    def run_update() -> dict:
        """One partial_fit call per batch of new rows, exactly like `python -m klia.jobs.update`.
        
        Returns a dict with keys: rows_learned, drift_events, scored_rows, roc_auc, f1, log_loss, delay_rate.
        
        Raises AirflowSkipException if there are no new rows (not an error, just nothing to do).
        """
        from klia.jobs.update import run
        try:
            result = run()
            log.info("Update completed: %s", result)
            if result.get("status") == "no_new_rows":
                raise AirflowSkipException("No new rows since the last run.")
            return result
        except AirflowSkipException:
            raise  # Re-raise skip exceptions as-is
        except Exception as exc:
            log.error("Update job failed: %s", exc, exc_info=True)
            raise

    # Dependency: check connection first, then run the update
    check_connection() >> run_update()


# Instantiate the DAG
klia_incremental_update()