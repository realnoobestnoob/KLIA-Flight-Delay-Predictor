"""Airflow DAG — KLIA incremental model update with Evidently drift monitoring.

Runs daily at 20:00 Singapore time (12:00 UTC).

Offline behaviour
-----------------
catchup=True + max_active_runs=1 means: if the laptop/container was off when a
scheduled run was due, Airflow runs the missed run immediately when it comes back
online, then resumes the normal schedule.  Because the update job is watermark-
based (only rows with id > last watermark are processed), catching up multiple
missed runs is safe — the first run processes all accumulated data, and subsequent
catch-up runs hit no new rows and skip cleanly.

Docker path
-----------
Project is mounted at /opt/klia-mlops inside the container.
PYTHONPATH=/opt/klia-mlops is set in docker-compose.airflow.yml, so no sys.path
manipulation is needed here.
"""
from __future__ import annotations

import datetime as dt
import logging

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

log = logging.getLogger(__name__)

default_args = {
    "owner": "klia",
    "retries": 2,
    "retry_delay": dt.timedelta(minutes=10),
    "retry_exponential_backoff": True,
}


@dag(
    dag_id="klia_incremental_update",
    description=(
        "Validate new KLIA flights → partial_fit → Evidently drift check → save bundle. "
        "Runs at 20:00 SGT daily. Catches up missed runs on restart."
    ),
    schedule="0 12 * * *",           # 20:00 SGT = 12:00 UTC (SGT is UTC+8)
    start_date=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    catchup=True,                     # run missed intervals immediately on restart
    max_active_runs=1,                # one run at a time (watermark is safe but advisory lock is not)
    default_args=default_args,
    tags=["klia", "mlops"],
    doc_md=__doc__,
)
def klia_incremental_update():

    @task
    def check_connection() -> None:
        """Fail fast if Neon or the table/schema is unreachable before touching the model."""
        from klia.jobs.check import main as check_main
        try:
            result = check_main()
            if result != 0:
                raise RuntimeError(
                    f"klia.jobs.check returned exit code {result} — see task logs"
                )
            log.info("Connection check passed.")
        except Exception as exc:
            log.error("Connection check failed: %s", exc, exc_info=True)
            raise

    @task
    def run_update() -> dict:
        """Fetch new rows, partial_fit, Evidently drift check, save bundle.
        Skips cleanly if there are no new rows (not a failure).
        """
        from klia.jobs.update import run
        try:
            result = run()
            log.info("Update completed: %s", result)
            if result.get("status") == "no_new_rows":
                raise AirflowSkipException(
                    "No new rows since the last watermark — nothing to learn from."
                )
            return result
        except AirflowSkipException:
            raise
        except Exception as exc:
            log.error("Update job failed: %s", exc, exc_info=True)
            raise

    check_connection() >> run_update()


klia_incremental_update()
