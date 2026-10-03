"""Incremental update job: fetch only NEW rows, validate, and update the model with ONE
`partial_fit` call on that whole batch (incremental learning, not row-by-row online learning).

    python -m klia.jobs.update                 # normal run (cron / Task Scheduler / Airflow)
    python -m klia.jobs.update --bootstrap     # first run: start from the notebook's trained model
                                                 #  (artifacts/pretrained_bundle.gz) if present,
                                                 #  else a fresh untrained model, then replay history
    python -m klia.jobs.update --csv data/departures.csv   # run against a CSV instead of Neon
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from klia.config import artifacts_dir, load_config
from klia.etl.validate import clean
from klia.model.bundle import Bundle, new_bundle
from klia.pipeline import apply_rows
from klia.store.base import open_store
from klia.store.postgres import PostgresStore

log = logging.getLogger("klia.update")


def _log_mlflow(info: dict, version: int) -> None:
    """Best-effort MLflow logging; silently skipped if mlflow is not installed."""
    try:
        import mlflow
        db = Path(__file__).resolve().parents[2] / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db}")
        mlflow.set_experiment("klia-flight-delay")
        with mlflow.start_run(run_name=f"update_v{version}"):
            numeric = {k: v for k, v in info.items() if isinstance(v, (int, float))}
            mlflow.log_metrics({k: float(v) for k, v in numeric.items()})
            mlflow.log_params({"version": version})
    except Exception as e:
        log.debug("mlflow logging skipped: %s", e)


def _starting_bundle(cfg: dict, bootstrap: bool) -> Bundle:
    """On --bootstrap, prefer the fully-trained bundle the notebook saved; else start fresh."""
    pretrained = artifacts_dir() / "pretrained_bundle.gz"
    if bootstrap and pretrained.exists():
        log.info("bootstrapping from the notebook's trained model: %s", pretrained)
        return Bundle.loads(pretrained.read_bytes())
    log.info("starting a fresh, untrained model (%s)", cfg["model"]["name"])
    return new_bundle(cfg)


def run(bootstrap: bool = False, csv: str | None = None,
        dry_run: bool = False, max_rows: int | None = None) -> dict:
    cfg = load_config()
    store = open_store(cfg, csv)
    store.ensure_schema()
    t0 = time.time()
    with store.lock():
        loaded = None if bootstrap else store.load_bundle_bytes()
        if loaded is None:
            if not bootstrap and store.get_watermark() > 0:
                raise RuntimeError("watermark exists but no model bundle found; re-run with --bootstrap")
            bundle, watermark = _starting_bundle(cfg, bootstrap), 0
        else:
            bundle, watermark = Bundle.loads(loaded[1]), store.get_watermark()
            log.info("loaded bundle v%s, watermark id=%s", loaded[0], watermark)

        totals = {"rows_in": 0, "rows_ok": 0, "rows_rejected": 0,
                  "rows_learned": 0, "drift_events": 0}
        reasons: dict[str, int] = {}
        batch = cfg["data"]["batch_size"]
        stats: dict = {}
        while True:
            raw = store.fetch_new_rows(watermark, batch)
            if raw.empty:
                break
            ok, bad = clean(raw, cfg)
            watermark = int(raw["id"].max())
            totals["rows_in"] += len(raw)
            totals["rows_ok"] += len(ok)
            totals["rows_rejected"] += len(bad)
            for r, n in bad["reason"].value_counts().items():
                reasons[r] = reasons.get(r, 0) + int(n)
            stats = apply_rows(bundle, ok, cfg)                 # one partial_fit call for this batch
            totals["rows_learned"] += stats.get("rows_learned", 0)
            totals["drift_events"] += stats.get("drift_events", 0)
            log.info("processed %s rows as one batch (watermark id=%s)", len(raw), watermark)
            if len(raw) < batch or (max_rows and totals["rows_in"] >= max_rows):
                break

        if totals["rows_in"] == 0:
            log.info("no new rows, nothing to do")
            return {"status": "no_new_rows"}

        info = {**totals, "reject_reasons": reasons, "seconds": round(time.time() - t0, 1),
                **{k: stats[k] for k in ("roc_auc", "f1", "log_loss", "threshold") if k in stats}}
        blob = bundle.dumps()
        info["bundle_mb"] = round(len(blob) / 1e6, 2)

        if isinstance(store, PostgresStore):
            cap = cfg["registry"].get("max_mb_neon", 400)
            if info["bundle_mb"] > cap:
                raise RuntimeError(
                    f"model bundle is {info['bundle_mb']} MB, over the Neon cap of {cap} MB. "
                    f"Either shrink the ensemble (fewer/smaller members) or store it locally instead "
                    f"(unset DATABASE_URL / use the file store), which has no size cap.")

        if dry_run:
            log.info("dry run, not saving: %s", info)
            return {"status": "dry_run", **info}

        version = store.commit(blob, bundle.describe(), watermark, info)
        log.info("saved model v%s: %s", version, info)
        _log_mlflow(info, version)
        return {"status": "ok", "version": version, **info}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", action="store_true", help="start from the notebook's trained model and replay all history")
    ap.add_argument("--csv", help="read rows from this CSV instead of Neon")
    ap.add_argument("--dry-run", action="store_true", help="process but do not save")
    ap.add_argument("--max-rows", type=int, help="stop after about this many rows (for quick trials)")
    a = ap.parse_args()
    run(bootstrap=a.bootstrap, csv=a.csv, dry_run=a.dry_run, max_rows=a.max_rows)


if __name__ == "__main__":
    main()
