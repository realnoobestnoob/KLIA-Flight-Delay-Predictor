"""Incremental update job: fetch only NEW rows, validate, and update the model with ONE
`partial_fit` call on that whole batch (incremental learning, not row-by-row online learning).

    python -m klia.jobs.update                 # normal run (cron / Task Scheduler / GitHub Actions)
    python -m klia.jobs.update --bootstrap     # first run: start from the notebook's trained model
                                                 #  (artifacts/pretrained_bundle.gz) if present,
                                                 #  else a fresh untrained model, then replay history.
                                                 #  Also runs the feature-selection probe to determine
                                                 #  which features the model trains on.
    python -m klia.jobs.update --csv data/departures.csv   # run against a CSV instead of Neon
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from klia.config import artifacts_dir, load_config
from klia.etl.validate import clean
from klia.features.state import FeatureState
from klia.model.bundle import Bundle, new_bundle
from klia.pipeline import apply_rows, feature_stream
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


def _run_feature_selection_probe(bundle: Bundle, store, cfg: dict) -> None:
    """Rank all features by importance on a small sample and fit bundle.selector.

    Uses a temporary FeatureState so the probe does not contaminate the bundle's
    running statistics.  Called only during --bootstrap, before any training,
    so that model.feature_names locks to the selected subset on its first
    partial_fit call.

    If feature_selection.enabled is False in config.yaml, this is a no-op.
    """
    fs_cfg   = cfg.get("feature_selection", {})
    if not fs_cfg.get("enabled", True):
        log.info("feature selection disabled in config; using all features")
        return

    top_k     = int(fs_cfg.get("top_k",      20))
    probe_n   = int(fs_cfg.get("probe_rows", 5_000))

    log.info("feature selection probe: sampling up to %d rows...", probe_n)
    probe_raw = store.fetch_new_rows(0, probe_n)
    if probe_raw.empty:
        log.warning("no rows available for probe; skipping feature selection")
        return

    probe_ok, _ = clean(probe_raw, cfg)
    if probe_ok.empty:
        log.warning("probe rows all rejected by validation; skipping feature selection")
        return

    # Temporary state — does not touch bundle.state
    probe_state = FeatureState(cfg)
    probe_X, probe_y = feature_stream(probe_ok, probe_state)   # no selector yet

    fitted = bundle.selector.fit_from_probe(probe_X, probe_y, top_k=top_k)
    if fitted:
        bundle.meta["feature_selection"] = {
            "enabled":    True,
            "top_k":      top_k,
            "n_selected": len(bundle.selector.selected),
            "selected":   bundle.selector.selected,
            "importances": bundle.selector.importances,
        }
        log.info("feature selection complete: %d features selected for training",
                 len(bundle.selector.selected))
    else:
        log.warning("feature selection probe failed; training will use all features")


def run(bootstrap: bool = False, csv: str | None = None,
        dry_run: bool = False, max_rows: int | None = None) -> dict:
    cfg   = load_config()
    store = open_store(cfg, csv)
    store.ensure_schema()
    t0 = time.time()

    with store.lock():
        loaded = None if bootstrap else store.load_bundle_bytes()
        if loaded is None:
            if not bootstrap and store.get_watermark() > 0:
                raise RuntimeError(
                    "watermark exists but no model bundle found; re-run with --bootstrap")
            bundle, watermark = _starting_bundle(cfg, bootstrap), 0
        else:
            bundle, watermark = Bundle.loads(loaded[1]), store.get_watermark()
            log.info("loaded bundle v%s, watermark id=%s", loaded[0], watermark)

        # ── Data detection guard (skip training if no new data) ──────────────
        # Only check for normal runs; --bootstrap always proceeds (replays history)
        if not bootstrap:
            new_row_count = store.count_new_rows(watermark)
            if new_row_count == 0:
                elapsed = round(time.time() - t0, 1)
                log.info("no new data, skipped incremental training (elapsed=%.1fs)", elapsed)
                return {"status": "no_new_rows", "seconds": elapsed}
            log.info("detected %d new rows beyond watermark; proceeding with training", new_row_count)

        # ── Feature selection probe (bootstrap only) ──────────────────────────
        # Must run BEFORE the main training loop so that model.feature_names
        # locks to the selected subset on the model's first partial_fit call.
        if bootstrap:
            _run_feature_selection_probe(bundle, store, cfg)

        totals  = {"rows_in": 0, "rows_ok": 0, "rows_rejected": 0,
                   "rows_learned": 0, "drift_events": 0}
        reasons: dict[str, int] = {}
        batch   = cfg["data"]["batch_size"]
        stats:  dict = {}

        while True:
            raw = store.fetch_new_rows(watermark, batch)
            if raw.empty:
                break
            ok, bad = clean(raw, cfg)
            watermark = int(raw["id"].max())
            totals["rows_in"]       += len(raw)
            totals["rows_ok"]       += len(ok)
            totals["rows_rejected"] += len(bad)
            for r, n in bad["reason"].value_counts().items():
                reasons[r] = reasons.get(r, 0) + int(n)
            stats = apply_rows(bundle, ok, cfg)   # selector applied inside pipeline.py
            totals["rows_learned"]  += stats.get("rows_learned", 0)
            totals["drift_events"]  += stats.get("drift_events", 0)
            log.info("processed %s rows as one batch (watermark id=%s)", len(raw), watermark)
            if len(raw) < batch or (max_rows and totals["rows_in"] >= max_rows):
                break

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
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", action="store_true",
                    help="start from the notebook's trained model and replay all history; "
                         "also runs the feature-selection probe")
    ap.add_argument("--csv",      help="read rows from this CSV instead of Neon")
    ap.add_argument("--dry-run",  action="store_true", help="process but do not save")
    ap.add_argument("--max-rows", type=int,
                    help="stop after about this many rows (for quick trials)")
    a = ap.parse_args()
    run(bootstrap=a.bootstrap, csv=a.csv, dry_run=a.dry_run, max_rows=a.max_rows)


if __name__ == "__main__":
    main()
